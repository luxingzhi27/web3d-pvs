import fs from 'node:fs';
import path from 'node:path';

import { readMonolithicGlb, writePrototypeGlb } from './glb.mjs';
import { describeRegions, extractRegionGeometry, groupSimilarRegions } from './rigid.mjs';

function readJson(filePath, fallback = null) {
  return fs.existsSync(filePath) ? JSON.parse(fs.readFileSync(filePath, 'utf8')) : fallback;
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function readSourceEntries(sourceAssets, lod) {
  const index = readJson(path.join(sourceAssets, 'glbIndex.json'));
  if (index?.entries?.length) {
    return index.entries.map((entry) => ({
      componentId: Number(entry.globalId),
      source: path.join(
        sourceAssets,
        entry.path || `task-${entry.taskId}/glb/${lod}/sub_${entry.baseId}.glb`,
      ),
    })).sort((a, b) => a.componentId - b.componentId);
  }
  const glbDir = path.join(sourceAssets, 'task-0', 'glb', lod);
  return fs.readdirSync(glbDir)
    .filter((name) => /^sub_\d+\.glb$/.test(name))
    .map((name) => ({
      componentId: Number(name.slice(4, -4)),
      source: path.join(glbDir, name),
    }))
    .sort((a, b) => a.componentId - b.componentId);
}

function loadComponentMesh(entries, progressEvery) {
  const parts = [];
  let totalVertices = 0;
  let totalTriangles = 0;
  let inputGlbBytes = 0;
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const part = readMonolithicGlb(entry.source);
    parts.push({ ...entry, mesh: part });
    totalVertices += part.vertexCount;
    totalTriangles += part.triangleCount;
    inputGlbBytes += part.bytes;
    if (progressEvery > 0 && (index + 1) % progressEvery === 0) {
      console.log(`[glb-instancer] loaded sub glbs=${index + 1}/${entries.length} vertices=${totalVertices} triangles=${totalTriangles}`);
    }
  }

  const positions = new Float32Array(totalVertices * 3);
  const colors = new Uint8Array(totalVertices * 4);
  const indices = new Uint32Array(totalTriangles * 3);
  const verticesByRegion = new Uint32Array(totalVertices);
  const trianglesByRegion = new Uint32Array(totalTriangles);
  const regions = [];
  let vertexOffset = 0;
  let triangleOffset = 0;
  for (const part of parts) {
    positions.set(part.mesh.positions, vertexOffset * 3);
    colors.set(part.mesh.colors, vertexOffset * 4);
    for (let local = 0; local < part.mesh.vertexCount; local += 1) {
      verticesByRegion[vertexOffset + local] = vertexOffset + local;
    }
    for (let local = 0; local < part.mesh.triangleCount; local += 1) {
      trianglesByRegion[triangleOffset + local] = triangleOffset + local;
      for (let corner = 0; corner < 3; corner += 1) {
        indices[(triangleOffset + local) * 3 + corner] = vertexOffset + part.mesh.indices[local * 3 + corner];
      }
    }
    regions.push({
      id: part.componentId,
      vertexOffset,
      vertexCount: part.mesh.vertexCount,
      triangleOffset,
      triangleCount: part.mesh.triangleCount,
      source: part.source,
    });
    vertexOffset += part.mesh.vertexCount;
    triangleOffset += part.mesh.triangleCount;
  }
  return {
    mesh: {
      filePath: '',
      bytes: inputGlbBytes,
      positions,
      colors,
      indices,
      vertexCount: totalVertices,
      triangleCount: totalTriangles,
    },
    connectivity: { regions, verticesByRegion, trianglesByRegion, unusedVertexCount: 0 },
    inputGlbBytes,
  };
}

function sceneBoundsFromDescriptors(descriptors) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const descriptor of descriptors) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], descriptor.bounds.min[axis]);
      max[axis] = Math.max(max[axis], descriptor.bounds.max[axis]);
    }
  }
  return {
    min,
    max,
    center: min.map((value, axis) => (value + max[axis]) * 0.5),
    size: min.map((value, axis) => Math.max(1e-6, max[axis] - value)),
  };
}

function boundsCenterSize(bounds) {
  return {
    center: bounds.min.map((value, axis) => (value + bounds.max[axis]) * 0.5),
    size: bounds.min.map((value, axis) => Math.max(1e-6, bounds.max[axis] - value)),
  };
}

function includeBounds(target, source) {
  for (let axis = 0; axis < 3; axis += 1) {
    target.min[axis] = Math.min(target.min[axis], source.min[axis]);
    target.max[axis] = Math.max(target.max[axis], source.max[axis]);
  }
}

function zeroRvcStats() {
  return { visiblePoses: 0, visibleRatio: 0, weightMean: 0, weightP90: 0, weightMax: 0, priorityPrior: 0 };
}

function summarizeGroups(groups, inputBytes) {
  const histogram = {};
  let reusedRegionCount = 0;
  let repeatedPrototypeCount = 0;
  let largestInstanceCount = 0;
  for (const group of groups) {
    const count = group.members.length;
    histogram[String(count)] = (histogram[String(count)] || 0) + 1;
    largestInstanceCount = Math.max(largestInstanceCount, count);
    if (count > 1) {
      repeatedPrototypeCount += 1;
      reusedRegionCount += count - 1;
    }
  }
  return {
    componentCount: groups.reduce((sum, group) => sum + group.members.length, 0),
    prototypeCount: groups.length,
    repeatedPrototypeCount,
    singletonPrototypeCount: groups.length - repeatedPrototypeCount,
    reusedComponentCount: reusedRegionCount,
    reusedRegionCount,
    reuseRatio: groups.length ? reusedRegionCount / (groups.length + reusedRegionCount) : 0,
    largestInstanceCount,
    instanceCountHistogram: histogram,
    inputGlbBytes: inputBytes,
  };
}

function prepareOutput(outputAssets, overwrite) {
  if (fs.existsSync(outputAssets)) {
    if (!overwrite) throw new Error(`output exists; pass --overwrite: ${outputAssets}`);
    fs.rmSync(outputAssets, { recursive: true, force: true });
  }
  fs.mkdirSync(path.join(outputAssets, 'task-0', 'glb', 'LOD0'), { recursive: true });
}

function copyProxyAssets(sourceAssets, outputAssets) {
  const source = path.join(sourceAssets, 'task-0', 'proxy');
  if (fs.existsSync(source)) fs.cpSync(source, path.join(outputAssets, 'task-0', 'proxy'), { recursive: true });
  const sourceImages = path.join(sourceAssets, 'task-0', 'images');
  const outputImages = path.join(outputAssets, 'task-0', 'images');
  if (fs.existsSync(sourceImages)) fs.cpSync(sourceImages, outputImages, { recursive: true });
  const imageLod = path.join(outputImages, 'image_lod.json');
  if (!fs.existsSync(imageLod)) writeJson(imageLod, {});
}

export async function buildSubGlbInstancedScene(options) {
  const startedAt = Date.now();
  const entries = readSourceEntries(options.sourceAssets, options.lod || 'LOD0');
  if (!entries.length) throw new Error(`no sub GLBs found under ${options.sourceAssets}`);
  const loaded = loadComponentMesh(entries, options.regionProgressEvery);
  console.log(`[glb-instancer] source components=${entries.length} vertices=${loaded.mesh.vertexCount} triangles=${loaded.mesh.triangleCount} bytes=${loaded.inputGlbBytes}`);
  const descriptors = describeRegions(loaded.mesh, loaded.connectivity, {
    fingerprintTolerance: options.fingerprintTolerance,
    progressEvery: options.regionProgressEvery,
  });
  const groups = groupSimilarRegions(loaded.mesh, loaded.connectivity, descriptors, {
    rmsTolerance: options.rmsTolerance,
    maxTolerance: options.maxTolerance,
    progressEvery: options.regionProgressEvery,
  });
  const sceneBounds = sceneBoundsFromDescriptors(descriptors);
  const summary = summarizeGroups(groups, loaded.inputGlbBytes);
  const report = {
    schemaVersion: 1,
    generatedBy: 'slm-glb-instancer 0.2.0',
    mode: 'sub-glb-components',
    sourceAssets: options.sourceAssets,
    sourceComponentSemantics: 'Each original sub_*.glb remains one component and one stable component ID.',
    sourceVertexCount: loaded.mesh.vertexCount,
    sourceTriangleCount: loaded.mesh.triangleCount,
    ...summary,
    sceneBounds,
    tolerances: {
      fingerprint: options.fingerprintTolerance,
      rms: options.rmsTolerance,
      max: options.maxTolerance,
    },
    elapsedSec: Number(((Date.now() - startedAt) / 1000).toFixed(3)),
  };
  if (options.analyzeOnly) {
    if (options.report) writeJson(options.report, report);
    return report;
  }

  prepareOutput(options.outputAssets, options.overwrite);
  copyProxyAssets(options.sourceAssets, options.outputAssets);
  const sourceSceneWeb = readJson(path.join(options.sourceAssets, 'sceneWeb.json'), {});
  const sourceRuntime = readJson(path.join(options.sourceAssets, 'runtimeVisibilityMeta.json'), {});
  const sourceManifest = readJson(path.join(options.sourceAssets, 'conversionManifest.json'), {});
  const sourceComponentsById = new Map(
    (sourceRuntime.componentRecords || []).map((record) => [Number(record.componentGlobalId), record]),
  );
  const sortedGroups = groups.slice().sort((a, b) => a.prototype.id - b.prototype.id);
  const sceneInstances = {};
  const glbEntries = [];
  const componentPlacement = new Map();
  const prototypeRecords = [];
  let outputGlbBytes = 0;

  for (let globalGlbId = 0; globalGlbId < sortedGroups.length; globalGlbId += 1) {
    const group = sortedGroups[globalGlbId];
    group.members.sort((a, b) => a.descriptor.id - b.descriptor.id);
    const baseId = group.prototype.id;
    const relativePath = `task-0/glb/LOD0/sub_${baseId}.glb`;
    const outputPath = path.join(options.outputAssets, relativePath);
    const geometry = extractRegionGeometry(loaded.mesh, loaded.connectivity, group.prototype);
    const instances = group.members.map((member) => ({
      componentId: member.descriptor.id,
      translation: member.alignment.translation,
      rotation: member.alignment.rotation,
      rmsError: member.alignment.rmsError,
      maxError: member.alignment.maxError,
    }));
    writePrototypeGlb(outputPath, { name: `prototype_${baseId}`, ...geometry, instances });
    const outputBytes = fs.statSync(outputPath).size;
    outputGlbBytes += outputBytes;
    glbEntries.push({ globalId: globalGlbId, taskId: 0, baseId, hash: `0-${baseId}`, path: relativePath });
    for (const member of group.members) {
      const componentId = member.descriptor.id;
      componentPlacement.set(componentId, { globalGlbId, baseId, descriptor: member.descriptor });
      if (componentId !== baseId) sceneInstances[String(componentId)] = baseId;
    }
    prototypeRecords.push({ baseId, globalGlbId, output: relativePath, outputBytes, instances });
    if (options.regionProgressEvery > 0 && (globalGlbId + 1) % options.regionProgressEvery === 0) {
      console.log(`[glb-instancer] exported prototypes=${globalGlbId + 1}/${sortedGroups.length} bytes=${outputGlbBytes}`);
    }
  }

  const sourceGroups = sourceSceneWeb.groups || [];
  const minId = descriptors.reduce((value, item) => Math.min(value, item.id), Infinity);
  const maxId = descriptors.reduce((value, item) => Math.max(value, item.id), -Infinity);
  const sceneWeb = {
    ...sourceSceneWeb,
    groups: [{ idRange: [minId, maxId], instances: sceneInstances }],
    groupSemantics: 'Original sub-GLB component IDs; duplicate IDs map to a rigidly verified prototype component.',
    config: {
      ...(sourceSceneWeb.config || {}),
      sceneName: options.sceneName,
      sourceSceneName: sourceSceneWeb.config?.sceneName || path.basename(path.dirname(options.sourceAssets)),
      instancingSemantics: 'Complete sub-GLB components grouped by geometry/color fingerprint and rigid alignment verification.',
      bounds: { center: sceneBounds.center, size: sceneBounds.size },
      sourceGroupCount: sourceGroups.length,
    },
  };
  const glbIndex = { version: 1, ordering: 'task-baseId', lod: 'LOD0', total: glbEntries.length, entries: glbEntries };

  const componentRecords = descriptors.map((descriptor) => {
    const placement = componentPlacement.get(descriptor.id);
    const source = sourceComponentsById.get(descriptor.id) || {};
    return {
      ...source,
      componentGlobalId: descriptor.id,
      instanceId: descriptor.id,
      globalGlbId: placement.globalGlbId,
      taskId: 0,
      baseId: placement.baseId,
      glbHash: `0-${placement.baseId}`,
      bounds: source.bounds || boundsCenterSize(descriptor.bounds),
      rvcStats: source.rvcStats || zeroRvcStats(),
    };
  }).sort((a, b) => a.componentGlobalId - b.componentGlobalId);
  const globalGlbRecords = sortedGroups.map((group, globalGlbId) => {
    const aggregate = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    for (const member of group.members) includeBounds(aggregate, member.descriptor.bounds);
    const baseId = group.prototype.id;
    return {
      globalGlbId,
      taskId: 0,
      baseId,
      glbHash: `0-${baseId}`,
      componentGlobalIds: group.members.map((member) => member.descriptor.id),
      aabb: aggregate,
      rvcStats: zeroRvcStats(),
    };
  });
  const runtimeVisibilityMeta = {
    ...sourceRuntime,
    schemaVersion: sourceRuntime.schemaVersion || 1,
    idSpaces: {
      componentGlobalId: 'original source sub-GLB component id',
      globalGlbId: 'dense prototype glbIndex entry id',
    },
    sceneName: options.sceneName,
    sceneBounds: { center: sceneBounds.center, size: sceneBounds.size },
    componentCount: descriptors.length,
    globalGlbCount: sortedGroups.length,
    instanceCount: descriptors.length,
    componentRecords,
    globalGlbRecords,
    buildStats: {
      ...(sourceRuntime.buildStats || {}),
      sourceComponentCount: descriptors.length,
      prototypeCount: sortedGroups.length,
      reusedComponentCount: report.reusedComponentCount,
      outputGlbBytes,
    },
  };
  const manifest = {
    ...sourceManifest,
    ...report,
    outputAssets: options.outputAssets,
    sceneName: options.sceneName,
    sourceSceneName: sourceManifest.sceneName || sourceSceneWeb.config?.sceneName || null,
    outputGlbBytes,
    actualByteReductionRatio: loaded.inputGlbBytes > 0 ? 1 - outputGlbBytes / loaded.inputGlbBytes : 0,
    componentIdSemantics: 'Original sub-GLB component IDs are preserved without connected-component splitting.',
    prototypes: prototypeRecords,
  };

  writeJson(path.join(options.outputAssets, 'sceneWeb.json'), sceneWeb);
  writeJson(path.join(options.outputAssets, 'glbIndex.json'), glbIndex);
  writeJson(path.join(options.outputAssets, 'runtimeVisibilityMeta.json'), runtimeVisibilityMeta);
  writeJson(path.join(options.outputAssets, 'conversionManifest.json'), manifest);
  if (options.report) writeJson(options.report, manifest);
  return manifest;
}
