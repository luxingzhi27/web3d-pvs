import fs from 'node:fs';
import path from 'node:path';

import { extractConnectedRegions } from './connectivity.mjs';
import { readMonolithicGlb, writePrototypeGlb } from './glb.mjs';
import { describeRegions, extractRegionGeometry, groupSimilarRegions } from './rigid.mjs';

function sceneBounds(positions) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], positions[i + axis]);
      max[axis] = Math.max(max[axis], positions[i + axis]);
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
  let estimatedOutputBytes = 0;
  let repeatedPrototypeCount = 0;
  let reusedRegionCount = 0;
  let largestInstanceCount = 0;
  for (const group of groups) {
    const count = group.members.length;
    largestInstanceCount = Math.max(largestInstanceCount, count);
    histogram[String(count)] = (histogram[String(count)] || 0) + 1;
    if (count > 1) {
      repeatedPrototypeCount += 1;
      reusedRegionCount += count - 1;
    }
    estimatedOutputBytes += group.prototype.vertexCount * 16;
    estimatedOutputBytes += group.prototype.triangleCount * 12;
    if (count > 1) estimatedOutputBytes += count * 40;
    estimatedOutputBytes += 2048;
  }
  return {
    prototypeCount: groups.length,
    repeatedPrototypeCount,
    singletonPrototypeCount: groups.length - repeatedPrototypeCount,
    reusedRegionCount,
    largestInstanceCount,
    instanceCountHistogram: histogram,
    estimatedOutputGlbBytes: estimatedOutputBytes,
    estimatedByteReductionRatio: inputBytes > 0 ? 1 - estimatedOutputBytes / inputBytes : 0,
  };
}

function prepareOutput(outputAssets, overwrite) {
  if (fs.existsSync(outputAssets)) {
    if (!overwrite) throw new Error(`output exists; pass --overwrite: ${outputAssets}`);
    fs.rmSync(outputAssets, { recursive: true, force: true });
  }
  fs.mkdirSync(path.join(outputAssets, 'task-0', 'glb', 'LOD0'), { recursive: true });
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

export async function analyzeMonolithicGlb(options) {
  const startedAt = Date.now();
  const mesh = readMonolithicGlb(options.input);
  console.log(`[glb-instancer] input vertices=${mesh.vertexCount} triangles=${mesh.triangleCount} bytes=${mesh.bytes}`);
  const connectivity = extractConnectedRegions(mesh, { progressEvery: options.triangleProgressEvery });
  console.log(`[glb-instancer] connected regions=${connectivity.regions.length} unusedVertices=${connectivity.unusedVertexCount}`);
  const descriptors = describeRegions(mesh, connectivity, {
    fingerprintTolerance: options.fingerprintTolerance,
    progressEvery: options.regionProgressEvery,
  });
  const groups = groupSimilarRegions(mesh, connectivity, descriptors, {
    rmsTolerance: options.rmsTolerance,
    maxTolerance: options.maxTolerance,
    progressEvery: options.regionProgressEvery,
  });
  const bounds = sceneBounds(mesh.positions);
  const grouping = summarizeGroups(groups, mesh.bytes);
  const report = {
    schemaVersion: 1,
    generatedBy: 'slm-glb-instancer 0.1.0',
    input: options.input,
    inputBytes: mesh.bytes,
    sourceVertexCount: mesh.vertexCount,
    sourceTriangleCount: mesh.triangleCount,
    connectedRegionCount: connectivity.regions.length,
    unusedVertexCount: connectivity.unusedVertexCount,
    ...grouping,
    sceneBounds: bounds,
    tolerances: {
      fingerprint: options.fingerprintTolerance,
      rms: options.rmsTolerance,
      max: options.maxTolerance,
    },
    elapsedSec: Number(((Date.now() - startedAt) / 1000).toFixed(3)),
  };
  return { mesh, connectivity, descriptors, groups, report };
}

export async function buildInstancedScene(options) {
  const analysis = await analyzeMonolithicGlb(options);
  if (options.analyzeOnly) {
    if (options.report) writeJson(options.report, analysis.report);
    return analysis.report;
  }
  prepareOutput(options.outputAssets, options.overwrite);
  const { mesh, connectivity, groups, report } = analysis;
  const sortedGroups = groups.slice().sort((a, b) => a.prototype.id - b.prototype.id);
  const sceneInstances = {};
  const glbEntries = [];
  const componentToGlobalGlb = new Int32Array(connectivity.regions.length);
  const prototypeRecords = [];
  let outputGlbBytes = 0;

  for (let globalGlbId = 0; globalGlbId < sortedGroups.length; globalGlbId += 1) {
    const group = sortedGroups[globalGlbId];
    group.members.sort((a, b) => a.descriptor.id - b.descriptor.id);
    const baseId = group.members[0].descriptor.id;
    const relativePath = `task-0/glb/LOD0/sub_${baseId}.glb`;
    const outputPath = path.join(options.outputAssets, relativePath);
    const geometry = extractRegionGeometry(mesh, connectivity, group.prototype);
    const instances = group.members.map((member) => ({
      componentId: member.descriptor.id,
      translation: member.alignment.translation,
      rotation: member.alignment.rotation,
      rmsError: member.alignment.rmsError,
      maxError: member.alignment.maxError,
    }));
    writePrototypeGlb(outputPath, {
      name: `prototype_${baseId}`,
      ...geometry,
      instances,
    });
    const bytes = fs.statSync(outputPath).size;
    outputGlbBytes += bytes;
    glbEntries.push({ globalId: globalGlbId, taskId: 0, baseId, hash: `0-${baseId}`, path: relativePath });
    for (let memberIndex = 0; memberIndex < group.members.length; memberIndex += 1) {
      const componentId = group.members[memberIndex].descriptor.id;
      componentToGlobalGlb[componentId] = globalGlbId;
      if (componentId !== baseId) sceneInstances[String(componentId)] = baseId;
    }
    prototypeRecords.push({ baseId, globalGlbId, output: relativePath, outputBytes: bytes, instances });
    if (options.regionProgressEvery > 0 && (globalGlbId + 1) % options.regionProgressEvery === 0) {
      console.log(`[glb-instancer] exported=${globalGlbId + 1}/${sortedGroups.length} bytes=${outputGlbBytes}`);
    }
  }

  const sceneWeb = {
    materials: { proxy: [], useHdrJpg: false },
    groups: [{ idRange: [0, Math.max(0, connectivity.regions.length - 1)], instances: sceneInstances }],
    groupSemantics: 'connected-region instances: component IDs are triangle-connected regions; repeated IDs map to one prototype GLB',
    config: {
      sceneName: options.sceneName,
      source: path.basename(options.input),
      instancingSemantics: 'Connected regions grouped by rigid-transform-invariant fingerprint and Horn quaternion verification.',
      bounds: { center: report.sceneBounds.center, size: report.sceneBounds.size },
    },
  };
  const glbIndex = { version: 1, ordering: 'task-baseId', lod: 'LOD0', total: glbEntries.length, entries: glbEntries };

  const componentRecords = analysis.descriptors.map((descriptor) => {
    const globalGlbId = componentToGlobalGlb[descriptor.id];
    const entry = glbEntries[globalGlbId];
    return {
      componentGlobalId: descriptor.id,
      instanceId: descriptor.id,
      globalGlbId,
      taskId: 0,
      baseId: entry.baseId,
      glbHash: entry.hash,
      bounds: boundsCenterSize(descriptor.bounds),
      primaryLeafChunkId: -1,
      overlapLeafChunkIds: [],
      representativeLeafChunkIds: [],
      rvcStats: zeroRvcStats(),
    };
  });
  const globalGlbRecords = sortedGroups.map((group, globalGlbId) => {
    const aggregate = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    for (const member of group.members) includeBounds(aggregate, member.descriptor.bounds);
    const baseId = group.members[0].descriptor.id;
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
    schemaVersion: 1,
    idSpaces: {
      componentGlobalId: 'connected-region component id',
      globalGlbId: 'dense prototype glbIndex entry id',
    },
    sceneName: options.sceneName,
    sceneBounds: { center: report.sceneBounds.center, size: report.sceneBounds.size },
    componentCount: connectivity.regions.length,
    globalGlbCount: sortedGroups.length,
    instanceCount: connectivity.regions.length,
    rvcStatsSummary: {
      source: 'none',
      note: 'Generated directly from monolithic GLB connected-region analysis.',
    },
    componentRecords,
    globalGlbRecords,
    buildStats: {
      connectedRegionCount: connectivity.regions.length,
      prototypeCount: sortedGroups.length,
      reusedRegionCount: report.reusedRegionCount,
      outputGlbBytes,
    },
  };
  const manifest = {
    ...report,
    outputAssets: options.outputAssets,
    sceneName: options.sceneName,
    outputGlbBytes,
    actualByteReductionRatio: report.inputBytes > 0 ? 1 - outputGlbBytes / report.inputBytes : 0,
    componentIdSemantics: 'Each component is a triangle-connected region extracted from the monolithic input mesh.',
    prototypes: prototypeRecords,
  };

  writeJson(path.join(options.outputAssets, 'sceneWeb.json'), sceneWeb);
  writeJson(path.join(options.outputAssets, 'glbIndex.json'), glbIndex);
  writeJson(path.join(options.outputAssets, 'runtimeVisibilityMeta.json'), runtimeVisibilityMeta);
  writeJson(path.join(options.outputAssets, 'conversionManifest.json'), manifest);
  if (options.report) writeJson(options.report, manifest);
  return manifest;
}

