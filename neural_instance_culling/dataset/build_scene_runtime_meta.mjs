#!/usr/bin/env node
/* Build glbIndex.json and runtimeVisibilityMeta.json from an SLM scene package.
 *
 * The current neural PVS pipeline needs dense component AABBs and a
 * component->global-GLB mapping. New scene packages usually only contain
 * sceneWeb.json plus task-N/glb/LOD0/sub_*.glb files, so this script extracts
 * the missing runtime metadata in the same semantic shape used by hkust-v3.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const THREE_URL = pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'build', 'three.module.js')).href;
const GLTF_LOADER_URL = pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'loaders', 'GLTFLoader.js')).href;
const DRACO_LOADER_URL = pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'loaders', 'DRACOLoader.js')).href;
const MESHOPT_URL = pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'libs', 'meshopt_decoder.module.js')).href;

if (!globalThis.self) globalThis.self = globalThis;

function parseArgs(argv) {
  const args = {
    sceneRoot: '',
    assetsDir: '',
    sceneName: '',
    outputRuntimeMeta: '',
    outputGlbIndex: '',
    overwrite: false,
    progressEvery: 100,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    if (key === '--overwrite') {
      args.overwrite = true;
      continue;
    }
    const value = argv[i + 1];
    i += 1;
    const name = key.slice(2);
    if (name === 'scene-root') args.sceneRoot = path.resolve(value);
    else if (name === 'assets-dir') args.assetsDir = path.resolve(value);
    else if (name === 'scene-name') args.sceneName = value;
    else if (name === 'runtime-meta') args.outputRuntimeMeta = path.resolve(value);
    else if (name === 'glb-index') args.outputGlbIndex = path.resolve(value);
    else if (name === 'progress-every') args.progressEvery = Number(value);
  }
  if (!args.assetsDir) {
    if (!args.sceneRoot) throw new Error('Expected --scene-root or --assets-dir.');
    args.assetsDir = path.join(args.sceneRoot, 'assets');
  }
  if (!args.sceneName) args.sceneName = path.basename(path.dirname(args.assetsDir));
  if (!args.outputRuntimeMeta) args.outputRuntimeMeta = path.join(args.assetsDir, 'runtimeVisibilityMeta.json');
  if (!args.outputGlbIndex) args.outputGlbIndex = path.join(args.assetsDir, 'glbIndex.json');
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function listGlbFiles(assetsDir) {
  const taskDirs = fs.readdirSync(assetsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^task-\d+$/.test(entry.name))
    .sort((a, b) => Number(a.name.slice(5)) - Number(b.name.slice(5)));
  const out = [];
  for (const taskDir of taskDirs) {
    const taskId = Number(taskDir.name.slice(5));
    const glbDir = path.join(assetsDir, taskDir.name, 'glb', 'LOD0');
    if (!fs.existsSync(glbDir)) continue;
    for (const file of fs.readdirSync(glbDir).sort()) {
      const match = /^sub_(\d+)\.glb$/i.exec(file);
      if (!match) continue;
      const baseId = Number(match[1]);
      out.push({
        taskId,
        baseId,
        hash: `${taskId}-${baseId}`,
        path: `${taskDir.name}/glb/LOD0/${file}`,
        absPath: path.join(glbDir, file),
      });
    }
  }
  out.sort((a, b) => (a.taskId - b.taskId) || (a.baseId - b.baseId));
  return out;
}

function buildComponentGroups(sceneWeb, glbEntries) {
  const byHash = new Map();
  for (const entry of glbEntries) {
    byHash.set(entry.hash, { entry, componentIds: [] });
  }
  let componentCount = 0;
  const missing = [];
  for (let groupIndex = 0; groupIndex < (sceneWeb.groups || []).length; groupIndex += 1) {
    const group = sceneWeb.groups[groupIndex];
    const range = group.idRange || [0, -1];
    const start = Number(range[0]);
    const end = Number(range[1]);
    componentCount = Math.max(componentCount, end + 1);
    const instanceMap = group.instances || {};
    for (let localId = 0; localId <= end - start; localId += 1) {
      const baseId = Object.prototype.hasOwnProperty.call(instanceMap, String(localId))
        ? Number(instanceMap[String(localId)])
        : localId;
      const hash = `${groupIndex}-${baseId}`;
      const bucket = byHash.get(hash);
      if (!bucket) {
        missing.push({ componentGlobalId: start + localId, hash, baseId, groupIndex });
        continue;
      }
      bucket.componentIds.push(start + localId);
    }
  }
  for (const bucket of byHash.values()) {
    bucket.componentIds.sort((a, b) => a - b);
  }
  return { byHash, componentCount, missing };
}

function makeEmptyBounds() {
  return {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
  };
}

function includePoint(bounds, x, y, z) {
  bounds.min[0] = Math.min(bounds.min[0], x);
  bounds.min[1] = Math.min(bounds.min[1], y);
  bounds.min[2] = Math.min(bounds.min[2], z);
  bounds.max[0] = Math.max(bounds.max[0], x);
  bounds.max[1] = Math.max(bounds.max[1], y);
  bounds.max[2] = Math.max(bounds.max[2], z);
}

function includeBox(bounds, box) {
  includePoint(bounds, box.min.x, box.min.y, box.min.z);
  includePoint(bounds, box.max.x, box.max.y, box.max.z);
}

function isFiniteBounds(bounds) {
  return bounds.min.every(Number.isFinite) && bounds.max.every(Number.isFinite);
}

function finalizeBounds(bounds) {
  if (!isFiniteBounds(bounds)) return null;
  const min = bounds.min.map(Number);
  const max = bounds.max.map(Number);
  const center = min.map((v, i) => (v + max[i]) * 0.5);
  const size = min.map((v, i) => Math.max(1e-6, max[i] - v));
  return { min, max, center, size };
}

function expandBoundsToCorners(THREE, sourceBox, matrix, target) {
  const xs = [sourceBox.min.x, sourceBox.max.x];
  const ys = [sourceBox.min.y, sourceBox.max.y];
  const zs = [sourceBox.min.z, sourceBox.max.z];
  const p = new THREE.Vector3();
  for (const x of xs) {
    for (const y of ys) {
      for (const z of zs) {
        p.set(x, y, z).applyMatrix4(matrix);
        includePoint(target, p.x, p.y, p.z);
      }
    }
  }
}

async function loadGltf(loader, filePath) {
  const data = fs.readFileSync(filePath);
  const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return new Promise((resolve, reject) => {
    loader.parse(arrayBuffer, `${path.dirname(filePath).replace(/\\/g, '/')}/`, resolve, reject);
  });
}

function extractComponentBounds(THREE, gltf, componentIds) {
  const perComponent = new Map();
  const templateBounds = makeEmptyBounds();
  const instanceMatrix = new THREE.Matrix4();
  const finalMatrix = new THREE.Matrix4();
  const localBox = new THREE.Box3();

  const ensure = (componentId) => {
    let bounds = perComponent.get(componentId);
    if (!bounds) {
      bounds = makeEmptyBounds();
      perComponent.set(componentId, bounds);
    }
    return bounds;
  };

  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((object) => {
    if (!object.isMesh && !object.isInstancedMesh) return;
    const geometry = object.geometry;
    if (!geometry || !geometry.attributes || !geometry.attributes.position) return;
    if (!geometry.boundingBox) geometry.computeBoundingBox();
    localBox.copy(geometry.boundingBox);

    if (object.isInstancedMesh) {
      const count = Math.min(Number(object.count || 0), componentIds.length);
      for (let i = 0; i < count; i += 1) {
        object.getMatrixAt(i, instanceMatrix);
        finalMatrix.multiplyMatrices(object.matrixWorld, instanceMatrix);
        const bounds = ensure(componentIds[i]);
        expandBoundsToCorners(THREE, localBox, finalMatrix, bounds);
        expandBoundsToCorners(THREE, localBox, finalMatrix, templateBounds);
      }
    } else {
      const componentId = componentIds[0];
      if (componentId !== undefined) {
        const bounds = ensure(componentId);
        expandBoundsToCorners(THREE, localBox, object.matrixWorld, bounds);
      }
      expandBoundsToCorners(THREE, localBox, object.matrixWorld, templateBounds);
    }
  });

  const templateFinal = finalizeBounds(templateBounds);
  if (templateFinal) {
    for (const componentId of componentIds) {
      if (!perComponent.has(componentId)) {
        perComponent.set(componentId, {
          min: templateFinal.min.slice(),
          max: templateFinal.max.slice(),
        });
      }
    }
  }
  return { perComponent, templateBounds: templateFinal };
}

function sceneBoundsFromRecords(sceneWeb, records) {
  const configured = sceneWeb.config && sceneWeb.config.bounds;
  if (configured && configured.center && configured.size) return configured;
  const merged = makeEmptyBounds();
  for (const record of records) {
    const bounds = record.bounds || {};
    if (bounds.min && bounds.max) {
      includePoint(merged, bounds.min[0], bounds.min[1], bounds.min[2]);
      includePoint(merged, bounds.max[0], bounds.max[1], bounds.max[2]);
    } else if (bounds.center && bounds.size) {
      const min = bounds.center.map((v, i) => Number(v) - Number(bounds.size[i]) * 0.5);
      const max = bounds.center.map((v, i) => Number(v) + Number(bounds.size[i]) * 0.5);
      includePoint(merged, min[0], min[1], min[2]);
      includePoint(merged, max[0], max[1], max[2]);
    }
  }
  const final = finalizeBounds(merged);
  if (!final) return { center: [0, 0, 0], size: [1, 1, 1] };
  return { center: final.center, size: final.size };
}

async function main() {
  const args = parseArgs(process.argv);
  if (!args.overwrite) {
    for (const filePath of [args.outputRuntimeMeta, args.outputGlbIndex]) {
      if (fs.existsSync(filePath)) {
        throw new Error(`${filePath} already exists. Pass --overwrite to replace it.`);
      }
    }
  }

  const THREE = await import(THREE_URL);
  const { GLTFLoader } = await import(GLTF_LOADER_URL);
  const { DRACOLoader } = await import(DRACO_LOADER_URL);
  const { MeshoptDecoder } = await import(MESHOPT_URL);
  const sceneWebPath = path.join(args.assetsDir, 'sceneWeb.json');
  const sceneWeb = readJson(sceneWebPath);
  const rawGlbEntries = listGlbFiles(args.assetsDir);
  const glbEntries = rawGlbEntries.map((entry, index) => ({
    globalId: index,
    taskId: entry.taskId,
    baseId: entry.baseId,
    hash: entry.hash,
    path: entry.path,
    absPath: entry.absPath,
  }));
  const glbIdByHash = new Map(glbEntries.map((entry) => [entry.hash, entry.globalId]));
  const { byHash, componentCount, missing } = buildComponentGroups(sceneWeb, glbEntries);

  // Optional precomputed per-prototype bounds from conversionManifest.json. When
  // available, bound lookups are served from the manifest and the (very large)
  // GLB set is not re-decoded here.
  const manifestPath = path.join(args.assetsDir, 'conversionManifest.json');
  const manifestBoundsByBaseId = new Map();
  let manifestHits = 0;
  if (fs.existsSync(manifestPath)) {
    try {
      const manifest = readJson(manifestPath);
      for (const p of (manifest.prototypes || [])) {
        if (p && p.bounds && p.bounds.center && p.bounds.size && p.baseComponentId != null) {
          manifestBoundsByBaseId.set(Number(p.baseComponentId), p.bounds);
          manifestHits += 1;
        }
      }
      if (manifestHits > 0) console.log(`[scene-meta] using ${manifestHits} precomputed bounds from conversionManifest.json (skipping GLB decode where available)`);
    } catch (error) {
      console.warn(`[scene-meta] could not read conversionManifest.json bounds: ${String(error && error.message ? error.message : error)}`);
    }
  }

  const dracoLoader = new DRACOLoader().setDecoderPath(pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'libs', 'draco', 'gltf')).href + '/');
  const loader = new GLTFLoader().setDRACOLoader(dracoLoader).setMeshoptDecoder(MeshoptDecoder);

  const recordsById = new Array(componentCount);
  const globalGlbComponentIds = new Map();
  const globalGlbBounds = new Map();
  let decoded = 0;
  let failed = 0;
  let fallbackBounds = 0;

  for (const entry of glbEntries) {
    const bucket = byHash.get(entry.hash);
    const componentIds = bucket ? bucket.componentIds : [];
    if (componentIds.length === 0) continue;
    let extracted = null;
    let errorMessage = null;
    const precomputed = manifestBoundsByBaseId.get(Number(entry.baseId));
    if (precomputed) {
      // Use the manifest prototype AABB directly; no GLB decode needed for this entry.
      // Store as a {min, max} pair so the shared finalizeBounds() path below produces the
      // standard {center, size, min, max} shape identical to the GLB-decode branch.
      const min = [
        Number(precomputed.center[0]) - Number(precomputed.size[0]) * 0.5,
        Number(precomputed.center[1]) - Number(precomputed.size[1]) * 0.5,
        Number(precomputed.center[2]) - Number(precomputed.size[2]) * 0.5,
      ];
      const max = [
        Number(precomputed.center[0]) + Number(precomputed.size[0]) * 0.5,
        Number(precomputed.center[1]) + Number(precomputed.size[1]) * 0.5,
        Number(precomputed.center[2]) + Number(precomputed.size[2]) * 0.5,
      ];
      const perComponent = new Map();
      for (const componentId of componentIds) perComponent.set(componentId, { min, max });
      extracted = { perComponent, templateBounds: { min, max } };
    } else {
      try {
        const gltf = await loadGltf(loader, entry.absPath);
        extracted = extractComponentBounds(THREE, gltf, componentIds);
        decoded += 1;
      } catch (error) {
        errorMessage = String(error && error.message ? error.message : error);
        failed += 1;
        extracted = { perComponent: new Map(), templateBounds: null };
      }
    }

    const glbComponentIds = [];
    const glbBounds = makeEmptyBounds();
    for (const componentId of componentIds) {
      const rawBounds = extracted.perComponent.get(componentId) || extracted.templateBounds;
      const bounds = rawBounds && rawBounds.center ? rawBounds : finalizeBounds(rawBounds || makeEmptyBounds());
      if (!bounds) {
        fallbackBounds += 1;
        continue;
      }
      includePoint(glbBounds, bounds.min[0], bounds.min[1], bounds.min[2]);
      includePoint(glbBounds, bounds.max[0], bounds.max[1], bounds.max[2]);
      recordsById[componentId] = {
        componentGlobalId: componentId,
        instanceId: componentId,
        globalGlbId: entry.globalId,
        taskId: entry.taskId,
        baseId: entry.baseId,
        glbHash: entry.hash,
        bounds: { center: bounds.center, size: bounds.size },
        primaryLeafChunkId: -1,
        overlapLeafChunkIds: [],
        representativeLeafChunkIds: [],
        rvcStats: {
          visiblePoses: 0,
          visibleRatio: 0,
          weightMean: 0,
          weightP90: 0,
          weightMax: 0,
          priorityPrior: 0,
        },
      };
      glbComponentIds.push(componentId);
    }
    globalGlbComponentIds.set(entry.globalId, glbComponentIds);
    const glbFinal = finalizeBounds(glbBounds);
    if (glbFinal) globalGlbBounds.set(entry.globalId, glbFinal);
    if (decoded % args.progressEvery === 0 || decoded + failed === glbEntries.length) {
      console.log(`[scene-meta] decoded=${decoded} failed=${failed} entry=${entry.hash}`);
    }
    if (errorMessage) {
      console.warn(`[scene-meta] failed ${entry.path}: ${errorMessage}`);
    }
  }

  const componentRecords = recordsById.filter(Boolean);
  const sceneBounds = sceneBoundsFromRecords(sceneWeb, componentRecords);
  const globalGlbRecords = glbEntries.map((entry) => {
    const bounds = globalGlbBounds.get(entry.globalId);
    return {
      globalGlbId: entry.globalId,
      taskId: entry.taskId,
      baseId: entry.baseId,
      glbHash: entry.hash,
      componentGlobalIds: globalGlbComponentIds.get(entry.globalId) || [],
      aabb: bounds ? { min: bounds.min, max: bounds.max } : { min: [0, 0, 0], max: [0, 0, 0] },
      rvcStats: {
        visiblePoses: 0,
        visibleRatio: 0,
        weightMean: 0,
        weightP90: 0,
        weightMax: 0,
        priorityPrior: 0,
      },
    };
  });
  const glbIndex = {
    version: 1,
    ordering: 'task-baseId',
    lod: 'LOD0',
    total: glbEntries.length,
    entries: glbEntries.map(({ absPath, ...entry }) => entry),
  };
  const runtimeMeta = {
    schemaVersion: 1,
    idSpaces: {
      componentGlobalId: 'sceneWeb group idRange component id',
      globalGlbId: 'dense glbIndex entry id',
    },
    sceneName: args.sceneName,
    sceneBounds,
    componentCount,
    globalGlbCount: glbEntries.length,
    instanceCount: componentCount,
    rvcStatsSummary: {
      source: 'none',
      note: 'Generated from sceneWeb.json and GLB geometry for Color-ID raster sampling.',
    },
    componentRecords,
    globalGlbRecords,
    buildStats: {
      sceneWeb: path.relative(REPO_ROOT, sceneWebPath),
      decodedGlbs: decoded,
      failedGlbs: failed,
      missingComponentMappings: missing.length,
      missingComponentMappingPreview: missing.slice(0, 20),
      fallbackBounds,
      componentRecords: componentRecords.length,
    },
  };

  fs.mkdirSync(path.dirname(args.outputGlbIndex), { recursive: true });
  fs.mkdirSync(path.dirname(args.outputRuntimeMeta), { recursive: true });
  fs.writeFileSync(args.outputGlbIndex, JSON.stringify(glbIndex, null, 2), 'utf8');
  fs.writeFileSync(args.outputRuntimeMeta, JSON.stringify(runtimeMeta, null, 2), 'utf8');
  console.log(JSON.stringify({
    sceneName: args.sceneName,
    glbIndex: args.outputGlbIndex,
    runtimeMeta: args.outputRuntimeMeta,
    componentCount,
    componentRecords: componentRecords.length,
    globalGlbCount: glbEntries.length,
    decoded,
    failed,
    missingComponentMappings: missing.length,
    fallbackBounds,
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
