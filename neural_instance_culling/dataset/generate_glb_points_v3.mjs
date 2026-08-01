#!/usr/bin/env node
/* 为 fixed PointNet / MSG-OPVS v4 生成 GLB 点云缓存。
 *
 * 每个唯一 GLB 生成固定数量的归一化表面点：
 *   glb_points.bin: header(16B) + Float32[numGlbs, pointsPerGlb, 3]
 *   glb_points_meta.json: globalGlbId、hash、path、bounds、fallback 统计
 *
 * 点云只用于训练离线几何编码器。最终前端只下发训练后的几何特征，
 * 不下发原始点云。
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const DEFAULT_ASSETS_DIR = path.join(REPO_ROOT, 'hkust-v3', 'assets');
const THREE_URL = pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'build', 'three.module.js')).href;
const GLTF_LOADER_URL = pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'loaders', 'GLTFLoader.js')).href;
const DRACO_LOADER_URL = pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'loaders', 'DRACOLoader.js')).href;
const MESHOPT_URL = pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'libs', 'meshopt_decoder.module.js')).href;

if (!globalThis.self) globalThis.self = globalThis;

function parseArgs(argv) {
  const args = {
    assetsDir: DEFAULT_ASSETS_DIR,
    glbIndex: path.join(DEFAULT_ASSETS_DIR, 'glbIndex.json'),
    output: path.join(REPO_ROOT, 'neural_instance_culling', 'dataset', 'out', 'glb_points_v3.bin'),
    meta: path.join(REPO_ROOT, 'neural_instance_culling', 'dataset', 'out', 'glb_points_v3_meta.json'),
    pointsPerGlb: 1024,
    maxGlbs: null,
    seed: 20260501,
    progressEvery: 50,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    const value = argv[i + 1];
    i += 1;
    const name = key.slice(2);
    if (name === 'assets-dir') args.assetsDir = path.resolve(value);
    else if (name === 'glb-index') args.glbIndex = path.resolve(value);
    else if (name === 'output') args.output = path.resolve(value);
    else if (name === 'meta') args.meta = path.resolve(value);
    else if (name === 'points-per-glb') args.pointsPerGlb = Number(value);
    else if (name === 'max-glbs') args.maxGlbs = Number(value);
    else if (name === 'seed') args.seed = Number(value);
    else if (name === 'progress-every') args.progressEvery = Number(value);
  }
  return args;
}

function createRng(seed) {
  let state = seed >>> 0;
  return function rng() {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function collectTriangles(THREE, object, instanceIndex = null) {
  const geometry = object.geometry;
  if (!geometry || !geometry.attributes || !geometry.attributes.position) return [];
  const pos = geometry.attributes.position;
  const idx = geometry.index;
  const meshMatrix = object.matrixWorld;
  const instanceMatrix = new THREE.Matrix4();
  const finalMatrix = new THREE.Matrix4();
  if (instanceIndex != null && object.isInstancedMesh) object.getMatrixAt(instanceIndex, instanceMatrix);
  finalMatrix.multiplyMatrices(meshMatrix, instanceIndex != null ? instanceMatrix : new THREE.Matrix4().identity());
  const triangles = [];
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  for (let t = 0; t < triCount; t += 1) {
    const ia = idx ? idx.getX(t * 3) : t * 3;
    const ib = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const ic = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(pos, ia).applyMatrix4(finalMatrix);
    b.fromBufferAttribute(pos, ib).applyMatrix4(finalMatrix);
    c.fromBufferAttribute(pos, ic).applyMatrix4(finalMatrix);
    ab.subVectors(b, a);
    ac.subVectors(c, a);
    const area = ab.cross(ac).length() * 0.5;
    if (area > 1e-12) {
      triangles.push({
        a: [a.x, a.y, a.z],
        b: [b.x, b.y, b.z],
        c: [c.x, c.y, c.z],
        area,
      });
    }
  }
  return triangles;
}

function sampleSurfacePoints(triangles, count, rng) {
  const points = new Float32Array(count * 3);
  // Some source GLBs are valid empty placeholders (materials/scene only).
  // They are not decode failures and should be represented by a zero feature
  // block with an explicit emptyGeometry flag rather than a silent fallback.
  if (!triangles.length) return { points, bounds: null, fallback: false, emptyGeometry: true };
  const cumulative = [];
  let total = 0;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const tri of triangles) {
    total += tri.area;
    cumulative.push(total);
    for (const p of [tri.a, tri.b, tri.c]) {
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], p[axis]);
        max[axis] = Math.max(max[axis], p[axis]);
      }
    }
  }
  for (let i = 0; i < count; i += 1) {
    const pick = rng() * total;
    let lo = 0;
    let hi = cumulative.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cumulative[mid] < pick) lo = mid + 1;
      else hi = mid;
    }
    const tri = triangles[lo];
    let u = rng();
    let v = rng();
    if (u + v > 1) {
      u = 1 - u;
      v = 1 - v;
    }
    const p = [
      tri.a[0] + (tri.b[0] - tri.a[0]) * u + (tri.c[0] - tri.a[0]) * v,
      tri.a[1] + (tri.b[1] - tri.a[1]) * u + (tri.c[1] - tri.a[1]) * v,
      tri.a[2] + (tri.b[2] - tri.a[2]) * u + (tri.c[2] - tri.a[2]) * v,
    ];
    for (let axis = 0; axis < 3; axis += 1) points[i * 3 + axis] = p[axis];
  }
  const center = min.map((v, axis) => (v + max[axis]) * 0.5);
  const size = min.map((v, axis) => Math.max(1e-6, max[axis] - v));
  const scale = Math.max(size[0], size[1], size[2], 1e-6);
  for (let i = 0; i < count; i += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      points[i * 3 + axis] = (points[i * 3 + axis] - center[axis]) / scale;
    }
  }
  return { points, bounds: { min, max, center, size, scale }, fallback: false, emptyGeometry: false };
}

async function loadGltf(loader, filePath) {
  const data = fs.readFileSync(filePath);
  const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return new Promise((resolve, reject) => {
    loader.parse(arrayBuffer, `${path.dirname(filePath).replace(/\\/g, '/')}/`, resolve, reject);
  });
}

async function main() {
  const args = parseArgs(process.argv);
  const THREE = await import(THREE_URL);
  const { GLTFLoader } = await import(GLTF_LOADER_URL);
  const { DRACOLoader } = await import(DRACO_LOADER_URL);
  const { MeshoptDecoder } = await import(MESHOPT_URL);
  const glbIndex = JSON.parse(fs.readFileSync(args.glbIndex, 'utf8'));
  const entries = Number.isFinite(args.maxGlbs) ? glbIndex.entries.slice(0, args.maxGlbs) : glbIndex.entries;
  // Compute numGlbs without spreading the full entries array (275809 entries overflows the call stack).
  const numGlbs = glbIndex.entries.reduce((acc, entry) => Math.max(acc, Number(entry.globalId)), -1) + 1;
  fs.mkdirSync(path.dirname(args.output), { recursive: true });
  const fd = fs.openSync(args.output, 'w');
  const header = Buffer.alloc(16);
  header.writeUInt32LE(numGlbs, 0);
  header.writeUInt32LE(args.pointsPerGlb, 4);
  header.writeUInt32LE(3, 8);
  header.writeUInt32LE(0, 12);
  fs.writeSync(fd, header);
  const zeroBlock = new Float32Array(args.pointsPerGlb * 3);
  for (let i = 0; i < numGlbs; i += 1) fs.writeSync(fd, Buffer.from(zeroBlock.buffer));
  fs.closeSync(fd);

  const dracoLoader = new DRACOLoader().setDecoderPath(pathToFileURL(path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'libs', 'draco', 'gltf')).href + '/');
  const loader = new GLTFLoader().setDRACOLoader(dracoLoader).setMeshoptDecoder(MeshoptDecoder);
  const rng = createRng(args.seed);
  const metaEntries = [];
  let decoded = 0;
  let failed = 0;
  let fallback = 0;
  let emptyGeometry = 0;
  const outFd = fs.openSync(args.output, 'r+');
  for (let entryIndex = 0; entryIndex < entries.length; entryIndex += 1) {
    const entry = entries[entryIndex];
    const globalId = Number(entry.globalId);
    const offset = 16 + globalId * args.pointsPerGlb * 3 * 4;
    let result = null;
    let errorMessage = null;
    try {
      const gltf = await loadGltf(loader, path.join(args.assetsDir, entry.path));
      gltf.scene.updateMatrixWorld(true);
      const triangles = [];
      gltf.scene.traverse((object) => {
        if (!object.isMesh && !object.isInstancedMesh) return;
        if (object.isInstancedMesh) {
          for (let instanceIndex = 0; instanceIndex < object.count; instanceIndex += 1) {
            const collected = collectTriangles(THREE, object, instanceIndex);
            for (const triangle of collected) triangles.push(triangle);
          }
        } else {
          const collected = collectTriangles(THREE, object, null);
          for (const triangle of collected) triangles.push(triangle);
        }
      });
      result = sampleSurfacePoints(triangles, args.pointsPerGlb, rng);
      decoded += 1;
      if (result.fallback) fallback += 1;
      if (result.emptyGeometry) emptyGeometry += 1;
    } catch (error) {
      failed += 1;
      fallback += 1;
      errorMessage = String(error && error.message ? error.message : error);
      result = { points: zeroBlock, bounds: null, fallback: true, emptyGeometry: false };
    }
    fs.writeSync(outFd, Buffer.from(result.points.buffer, result.points.byteOffset, result.points.byteLength), 0, result.points.byteLength, offset);
    metaEntries.push({
      globalId,
      hash: entry.hash,
      path: entry.path,
      bounds: result.bounds,
      fallback: result.fallback,
      emptyGeometry: Boolean(result.emptyGeometry),
      error: errorMessage,
    });
    if (args.progressEvery > 0 && (entryIndex + 1) % args.progressEvery === 0) {
      console.log(`[glb-points-v3] ${entryIndex + 1}/${entries.length} decoded=${decoded} failed=${failed}`);
    }
  }
  fs.closeSync(outFd);
  dracoLoader.dispose();
  const meta = {
    output: args.output,
    pointsPerGlb: args.pointsPerGlb,
    numGlbs,
    decoded,
    failed,
    fallback,
    emptyGeometry,
    bytes: fs.statSync(args.output).size,
    entries: metaEntries,
  };
  fs.mkdirSync(path.dirname(args.meta), { recursive: true });
  fs.writeFileSync(args.meta, JSON.stringify(meta, null, 2), 'utf8');
  console.log(JSON.stringify({ ...meta, entries: undefined }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
