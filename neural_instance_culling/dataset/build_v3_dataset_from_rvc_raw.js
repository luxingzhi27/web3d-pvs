#!/usr/bin/env node
/* 将自适应 rvcServer raw JSONL 构建为 Instance-PVS v3 CSR 数据集。
 *
 * v3 与旧 directional 数据集的关键区别：
 *   - candidate_ids 保存“完整视锥候选集合”，不是少量随机 hard negative。
 *   - 每个 pose 保留 sample_category，便于按 street_gap / near_building 等类别评估。
 *   - candidate 生成口径尽量与前端 InstancePVS 的 AABB 视锥过滤一致，并强制并入 GT visible。
 */

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

const POSE_STRIDE_BYTES = 64;
const SPLIT_IDS = { train: 0, val: 1, test: 2 };
const SPLIT_NAMES = ['train', 'val', 'test'];
const CATEGORY_IDS = { street_gap: 0, near_building: 1, plaza: 2, perimeter: 3, sky: 4, far: 5, unknown: 255 };

function parseArgs(argv) {
  const args = {
    raw: 'neural_instance_culling/sampler/out/hkust_rvc_adaptive_v3_raw/samples_shard_00.jsonl',
    runtimeMeta: 'hkust-v3/assets/runtimeVisibilityMeta.json',
    outputDir: 'neural_instance_culling/dataset/out/pose_csr_rvc_adaptive_v3',
    seed: 20260501,
    maxRawRows: Infinity,
    minVisible: 1,
    frustumMarginDeg: 1.5,
    candidateFovScale: 1.18,
    candidateAspectScale: 1.08,
    maxCandidatesPerPose: 0,
    candidateMode: 'mvp',
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    const value = argv[i + 1];
    i += 1;
    const name = key.slice(2);
    if (name === 'raw') args.raw = path.resolve(value);
    else if (name === 'runtime-meta') args.runtimeMeta = path.resolve(value);
    else if (name === 'output-dir') args.outputDir = path.resolve(value);
    else if (name === 'seed') args.seed = Number(value);
    else if (name === 'max-raw-rows') args.maxRawRows = Number(value);
    else if (name === 'min-visible') args.minVisible = Number(value);
    else if (name === 'frustum-margin-deg') args.frustumMarginDeg = Number(value);
    else if (name === 'candidate-fov-scale') args.candidateFovScale = Number(value);
    else if (name === 'candidate-aspect-scale') args.candidateAspectScale = Number(value);
    else if (name === 'max-candidates-per-pose') args.maxCandidatesPerPose = Number(value);
    else if (name === 'candidate-mode') args.candidateMode = String(value || 'mvp');
  }
  return args;
}

function sceneMinMax(sceneBounds) {
  if (sceneBounds.min && sceneBounds.max) {
    return { min: sceneBounds.min.map(Number), max: sceneBounds.max.map(Number), size: sceneBounds.size.map(Number) };
  }
  const center = sceneBounds.center.map(Number);
  const size = sceneBounds.size.map(Number);
  return {
    min: center.map((v, i) => v - size[i] * 0.5),
    max: center.map((v, i) => v + size[i] * 0.5),
    size,
  };
}

function boundsMinMax(bounds) {
  if (bounds.min && bounds.max) {
    const min = bounds.min.map(Number);
    const max = bounds.max.map(Number);
    return { min, max, center: min.map((v, i) => (v + max[i]) * 0.5), size: min.map((v, i) => max[i] - v) };
  }
  const center = bounds.center.map(Number);
  const size = bounds.size.map(Number);
  return {
    min: center.map((v, i) => v - size[i] * 0.5),
    max: center.map((v, i) => v + size[i] * 0.5),
    center,
    size,
  };
}

function stableMix32(value, state) {
  let h = (state ^ (value >>> 0)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 2246822507) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 3266489909) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

function stableHashPose(worldPos, forward, seed) {
  let state = seed >>> 0;
  for (const value of worldPos) state = stableMix32(Math.round(Number(value || 0) * 20), state);
  for (const value of forward) state = stableMix32(Math.round(Number(value || 0) * 1000), state);
  return state >>> 0;
}

function splitForPose(worldPos, forward, seed) {
  const ratio = stableHashPose(worldPos, forward, seed) / 0x100000000;
  if (ratio < 0.8) return 'train';
  if (ratio < 0.9) return 'val';
  return 'test';
}

function normalizePoint(point, bounds) {
  return [
    (point[0] - bounds.min[0]) / Math.max(1e-6, bounds.size[0]),
    (point[1] - bounds.min[1]) / Math.max(1e-6, bounds.size[1]),
    (point[2] - bounds.min[2]) / Math.max(1e-6, bounds.size[2]),
  ].map((v) => Math.max(0, Math.min(1, v)));
}

function normalizeForward(forward) {
  const f = Array.isArray(forward) ? forward.map(Number) : [0, 0, -1];
  const len = Math.hypot(f[0], f[1], f[2]);
  if (!(len > 1e-6)) return [0, 0, -1];
  return [f[0] / len, f[1] / len, f[2] / len];
}

function loadComponentBounds(runtimeMeta) {
  const records = runtimeMeta.componentRecords || [];
  const numInstances = records.reduce((maxId, record) => Math.max(maxId, Number(record.componentGlobalId || 0)), -1) + 1;
  const bounds = new Array(numInstances);
  const instanceToGlb = new Int32Array(numInstances);
  instanceToGlb.fill(-1);
  for (const record of records) {
    const id = Number(record.componentGlobalId);
    if (!Number.isFinite(id) || id < 0) continue;
    const b = boundsMinMax(record.bounds);
    const size = b.size.map((v) => Math.max(0, Number(v) || 0));
    bounds[id] = {
      id,
      min: b.min,
      max: b.max,
      center: b.center,
      size,
      radius: Math.hypot(size[0], size[1], size[2]) * 0.5,
    };
    instanceToGlb[id] = Number(record.globalGlbId);
  }
  return { bounds, numInstances, instanceToGlb };
}

function expandBoundsFromPositions(sceneBounds, positions) {
  const min = [...sceneBounds.min];
  const max = [...sceneBounds.max];
  for (const pos of positions) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], pos[axis]);
      max[axis] = Math.max(max[axis], pos[axis]);
    }
  }
  const pad = [
    Math.max(1, (max[0] - min[0]) * 0.05),
    Math.max(1, (max[1] - min[1]) * 0.05),
    Math.max(1, (max[2] - min[2]) * 0.05),
  ];
  for (let axis = 0; axis < 3; axis += 1) {
    min[axis] -= pad[axis];
    max[axis] += pad[axis];
  }
  return { min, max, center: min.map((v, i) => (v + max[i]) * 0.5), size: min.map((v, i) => max[i] - v) };
}

function makeCameraBasis(forward) {
  const f = normalizeForward(forward);
  const upSeed = Math.abs(f[1]) > 0.98 ? [0, 0, 1] : [0, 1, 0];
  let rx = upSeed[1] * f[2] - upSeed[2] * f[1];
  let ry = upSeed[2] * f[0] - upSeed[0] * f[2];
  let rz = upSeed[0] * f[1] - upSeed[1] * f[0];
  const rLen = Math.hypot(rx, ry, rz) || 1;
  rx /= rLen; ry /= rLen; rz /= rLen;
  const ux = f[1] * rz - f[2] * ry;
  const uy = f[2] * rx - f[0] * rz;
  const uz = f[0] * ry - f[1] * rx;
  return { forward: f, right: [rx, ry, rz], up: [ux, uy, uz] };
}

function intersectsCandidateFrustum(bound, cameraPos, basis, tanHalfFovX, tanHalfFovY, marginDeg) {
  if (!bound) return false;
  const dx = bound.center[0] - cameraPos[0];
  const dy = bound.center[1] - cameraPos[1];
  const dz = bound.center[2] - cameraPos[2];
  const depth = dx * basis.forward[0] + dy * basis.forward[1] + dz * basis.forward[2];
  if (depth + bound.radius <= 0) return false;
  const dist = Math.max(1e-6, Math.hypot(dx, dy, dz));
  const angularRadius = Math.asin(Math.min(1, bound.radius / dist));
  const margin = Math.tan((Math.max(0, marginDeg) * Math.PI / 180) + angularRadius);
  const right = dx * basis.right[0] + dy * basis.right[1] + dz * basis.right[2];
  const up = dx * basis.up[0] + dy * basis.up[1] + dz * basis.up[2];
  const safeDepth = Math.max(1e-6, depth);
  return Math.abs(right / safeDepth) <= tanHalfFovX + margin
    && Math.abs(up / safeDepth) <= tanHalfFovY + margin;
}

function planesFromMvp(mvp) {
  if (!Array.isArray(mvp) || mvp.length < 16) return null;
  const me = mvp.map(Number);
  const raw = [
    [me[3] - me[0], me[7] - me[4], me[11] - me[8], me[15] - me[12]],
    [me[3] + me[0], me[7] + me[4], me[11] + me[8], me[15] + me[12]],
    [me[3] + me[1], me[7] + me[5], me[11] + me[9], me[15] + me[13]],
    [me[3] - me[1], me[7] - me[5], me[11] - me[9], me[15] - me[13]],
    [me[3] - me[2], me[7] - me[6], me[11] - me[10], me[15] - me[14]],
    [me[3] + me[2], me[7] + me[6], me[11] + me[10], me[15] + me[14]],
  ];
  return raw.map((p) => {
    const invLen = 1 / Math.max(1e-12, Math.hypot(p[0], p[1], p[2]));
    return [p[0] * invLen, p[1] * invLen, p[2] * invLen, p[3] * invLen];
  });
}

function intersectsMvpFrustum(bound, planes) {
  if (!bound || !planes) return false;
  for (const plane of planes) {
    const x = plane[0] >= 0 ? bound.max[0] : bound.min[0];
    const y = plane[1] >= 0 ? bound.max[1] : bound.min[1];
    const z = plane[2] >= 0 ? bound.max[2] : bound.min[2];
    if (plane[0] * x + plane[1] * y + plane[2] * z + plane[3] < 0) return false;
  }
  return true;
}

function candidateIdsForPose(sample, componentBounds, args) {
  if (args.candidateMode === 'mvp') {
    const planes = planesFromMvp(sample.mvp);
    if (planes) {
      const ids = [];
      for (let id = 0; id < componentBounds.length; id += 1) {
        if (intersectsMvpFrustum(componentBounds[id], planes)) ids.push(id >>> 0);
      }
      return ids;
    }
  }
  const cameraPos = (sample.camera_pos || [0, 0, 0]).map(Number);
  const forward = normalizeForward(sample.camera_forward);
  const fovY = Number(sample.fov_y || 66) * Math.max(1e-6, args.candidateFovScale);
  const aspect = Number(sample.aspect || 16 / 9) * Math.max(1e-6, args.candidateAspectScale);
  const tanHalfFovY = Math.tan((fovY * Math.PI / 180) * 0.5);
  const tanHalfFovX = tanHalfFovY * aspect;
  const basis = makeCameraBasis(forward);
  const ids = [];
  for (let id = 0; id < componentBounds.length; id += 1) {
    if (intersectsCandidateFrustum(componentBounds[id], cameraPos, basis, tanHalfFovX, tanHalfFovY, args.frustumMarginDeg)) {
      ids.push(id >>> 0);
    }
  }
  return ids;
}

function packPoses(poses) {
  const buffer = Buffer.alloc(poses.length * POSE_STRIDE_BYTES);
  for (let index = 0; index < poses.length; index += 1) {
    const pose = poses[index];
    const base = index * POSE_STRIDE_BYTES;
    for (let axis = 0; axis < 3; axis += 1) buffer.writeFloatLE(Number(pose.cameraNorm[axis] || 0), base + axis * 4);
    for (let axis = 0; axis < 3; axis += 1) buffer.writeFloatLE(Number(pose.cameraWorld[axis] || 0), base + 12 + axis * 4);
    for (let axis = 0; axis < 3; axis += 1) buffer.writeFloatLE(Number(pose.forward[axis] || 0), base + 24 + axis * 4);
    buffer.writeFloatLE(Number(pose.tanHalfFovX || 0), base + 36);
    buffer.writeFloatLE(Number(pose.tanHalfFovY || 0), base + 40);
    buffer.writeUInt8(pose.splitId & 0xff, base + 44);
    buffer.writeUInt8(pose.categoryId & 0xff, base + 45);
    buffer.writeUInt16LE(0, base + 46);
    buffer.writeUInt32LE(index >>> 0, base + 48);
  }
  return buffer;
}

function writeTypedArray(filePath, typedArray) {
  fs.writeFileSync(filePath, Buffer.from(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength));
}

function sortedUnique(values) {
  const arr = Array.from(values).map(Number).filter((v) => Number.isFinite(v) && v >= 0).sort((a, b) => a - b);
  const out = [];
  for (const v of arr) {
    if (out.length === 0 || out[out.length - 1] !== v) out.push(v >>> 0);
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv);
  const runtimeMeta = JSON.parse(fs.readFileSync(args.runtimeMeta, 'utf8'));
  const sceneBounds = sceneMinMax(runtimeMeta.sceneBounds);
  const { bounds: componentBounds, numInstances, instanceToGlb } = loadComponentBounds(runtimeMeta);

  const rawSamples = [];
  const positions = [];
  const rl = readline.createInterface({
    input: fs.createReadStream(args.raw, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });
  let rawRows = 0;
  let skippedRows = 0;
  for await (const line of rl) {
    if (!line.trim()) continue;
    if (rawRows >= args.maxRawRows) break;
    const sample = JSON.parse(line);
    rawRows += 1;
    const visibleIds = sample.visible_component_ids || [];
    if (visibleIds.length < args.minVisible) {
      skippedRows += 1;
      continue;
    }
    rawSamples.push(sample);
    positions.push((sample.camera_pos || [0, 0, 0]).map(Number));
    if (rawSamples.length % 10000 === 0) {
      console.log(`[v3-dataset] accepted=${rawSamples.length} raw=${rawRows}`);
    }
  }
  const cameraBounds = expandBoundsFromPositions(sceneBounds, positions);
  const poses = [];
  const visibleOffsets = [0];
  const visibleIds = [];
  const visiblePixels = [];
  const candidateOffsets = [0];
  fs.mkdirSync(args.outputDir, { recursive: true });
  const candidateIdsPath = path.join(args.outputDir, 'candidate_ids.bin');
  const candidateFd = fs.openSync(candidateIdsPath, 'w');
  const stats = {
    rawRows,
    skippedRows,
    poses: 0,
    totalVisibleRefs: 0,
    totalCandidateRefs: 0,
    splitPoseCounts: { train: 0, val: 0, test: 0 },
    categoryPoseCounts: {},
    candidateMissVisible: 0,
    avgVisiblePerPose: 0,
    avgCandidatePerPose: 0,
  };

  try {
  for (let index = 0; index < rawSamples.length; index += 1) {
    const sample = rawSamples[index];
    const cameraWorld = (sample.camera_pos || [0, 0, 0]).map(Number);
    const forward = normalizeForward(sample.camera_forward);
    const fovY = Number(sample.fov_y || 66);
    const aspect = Number(sample.aspect || 16 / 9);
    const tanHalfFovY = Math.tan((fovY * Math.PI / 180) * 0.5);
    const tanHalfFovX = tanHalfFovY * aspect;
    const split = splitForPose(cameraWorld, forward, args.seed);
    const category = sample.sample_category || 'unknown';
    const categoryId = Number.isFinite(Number(sample.sample_category_id))
      ? Number(sample.sample_category_id)
      : (CATEGORY_IDS[category] ?? CATEGORY_IDS.unknown);
    poses.push({
      cameraNorm: normalizePoint(cameraWorld, cameraBounds),
      cameraWorld,
      forward,
      tanHalfFovX,
      tanHalfFovY,
      splitId: SPLIT_IDS[split],
      categoryId,
    });

    const weights = sample.component_weights || [];
    const visible = sortedUnique(sample.visible_component_ids || []);
    const visibleWeightMap = new Map();
    for (let i = 0; i < (sample.visible_component_ids || []).length; i += 1) {
      const id = Number(sample.visible_component_ids[i]);
      const weight = Math.max(1, Math.round(Number(weights[i] || 1)));
      if (Number.isFinite(id) && id >= 0) visibleWeightMap.set(id >>> 0, Math.max(weight, visibleWeightMap.get(id >>> 0) || 0));
    }
    for (const id of visible) {
      visibleIds.push(id >>> 0);
      visiblePixels.push((visibleWeightMap.get(id >>> 0) || 1) >>> 0);
    }
    visibleOffsets.push(visibleIds.length);

    let candidates = candidateIdsForPose(sample, componentBounds, args);
    const beforeUnion = new Set(candidates);
    for (const id of visible) {
      if (!beforeUnion.has(id)) stats.candidateMissVisible += 1;
      beforeUnion.add(id);
    }
    candidates = Array.from(beforeUnion).sort((a, b) => a - b);
    if (args.maxCandidatesPerPose > 0 && candidates.length > args.maxCandidatesPerPose) {
      const visibleSet = new Set(visible);
      const negatives = candidates.filter((id) => !visibleSet.has(id));
      candidates = visible.concat(negatives.slice(0, Math.max(0, args.maxCandidatesPerPose - visible.length))).sort((a, b) => a - b);
    }
    const candidateArray = Uint32Array.from(candidates);
    fs.writeSync(candidateFd, Buffer.from(candidateArray.buffer, candidateArray.byteOffset, candidateArray.byteLength));
    candidateOffsets.push(candidateOffsets[candidateOffsets.length - 1] + candidates.length);

    stats.poses += 1;
    stats.totalVisibleRefs += visible.length;
    stats.totalCandidateRefs += candidates.length;
    stats.splitPoseCounts[split] += 1;
    stats.categoryPoseCounts[category] = (stats.categoryPoseCounts[category] || 0) + 1;
    if (stats.poses % 10000 === 0) {
      console.log(`[v3-dataset] packed poses=${stats.poses} avgCandidate=${(stats.totalCandidateRefs / stats.poses).toFixed(1)}`);
    }
  }
  } finally {
    fs.closeSync(candidateFd);
  }
  stats.avgVisiblePerPose = stats.poses > 0 ? stats.totalVisibleRefs / stats.poses : 0;
  stats.avgCandidatePerPose = stats.poses > 0 ? stats.totalCandidateRefs / stats.poses : 0;
  if (stats.totalCandidateRefs < stats.totalVisibleRefs) {
    throw new Error('Invalid dataset: candidate refs fewer than visible refs.');
  }

  writeTypedArray(path.join(args.outputDir, 'poses.bin'), new Uint8Array(packPoses(poses)));
  writeTypedArray(path.join(args.outputDir, 'visible_offsets.bin'), BigUint64Array.from(visibleOffsets, (v) => BigInt(v)));
  writeTypedArray(path.join(args.outputDir, 'visible_ids.bin'), Uint32Array.from(visibleIds));
  writeTypedArray(path.join(args.outputDir, 'visible_pixels.bin'), Uint32Array.from(visiblePixels));
  writeTypedArray(path.join(args.outputDir, 'candidate_offsets.bin'), BigUint64Array.from(candidateOffsets, (v) => BigInt(v)));
  writeTypedArray(path.join(args.outputDir, 'instance_to_glb.bin'), instanceToGlb);

  const firstSample = rawSamples[0] || {};
  const meta = {
    version: 3,
    format: 'pose-csr-instance-visibility-adaptive-v3',
    rawSampler: firstSample.sampler || 'rvc_or_legacy',
    poseStrideBytes: POSE_STRIDE_BYTES,
    numPoses: poses.length,
    numInstances,
    sceneBounds: runtimeMeta.sceneBounds,
    cameraBounds,
    splitIds: SPLIT_IDS,
    splitNames: SPLIT_NAMES,
    categoryIds: CATEGORY_IDS,
    candidateSemantics: 'full enlarged AABB frustum candidates plus all sampled visible positives',
    visibleWeightSemantics: firstSample.weight_semantics || 'legacy_visible_weight',
    visibleWeightScale: Number(firstSample.weight_scale || 1),
    visiblePixelsBinSemantics: firstSample.weight_semantics
      ? 'per-visible-component weight values, not raw pixel counts'
      : 'legacy visible weight values',
    totalVisibleRefs: stats.totalVisibleRefs,
    totalCandidateRefs: stats.totalCandidateRefs,
    args,
  };
  fs.writeFileSync(path.join(args.outputDir, 'dataset_meta.json'), JSON.stringify(meta, null, 2), 'utf8');
  fs.writeFileSync(path.join(args.outputDir, 'dataset_summary.json'), JSON.stringify(stats, null, 2), 'utf8');
  console.log(JSON.stringify({ outputDir: args.outputDir, ...stats }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
