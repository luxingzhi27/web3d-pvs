#!/usr/bin/env node
/*
 * Compare the InstancePVS spatial AABB index with the reference full scan.
 * This is an offline correctness/performance audit; it does not change the
 * runtime max-query-cell policy or the model assets.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PerspectiveCamera, Vector3 } from 'three';
import { InstancePVS } from '../src/InstancePVS.js';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_DIR = path.resolve(SCRIPT_DIR, '..');
const PROJECT_DIR = path.resolve(VIEWER_DIR, '..');

const DEFAULTS = Object.freeze({
  modelMeta: path.join(
    VIEWER_DIR,
    'assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best/instance_model_meta.json',
  ),
  runtimeMeta: path.join(VIEWER_DIR, 'assets/scenes/hkust-v3/runtimeVisibilityMeta.json'),
  dataset: path.join(
    PROJECT_DIR,
    'neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1',
  ),
  samples: 128,
  far: 2000,
  indexMaxQueryCells: 1000000000,
});

function parseArgs(argv) {
  const options = {
    modelMeta: DEFAULTS.modelMeta,
    runtimeMeta: DEFAULTS.runtimeMeta,
    dataset: DEFAULTS.dataset,
    samples: DEFAULTS.samples,
    far: DEFAULTS.far,
    indexMaxQueryCells: DEFAULTS.indexMaxQueryCells,
    out: null,
    help: false,
  };
  const valueOptions = new Set([
    'model-meta', 'runtime-meta', 'dataset', 'samples', 'far', 'index-max-query-cells', 'out',
  ]);
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') {
      options.help = true;
      continue;
    }
    const equalIndex = token.indexOf('=');
    const rawName = equalIndex >= 0 ? token.slice(0, equalIndex) : token;
    const name = rawName.replace(/^--/, '');
    if (!valueOptions.has(name)) throw new Error(`Unknown argument: ${token}`);
    const value = equalIndex >= 0 ? token.slice(equalIndex + 1) : argv[++index];
    if (value == null || String(value).startsWith('--')) throw new Error(`Missing value for --${name}`);
    if (name === 'model-meta') options.modelMeta = path.resolve(String(value));
    else if (name === 'runtime-meta') options.runtimeMeta = path.resolve(String(value));
    else if (name === 'dataset') options.dataset = path.resolve(String(value));
    else if (name === 'samples') options.samples = Math.max(1, Math.trunc(Number(value)));
    else if (name === 'far') options.far = Math.max(1, Number(value));
    else if (name === 'index-max-query-cells') options.indexMaxQueryCells = Math.max(1, Number(value));
    else if (name === 'out') options.out = path.resolve(String(value));
  }
  if (!Number.isFinite(options.samples) || !Number.isFinite(options.far) ||
      !Number.isFinite(options.indexMaxQueryCells)) {
    throw new Error('samples, far and index-max-query-cells must be finite numbers');
  }
  return options;
}

function usage() {
  return `Usage: node scripts/benchmark_m9_spatial_index.mjs [options]

Compare the forced spatial index against the full AABB scan using real CSR poses.

  --model-meta PATH              exported InstancePVS model metadata
  --runtime-meta PATH            scene runtimeVisibilityMeta.json
  --dataset PATH                 pose CSR dataset containing poses.bin
  --samples N                    deterministic pose sample count (default ${DEFAULTS.samples})
  --far METERS                   camera far plane for the audit (default ${DEFAULTS.far})
  --index-max-query-cells N      temporary audit-only index limit (default ${DEFAULTS.indexMaxQueryCells})
  --out PATH                     write JSON report
`;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function sceneMinMax(bounds) {
  if (Array.isArray(bounds?.min) && Array.isArray(bounds?.max)) {
    return { min: bounds.min.map(Number), max: bounds.max.map(Number) };
  }
  const center = bounds.center.map(Number);
  const size = bounds.size.map(Number);
  return {
    min: center.map((value, axis) => value - size[axis] * 0.5),
    max: center.map((value, axis) => value + size[axis] * 0.5),
  };
}

function readPoseRows(datasetDir, poseCount) {
  const meta = readJson(path.join(datasetDir, 'dataset_meta.json'));
  const stride = Number(meta.poseStrideBytes || 64);
  if (stride !== 64) throw new Error(`Expected directional 64-byte poses.bin, got stride=${stride}`);
  const buffer = fs.readFileSync(path.join(datasetDir, 'poses.bin'));
  const count = Math.min(Math.floor(buffer.byteLength / stride), Number(poseCount));
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const rows = [];
  for (let index = 0; index < count; index += 1) {
    const base = index * stride;
    rows.push({
      position: [
        view.getFloat32(base + 12, true),
        view.getFloat32(base + 16, true),
        view.getFloat32(base + 20, true),
      ],
      forward: [
        view.getFloat32(base + 24, true),
        view.getFloat32(base + 28, true),
        view.getFloat32(base + 32, true),
      ],
      tanX: view.getFloat32(base + 36, true),
      tanY: view.getFloat32(base + 40, true),
    });
  }
  return { meta, rows };
}

function chooseRows(rows, sampleCount) {
  if (rows.length <= sampleCount) return rows;
  const selected = [];
  for (let index = 0; index < sampleCount; index += 1) {
    const sourceIndex = Math.floor(index * rows.length / sampleCount);
    selected.push(rows[sourceIndex]);
  }
  return selected;
}

function makeCamera(row, far) {
  const tanY = Math.max(1e-4, Number(row.tanY) || Math.tan(66 * Math.PI / 360));
  const tanX = Math.max(1e-4, Number(row.tanX) || tanY * 16 / 9);
  const camera = new PerspectiveCamera(66, tanX / tanY, 0.05, far);
  camera.position.fromArray(row.position);
  const direction = new Vector3().fromArray(row.forward).normalize();
  camera.lookAt(camera.position.clone().add(direction));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return camera;
}

function sortedIds(values) {
  return Array.from(new Set(values || [])).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
}

function fnv1a(ids) {
  let hash = 2166136261;
  for (const value of ids) {
    hash ^= (Number(value) >>> 0);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function percentile(values, probability) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function summarize(values) {
  return {
    sampleCount: values.length,
    p50: percentile(values, 0.50),
    p95: percentile(values, 0.95),
    p99: percentile(values, 0.99),
    min: values.length ? Math.min(...values) : null,
    max: values.length ? Math.max(...values) : null,
    mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
  };
}

function makePvs(modelMeta, runtimeMeta) {
  const pvs = new InstancePVS('', { inferenceFovYDeg: 66 });
  pvs.meta = modelMeta;
  pvs.sceneBounds = sceneMinMax(modelMeta.sceneBounds);
  pvs.cameraBounds = sceneMinMax(modelMeta.cameraBounds || modelMeta.sceneBounds);
  pvs._buildInstanceIdMaps();
  pvs.setRuntimeMeta(runtimeMeta);
  if (!pvs.spatialAabbIndex) throw new Error('Spatial AABB index was not built');
  return pvs;
}

function run(options) {
  const modelMeta = readJson(options.modelMeta);
  const runtimeMeta = readJson(options.runtimeMeta);
  const poseData = readPoseRows(options.dataset, Number.MAX_SAFE_INTEGER);
  const rows = chooseRows(poseData.rows, options.samples);
  const pvs = makePvs(modelMeta, runtimeMeta);
  const runtimeMaxQueryCells = pvs.spatialAabbIndex.maxQueryCells;
  const forcedIndexMaxQueryCells = options.indexMaxQueryCells;
  const indexCounts = [];
  const fullCounts = [];
  const indexTimes = [];
  const fullTimes = [];
  const mismatches = [];
  const sourceCounts = {};

  for (let index = 0; index < rows.length; index += 1) {
    const camera = makeCamera(rows[index], options.far);
    pvs.spatialAabbIndex.maxQueryCells = forcedIndexMaxQueryCells;
    const indexStart = performance.now();
    const indexed = sortedIds(pvs._frustumCandidateIds(camera));
    indexTimes.push(performance.now() - indexStart);
    const indexedSelection = { ...pvs.lastCandidateSelection };
    sourceCounts[indexedSelection.source] = (sourceCounts[indexedSelection.source] || 0) + 1;

    pvs.spatialAabbIndex.maxQueryCells = 1;
    const fullStart = performance.now();
    const full = sortedIds(pvs._frustumCandidateIds(camera));
    fullTimes.push(performance.now() - fullStart);
    if (fnv1a(indexed) !== fnv1a(full) || indexed.length !== full.length || indexed.some((id, i) => id !== full[i])) {
      mismatches.push({
        poseIndex: index,
        indexed: { count: indexed.length, hash: fnv1a(indexed) },
        full: { count: full.length, hash: fnv1a(full) },
        candidateSelection: indexedSelection,
      });
    }
    indexCounts.push(indexed.length);
    fullCounts.push(full.length);
  }

  pvs.spatialAabbIndex.maxQueryCells = runtimeMaxQueryCells;
  return {
    schemaVersion: 'm9-spatial-aabb-index-audit-v1',
    status: mismatches.length === 0 ? 'pass' : 'fail-set-mismatch',
    modelMeta: path.relative(PROJECT_DIR, options.modelMeta),
    runtimeMeta: path.relative(PROJECT_DIR, options.runtimeMeta),
    dataset: path.relative(PROJECT_DIR, options.dataset),
    configuration: {
      poseRowsAvailable: poseData.rows.length,
      sampledPoseCount: rows.length,
      cameraFovYDeg: 66,
      cameraFarM: options.far,
      runtimeMaxQueryCells,
      forcedIndexMaxQueryCells,
      cellSizeM: pvs.spatialAabbIndex.cellSize,
      indexedInstanceCount: pvs.spatialAabbIndex.indexedInstanceCount,
      overflowInstanceCount: pvs.spatialAabbIndex.overflowInstanceCount,
      bucketCount: pvs.spatialAabbIndex.buckets.size,
    },
    candidateCount: {
      forcedIndex: summarize(indexCounts),
      fullScan: summarize(fullCounts),
    },
    elapsedMs: {
      forcedIndex: summarize(indexTimes),
      fullScan: summarize(fullTimes),
    },
    sourceCounts,
    setComparison: {
      mismatches: mismatches.length,
      firstMismatches: mismatches.slice(0, 5),
    },
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }
  const report = run(options);
  const serialized = JSON.stringify(report, null, 2) + '\n';
  if (options.out) {
    fs.mkdirSync(path.dirname(options.out), { recursive: true });
    fs.writeFileSync(options.out, serialized);
  }
  console.log(serialized);
  if (report.status !== 'pass') process.exitCode = 1;
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exitCode = 1;
});
