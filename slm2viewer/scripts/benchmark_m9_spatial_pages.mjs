#!/usr/bin/env node
/*
 * M9 spatial feature-page semantic audit.
 *
 * The audit compares the exact candidate set obtained from the full runtime
 * AABB table with the candidate set obtained by selecting conservative page
 * bounds and then exact-testing the FP32 AABBs contained in those pages. It
 * does not run neural inference and therefore cannot be used as a performance
 * or image-quality claim.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Box3, Frustum, Matrix4, PerspectiveCamera, Vector3 } from 'three';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(SCRIPT_DIR, '..');
const DEFAULTS = {
  modelMeta: path.join(PROJECT_DIR, 'assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_spatial_pages_m9/instance_model_meta.json'),
  runtimeMeta: path.join(PROJECT_DIR, 'assets/scenes/hkust-v3/runtimeVisibilityMeta.json'),
  poseDataset: path.resolve(PROJECT_DIR, '../neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1'),
  samples: 128,
  far: 2000,
};

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function parseArgs(argv) {
  const options = { ...DEFAULTS, out: null, help: false };
  const names = new Set(['model-meta', 'runtime-meta', 'pose-dataset', 'samples', 'far', 'out']);
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') {
      options.help = true;
      continue;
    }
    const equal = token.indexOf('=');
    const rawName = equal >= 0 ? token.slice(0, equal) : token;
    const name = rawName.replace(/^--/, '');
    if (!names.has(name)) throw new Error(`Unknown argument ${token}`);
    const value = equal >= 0 ? token.slice(equal + 1) : argv[++index];
    if (value == null || String(value).startsWith('--')) throw new Error(`Missing value for --${name}`);
    if (name === 'model-meta') options.modelMeta = path.resolve(String(value));
    else if (name === 'runtime-meta') options.runtimeMeta = path.resolve(String(value));
    else if (name === 'pose-dataset') options.poseDataset = path.resolve(String(value));
    else if (name === 'samples') options.samples = Math.max(1, Math.trunc(Number(value)));
    else if (name === 'far') options.far = Math.max(1, Number(value));
    else if (name === 'out') options.out = path.resolve(String(value));
  }
  if (!Number.isFinite(options.samples) || !Number.isFinite(options.far)) throw new Error('samples and far must be finite');
  return options;
}

function sceneMinMax(bounds) {
  if (Array.isArray(bounds?.min) && Array.isArray(bounds?.max)) return { min: bounds.min, max: bounds.max };
  const center = bounds.center.map(Number);
  const size = bounds.size.map(Number);
  return { min: center.map((v, i) => v - size[i] * 0.5), max: center.map((v, i) => v + size[i] * 0.5) };
}

function readPoseRows(datasetDir) {
  const meta = readJson(path.join(datasetDir, 'dataset_meta.json'));
  const stride = Number(meta.poseStrideBytes || 64);
  if (stride !== 64) throw new Error(`Expected 64-byte pose rows, got ${stride}`);
  const buffer = fs.readFileSync(path.join(datasetDir, 'poses.bin'));
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const count = Math.floor(buffer.byteLength / stride);
  const rows = [];
  for (let index = 0; index < count; index += 1) {
    const base = index * stride;
    rows.push({
      position: [view.getFloat32(base + 12, true), view.getFloat32(base + 16, true), view.getFloat32(base + 20, true)],
      forward: [view.getFloat32(base + 24, true), view.getFloat32(base + 28, true), view.getFloat32(base + 32, true)],
      tanX: view.getFloat32(base + 36, true),
      tanY: view.getFloat32(base + 40, true),
    });
  }
  return { meta, rows };
}

function sampleRows(rows, count) {
  if (rows.length <= count) return rows;
  return Array.from({ length: count }, (_, index) => rows[Math.floor(index * rows.length / count)]);
}

function makeCamera(row, far) {
  const tanY = Math.max(1e-4, Number(row.tanY) || Math.tan(66 * Math.PI / 360));
  const tanX = Math.max(1e-4, Number(row.tanX) || tanY * 16 / 9);
  const camera = new PerspectiveCamera(66, tanX / tanY, 0.05, far);
  camera.position.fromArray(row.position);
  camera.lookAt(camera.position.clone().add(new Vector3().fromArray(row.forward).normalize()));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return camera;
}

function frustumFor(camera) {
  return new Frustum().setFromProjectionMatrix(
    new Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
  );
}

function boxFromAabb(row) {
  return new Box3(new Vector3(row[0], row[1], row[2]), new Vector3(row[3], row[4], row[5]));
}

function sortedUnique(values) {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function sameIds(a, b) {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) if (a[index] !== b[index]) return false;
  return true;
}

function parsePage(file, expectedFeatureDim) {
  const buffer = fs.readFileSync(file);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  if (buffer.byteLength < 28 || Buffer.from(buffer.subarray(0, 4)).toString('ascii') !== 'NSPF') throw new Error(`Invalid page magic ${file}`);
  const version = view.getUint32(4, true);
  const count = view.getUint32(8, true);
  const featureDim = view.getUint32(12, true);
  const aabbByteWidth = view.getUint32(16, true);
  if (version !== 2 || featureDim !== expectedFeatureDim || aabbByteWidth !== 4) throw new Error(`Invalid page header ${file}`);
  const idsOffset = 28;
  const aabbOffset = idsOffset + count * 4;
  const featureOffset = aabbOffset + count * 6 * 4;
  if (featureOffset + count * featureDim * 2 !== buffer.byteLength) throw new Error(`Invalid page length ${file}`);
  const ids = new Uint32Array(buffer.buffer, buffer.byteOffset + idsOffset, count);
  const aabbs = new Float32Array(buffer.buffer, buffer.byteOffset + aabbOffset, count * 6);
  return { ids: Array.from(ids), aabbs: Array.from(aabbs) };
}

function percentile(values, probability) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lo = Math.floor(position);
  const hi = Math.ceil(position);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo);
}

function summarize(values) {
  return {
    count: values.length,
    p50: percentile(values, 0.5),
    p95: percentile(values, 0.95),
    min: values.length ? Math.min(...values) : null,
    max: values.length ? Math.max(...values) : null,
    mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
  };
}

function run(options) {
  const model = readJson(options.modelMeta);
  const runtime = readJson(options.runtimeMeta);
  const pagesManifest = readJson(path.resolve(path.dirname(options.modelMeta), model.spatialPageDirectory));
  if (pagesManifest.schema !== 'directional-occlusion-proxy-spatial-pages-v1' || pagesManifest.formatVersion !== 2) {
    throw new Error('The model does not contain the expected spatial-page v2 manifest');
  }
  const componentRecords = runtime.componentRecords || [];
  const aabbs = new Array(Number(model.numInstances));
  for (const record of componentRecords) {
    const id = Number(record.componentGlobalId);
    if (!Number.isInteger(id) || id < 0 || id >= aabbs.length) continue;
    const bounds = record.bounds;
    const min = bounds.min || bounds.center.map((v, i) => v - bounds.size[i] * 0.5);
    const max = bounds.max || bounds.center.map((v, i) => v + bounds.size[i] * 0.5);
    aabbs[id] = [...min.map(Number), ...max.map(Number)];
  }
  if (aabbs.some((value) => !value)) throw new Error('Runtime metadata does not cover all model instances');
  const pageRoot = path.dirname(options.modelMeta);
  const pages = pagesManifest.pages.map((entry) => ({
    ...entry,
    box: boxFromAabb([...entry.bounds.min, ...entry.bounds.max].map(Number)),
    parsed: null,
  }));
  const poseData = readPoseRows(options.poseDataset);
  const rows = sampleRows(poseData.rows, options.samples);
  const candidateCounts = [];
  const pageCandidateCounts = [];
  const pageCounts = [];
  const mismatches = [];
  const loaded = new Set();
  let cumulativePageBytes = 0;
  let initialPageBytes = null;
  for (let index = 0; index < rows.length; index += 1) {
    const frustum = frustumFor(makeCamera(rows[index], options.far));
    const full = [];
    for (let id = 0; id < aabbs.length; id += 1) if (frustum.intersectsBox(boxFromAabb(aabbs[id]))) full.push(id);
    const selected = pages.filter((page) => frustum.intersectsBox(page.box));
    const pageIds = [];
    for (const page of selected) {
      if (!page.parsed) {
        page.parsed = parsePage(path.resolve(pageRoot, page.file), Number(model.runtimeFeatureDim));
        if (!loaded.has(page.pageId)) {
          loaded.add(page.pageId);
          cumulativePageBytes += Number(page.byteSize);
        }
      }
      for (let row = 0; row < page.parsed.ids.length; row += 1) {
        const id = page.parsed.ids[row];
        const offset = row * 6;
        const box = new Box3(
          new Vector3(page.parsed.aabbs[offset], page.parsed.aabbs[offset + 1], page.parsed.aabbs[offset + 2]),
          new Vector3(page.parsed.aabbs[offset + 3], page.parsed.aabbs[offset + 4], page.parsed.aabbs[offset + 5]),
        );
        if (frustum.intersectsBox(box)) pageIds.push(id);
      }
    }
    const fullSorted = sortedUnique(full);
    const pageSorted = sortedUnique(pageIds);
    candidateCounts.push(fullSorted.length);
    pageCandidateCounts.push(pageSorted.length);
    pageCounts.push(selected.length);
    if (!sameIds(fullSorted, pageSorted)) {
      mismatches.push({
        poseIndex: index,
        fullCount: fullSorted.length,
        pageCount: pageSorted.length,
        missing: fullSorted.filter((id) => !pageSorted.includes(id)).slice(0, 20),
        extra: pageSorted.filter((id) => !fullSorted.includes(id)).slice(0, 20),
      });
    }
    if (initialPageBytes == null) initialPageBytes = cumulativePageBytes;
  }
  return {
    schemaVersion: 'm9-spatial-feature-page-audit-v1',
    status: mismatches.length === 0 ? 'pass' : 'fail-candidate-set-mismatch',
    modelMeta: path.relative(PROJECT_DIR, options.modelMeta),
    runtimeMeta: path.relative(PROJECT_DIR, options.runtimeMeta),
    poseDataset: path.relative(PROJECT_DIR, options.poseDataset),
    configuration: {
      modelInputFovYDeg: Number(model.modelInputFovYDeg || 66),
      frontendRenderFovYDeg: Number(model.frontendRenderFovYDeg || 60),
      predictionCameraMode: model.predictionCameraMode || null,
      pvsBackOffsetM: Number(model.pvsBackOffsetM || 0),
      cameraFarM: options.far,
      poseRowsAvailable: poseData.rows.length,
      sampledPoseCount: rows.length,
      numInstances: aabbs.length,
      pageCount: pages.length,
      runtimeFeatureDim: Number(model.runtimeFeatureDim),
    },
    candidateSet: {
      mismatches: mismatches.length,
      firstMismatches: mismatches.slice(0, 5),
      fullScan: summarize(candidateCounts),
      pageQuery: summarize(pageCandidateCounts),
      exactEqual: mismatches.length === 0,
    },
    pages: {
      selectedPageCount: summarize(pageCounts),
      uniquePagesLoaded: loaded.size,
      initialPageBytes,
      cumulativePageBytes,
      directoryBytes: Number(pagesManifest.directoryBytes || fs.statSync(path.resolve(path.dirname(options.modelMeta), model.spatialPageDirectory)).size),
      staticAssetBytes: Number(model.staticAssetBytes || 0),
    },
  };
}

function usage() {
  return 'Usage: node scripts/benchmark_m9_spatial_pages.mjs [--model-meta PATH] [--runtime-meta PATH] [--pose-dataset PATH] [--samples N] [--far M] [--out PATH]';
}

const options = parseArgs(process.argv.slice(2));
if (options.help) {
  console.log(usage());
} else {
  const report = run(options);
  if (options.out) fs.writeFileSync(options.out, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
  if (report.status !== 'pass') process.exitCode = 1;
}
