#!/usr/bin/env node
/* Run the geometry-shell HZB browser MVP with an explicit shell and dataset. */

import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  classifyGpuText,
  headlessWebGpuArgs,
  resolveVulkanEnvironment,
} from './chrome_gpu_flags.mjs';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_ROOT = path.resolve(SCRIPT_DIR, '..');
const REPO_ROOT = path.resolve(VIEWER_ROOT, '..');
const POSE_STRIDE_BYTES = 64;
const CAMERA_VIEW_OFFSET_BYTES = 36;
const CAMERA_VIEW_END_BYTES = CAMERA_VIEW_OFFSET_BYTES + 8;
const SOFTWARE_PATTERN = /swiftshader|llvmpipe|softpipe|swrast|software/i;
const REGION_SAMPLE_COUNTS = new Set([0, 1, 5, 9]);
const REGION_SELECTION_STRATEGY = 'canonical-nearest-then-farthest-point-world-position';
const HZB_TIMING_STAGES = [
  'depthRaster', 'hzbBuild', 'aabbTest', 'compaction', 'total',
];
const HZB_TIMING_DEFINITION = {
  source: 'browser-performance-now-wall-clock',
  stages: {
    depthRaster: 'depth-only render submission through queue completion',
    hzbBuild: 'explicit HZB mip-chain compute submission through queue completion',
    aabbTest: 'candidate AABB/HZB classification dispatch through queue completion; shader append is included',
    compaction: 'result copy, readback, ID sorting and GLB aggregation after the classification dispatch',
    total: 'depthRaster + hzbBuild + aabbTest + compaction',
  },
};

function parseArgs(argv) {
  const options = {
    shellDir: '',
    datasetDir: '',
    regionDatasetDir: '',
    output: '',
    mode: 'Point60',
    split: 'test',
    limit: 0,
    warmupCount: 0,
    timingRounds: 5,
    poseIndexPlan: '',
    width: 512,
    height: 288,
    aspect: 16 / 9,
    aspectProvided: false,
    fovYDeg: 60,
    regionFovYDeg: 66,
    near: 0.05,
    far: 20000,
    depthBiasM: 0.001,
    chromeExe: process.env.CHROME_PATH || process.env.CHROMIUM_PATH || '',
    port: 0,
    timeoutMs: 30 * 60 * 1000,
    requireHardwareGpu: true,
    regionSampleCount: 0,
  };
  const valueOptions = new Set([
    'shell-dir', 'dataset-dir', 'region-dataset-dir', 'output', 'mode', 'split', 'limit',
    'warmup-count', 'timing-rounds', 'pose-index-plan', 'width', 'height', 'aspect', 'fov-y-deg',
    'region-fov-y-deg', 'near', 'far',
    'depth-bias-m', 'chrome-exe', 'port', 'timeout-ms', 'region-sample-count',
  ]);
  for (let index = 2; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--allow-software-gpu') {
      options.requireHardwareGpu = false;
      continue;
    }
    if (token === '--require-hardware-gpu') {
      options.requireHardwareGpu = true;
      continue;
    }
    const equal = token.indexOf('=');
    const key = token.slice(0, equal >= 0 ? equal : undefined).replace(/^--/, '');
    if (!valueOptions.has(key)) throw new Error(`Unknown argument: ${token}`);
    const value = equal >= 0 ? token.slice(equal + 1) : argv[++index];
    if (value == null) throw new Error(`Missing value for --${key}`);
    if (key === 'shell-dir') options.shellDir = path.resolve(value);
    else if (key === 'dataset-dir') options.datasetDir = path.resolve(value);
    else if (key === 'region-dataset-dir') options.regionDatasetDir = path.resolve(value);
    else if (key === 'output') options.output = path.resolve(value);
    else if (key === 'mode') options.mode = String(value);
    else if (key === 'split') options.split = String(value);
    else if (key === 'limit') options.limit = Number(value);
    else if (key === 'warmup-count') options.warmupCount = Number(value);
    else if (key === 'timing-rounds') options.timingRounds = Number(value);
    else if (key === 'pose-index-plan') options.poseIndexPlan = path.resolve(value);
    else if (key === 'width') options.width = Number(value);
    else if (key === 'height') options.height = Number(value);
    else if (key === 'aspect') {
      options.aspect = Number(value);
      options.aspectProvided = true;
    }
    else if (key === 'fov-y-deg') options.fovYDeg = Number(value);
    else if (key === 'region-fov-y-deg') options.regionFovYDeg = Number(value);
    else if (key === 'near') options.near = Number(value);
    else if (key === 'far') options.far = Number(value);
    else if (key === 'depth-bias-m') options.depthBiasM = Number(value);
    else if (key === 'chrome-exe') options.chromeExe = path.resolve(value);
    else if (key === 'port') options.port = Number(value);
    else if (key === 'timeout-ms') options.timeoutMs = Number(value);
    else if (key === 'region-sample-count') options.regionSampleCount = Number(value);
  }
  if (!options.shellDir || !options.datasetDir || !options.output) {
    throw new Error('--shell-dir, --dataset-dir and --output are required.');
  }
  if (!['Point60', 'Region66'].includes(options.mode)) throw new Error('--mode must be Point60 or Region66.');
  if (options.mode === 'Region66' && !options.regionDatasetDir) {
    throw new Error('Region66 requires --region-dataset-dir with subpose camera arrays.');
  }
  for (const [name, value] of [
    ['limit', options.limit], ['warmup-count', options.warmupCount], ['timing-rounds', options.timingRounds],
    ['width', options.width], ['height', options.height],
  ]) {
    if (!Number.isInteger(value) || value < 0 || (name === 'timing-rounds' && value < 1)
        || (name === 'width' && value < 8) || (name === 'height' && value < 8)) {
      throw new Error(`--${name} is invalid.`);
    }
  }
  if (!(options.aspect > 0) || !(options.near >= 0) || !(options.far > options.near)
      || !(options.depthBiasM >= 0) || !(options.timeoutMs > 0)) throw new Error('camera/timing options are invalid.');
  if (!Number.isInteger(options.port) || options.port < 0 || options.port > 65535) throw new Error('--port is invalid.');
  if (!Number.isInteger(options.regionSampleCount) || !REGION_SAMPLE_COUNTS.has(options.regionSampleCount)) {
    throw new Error('--region-sample-count must be one of 0, 1, 5 or 9.');
  }
  return options;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function poseIndexFromPlanEntry(entry, source) {
  const value = entry && typeof entry === 'object'
    ? (entry.poseIndex ?? entry.pose_index ?? entry.poseId ?? entry.pose_id ?? entry.index)
    : entry;
  if (value == null || value === '' || typeof value === 'boolean'
      || !Number.isInteger(Number(value)) || Number(value) < 0) {
    throw new Error(`${source} contains a non-negative integer pose index requirement.`);
  }
  return Number(value);
}

function readPoseIndexPlan(filePath) {
  const source = path.resolve(filePath);
  const text = fs.readFileSync(source, 'utf8');
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    const entries = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`failed to parse pose index plan ${source}:${index + 1}: ${error.message}`);
      }
    });
    payload = entries;
  }
  const values = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.poseIndices)
      ? payload.poseIndices
      : Array.isArray(payload?.pose_indices)
        ? payload.pose_indices
      : Array.isArray(payload?.poseIndex)
        ? payload.poseIndex
        : Array.isArray(payload?.pose_index)
          ? payload.pose_index
          : Array.isArray(payload?.poseIds)
            ? payload.poseIds
            : Array.isArray(payload?.indices)
              ? payload.indices
              : Array.isArray(payload?.poses)
                ? payload.poses
                : null;
  if (!values || values.length === 0) {
    throw new Error(`pose index plan ${source} must contain a non-empty poseIndices array.`);
  }
  const poseIndices = values.map((entry, index) => poseIndexFromPlanEntry(entry, `${source}[${index}]`));
  const seen = new Set();
  for (const poseIndex of poseIndices) {
    if (seen.has(poseIndex)) throw new Error(`pose index plan ${source} contains duplicate pose index ${poseIndex}.`);
    seen.add(poseIndex);
  }
  return {
    schema: String(payload?.schema || 'geometry-shell-hzb-pose-index-plan-external-v1'),
    split: payload && !Array.isArray(payload) && payload.split != null ? String(payload.split) : null,
    poseIndices,
    source,
    metadata: payload && !Array.isArray(payload) ? {
      selection: payload.selection || payload.selectionStrategy || null,
      seed: payload.seed ?? null,
      targetPoseCount: payload.targetPoseCount ?? payload.poseCount ?? null,
      strata: Array.isArray(payload.strata) ? payload.strata : null,
    } : null,
  };
}

function inspectPoseAspectData(meta, poseBuffer, poseCount, selectedPoseIndices = null) {
  const declared = meta?.hasCameraView
    ?? meta?.cameraViewPresent
    ?? meta?.poseFormat?.hasCameraView
    ?? null;
  const stride = Number(meta?.poseStrideBytes ?? POSE_STRIDE_BYTES);
  const hasStorage = stride >= CAMERA_VIEW_END_BYTES
    && poseBuffer.byteLength >= Math.max(0, poseCount - 1) * POSE_STRIDE_BYTES + CAMERA_VIEW_END_BYTES;
  if (declared === false || !hasStorage) {
    return { hasPerPoseAspect: false, source: 'fixed-aspect-fallback', cameraView: null };
  }
  let validCount = 0;
  let missingCount = 0;
  const view = new DataView(poseBuffer.buffer, poseBuffer.byteOffset, poseBuffer.byteLength);
  const poseIndices = selectedPoseIndices || Array.from({ length: poseCount }, (_, index) => index);
  for (const poseIndex of poseIndices) {
    if (!Number.isInteger(poseIndex) || poseIndex < 0 || poseIndex >= poseCount) {
      throw new Error(`pose index ${poseIndex} is outside dataset poseCount.`);
    }
    const offset = poseIndex * POSE_STRIDE_BYTES + CAMERA_VIEW_OFFSET_BYTES;
    const tanX = view.getFloat32(offset, true);
    const tanY = view.getFloat32(offset + 4, true);
    if (Number.isFinite(tanX) && Number.isFinite(tanY) && tanX > 0 && tanY > 0
        && Number.isFinite(tanX / tanY) && tanX / tanY > 0) {
      validCount += 1;
    } else if (tanX === 0 && tanY === 0) {
      missingCount += 1;
    }
  }
  if (validCount === poseIndices.length) {
    return { hasPerPoseAspect: true, source: 'dataset.camera_view', cameraView: 'poses.bin:+36,+40' };
  }
  if (declared !== true && missingCount === poseIndices.length) {
    return { hasPerPoseAspect: false, source: 'fixed-aspect-fallback', cameraView: null };
  }
  throw new Error('poses.bin camera_view must contain finite positive tan_x/tan_y for every pose.');
}

function isIfcBenchDataset(options, meta, shellMeta) {
  return [
    options.datasetDir,
    meta?.scene,
    meta?.experiment,
    shellMeta?.sceneName,
  ].filter(Boolean).join(' ').toLowerCase().includes('ifcbench')
    || String(shellMeta?.sceneName || '').toLowerCase().includes('metropolis');
}

function poseAspect(pose, aspectInfo, fallbackAspect) {
  if (!aspectInfo.hasPerPoseAspect) return { aspect: fallbackAspect, cameraView: null };
  const [tanX, tanY] = pose.cameraView;
  const aspect = tanX / tanY;
  if (!Number.isFinite(aspect) || aspect <= 0) throw new Error('pose camera_view has an invalid aspect ratio.');
  return { aspect, cameraView: [tanX, tanY] };
}

function readPoseRecord(buffer, index) {
  const offset = index * POSE_STRIDE_BYTES;
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  return {
    position: [view.getFloat32(offset + 12, true), view.getFloat32(offset + 16, true), view.getFloat32(offset + 20, true)],
    forward: [view.getFloat32(offset + 24, true), view.getFloat32(offset + 28, true), view.getFloat32(offset + 32, true)],
    cameraView: [view.getFloat32(offset + CAMERA_VIEW_OFFSET_BYTES, true), view.getFloat32(offset + CAMERA_VIEW_OFFSET_BYTES + 4, true)],
    split: view.getUint8(offset + 44),
    category: view.getUint8(offset + 45),
  };
}

function readVec3File(filePath, count) {
  const buffer = fs.readFileSync(filePath);
  if (buffer.byteLength !== count * 3 * 4) throw new Error(`${filePath} has an invalid vec3 length.`);
  const values = new Float32Array(buffer.buffer, buffer.byteOffset, count * 3);
  return values;
}

function readU64File(filePath, count) {
  const buffer = fs.readFileSync(filePath);
  if (buffer.byteLength !== count * 8) throw new Error(`${filePath} has an invalid uint64 length.`);
  return new BigUint64Array(buffer.buffer, buffer.byteOffset, count);
}

function readU32File(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (buffer.byteLength % 4 !== 0) throw new Error(`${filePath} has an invalid uint32 length.`);
  return new Uint32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
}

function requireFile(directory, name) {
  const file = path.join(directory, name);
  if (!fs.existsSync(file)) throw new Error(`missing dataset file: ${file}`);
  return file;
}

function squaredDistance(left, right) {
  let value = 0;
  for (let index = 0; index < 3; index += 1) {
    const delta = Number(left[index]) - Number(right[index]);
    value += delta * delta;
  }
  return value;
}

/**
 * Select real source subposes without consulting any visibility labels.
 * The ordinal is the source order inside this view-cell and is the sole tie
 * breaker, so equal positions and equal distances remain reproducible.
 */
function selectRegionSubposes(subposes, canonicalCenter, requestedCount) {
  if (!Number.isInteger(requestedCount) || !REGION_SAMPLE_COUNTS.has(requestedCount)) {
    throw new Error('region sample count must be one of 0, 1, 5 or 9.');
  }
  if (!Array.isArray(canonicalCenter) || canonicalCenter.length !== 3
      || canonicalCenter.some((value) => !Number.isFinite(Number(value)))) {
    throw new Error('canonical query center must be a finite vec3.');
  }
  const ordered = subposes.map((subpose, index) => {
    const ordinal = Number(subpose.ordinal ?? index);
    const position = Array.from(subpose.position || [], Number);
    if (!Number.isInteger(ordinal) || ordinal < 0 || position.length !== 3
        || position.some((value) => !Number.isFinite(value))) {
      throw new Error('region subposes must have a finite position and non-negative integer ordinal.');
    }
    return { ...subpose, ordinal, position };
  }).sort((left, right) => left.ordinal - right.ordinal);
  for (let index = 1; index < ordered.length; index += 1) {
    if (ordered[index - 1].ordinal === ordered[index].ordinal) {
      throw new Error('region subpose ordinals must be unique within a view-cell.');
    }
  }
  if (requestedCount === 0) return ordered;
  const targetCount = Math.min(requestedCount, ordered.length);
  if (targetCount === 0) return [];

  const remaining = ordered.slice();
  const selected = [];
  let nearestIndex = 0;
  let nearestDistance = squaredDistance(remaining[0].position, canonicalCenter);
  for (let index = 1; index < remaining.length; index += 1) {
    const distance = squaredDistance(remaining[index].position, canonicalCenter);
    if (distance < nearestDistance
        || (distance === nearestDistance && remaining[index].ordinal < remaining[nearestIndex].ordinal)) {
      nearestIndex = index;
      nearestDistance = distance;
    }
  }
  selected.push(remaining.splice(nearestIndex, 1)[0]);

  while (selected.length < targetCount) {
    let bestIndex = 0;
    let bestDistance = -Infinity;
    for (let index = 0; index < remaining.length; index += 1) {
      const distance = Math.min(
        ...selected.map((subpose) => squaredDistance(remaining[index].position, subpose.position)),
      );
      if (distance > bestDistance
          || (distance === bestDistance && remaining[index].ordinal < remaining[bestIndex].ordinal)) {
        bestIndex = index;
        bestDistance = distance;
      }
    }
    selected.push(remaining.splice(bestIndex, 1)[0]);
  }
  return selected;
}

function regionSelectionMode(requestedCount) {
  return requestedCount === 0 ? 'all' : 'deterministic-fps-subset';
}

function makeRegionSamplingRecord(requestedCount, availableSubposeCount, selectedSubposes) {
  return {
    requestedCount,
    availableSubposeCount,
    selectedSubposeCount: selectedSubposes.length,
    selectionMode: regionSelectionMode(requestedCount),
    strategy: requestedCount === 0 ? 'all-subposes-in-source-order' : REGION_SELECTION_STRATEGY,
    labelSource: 'none',
    selectedSubposeOrdinals: selectedSubposes.map((subpose) => subpose.ordinal),
    selectedSourcePoseIndices: selectedSubposes
      .map((subpose) => subpose.sourcePoseIndex)
      .filter((value) => value != null),
  };
}

function summarizeRegionSampling(rows, requestedCount) {
  const records = rows.map((row) => row.regionSampling).filter(Boolean);
  const available = records.map((record) => Number(record.availableSubposeCount));
  const selected = records.map((record) => Number(record.selectedSubposeCount));
  return {
    schema: 'geometry-shell-hzb-region-sampling-v1',
    requestedCount,
    selectionMode: regionSelectionMode(requestedCount),
    strategy: requestedCount === 0 ? 'all-subposes-in-source-order' : REGION_SELECTION_STRATEGY,
    labelSource: 'none',
    viewCellCount: records.length,
    availableSubposeCount: available.reduce((sum, value) => sum + value, 0),
    selectedSubposeCount: selected.reduce((sum, value) => sum + value, 0),
    minAvailableSubposeCount: Math.min(...available),
    maxAvailableSubposeCount: Math.max(...available),
    meanAvailableSubposeCount: available.reduce((sum, value) => sum + value, 0) / Math.max(1, available.length),
    minSelectedSubposeCount: Math.min(...selected),
    maxSelectedSubposeCount: Math.max(...selected),
    meanSelectedSubposeCount: selected.reduce((sum, value) => sum + value, 0) / Math.max(1, selected.length),
    perViewCell: 'workload.poses[].regionSampling',
  };
}

function quantile(values, probability) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (sorted.length === 0) return null;
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  return lower === upper
    ? sorted[lower]
    : sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function timingValue(timings, stage) {
  if (stage === 'depthRaster') {
    return Number(timings?.depthRasterMs ?? timings?.depthRasterAndMipMs);
  }
  if (stage === 'hzbBuild') return Number(timings?.hzbBuildMs);
  if (stage === 'aabbTest') return Number(timings?.aabbTestMs ?? timings?.queryMs);
  if (stage === 'compaction') return Number(timings?.compactionMs ?? timings?.readbackMs);
  return Number(timings?.totalMs);
}

function summarizeTimingSamples(samples) {
  const rows = Array.isArray(samples) ? samples : [];
  const stages = {};
  for (const stage of HZB_TIMING_STAGES) {
    const values = rows.map((sample) => timingValue(sample?.timings, stage));
    const p50Ms = quantile(values, 0.5);
    const p95Ms = quantile(values, 0.95);
    stages[stage] = {
      sampleCount: values.filter(Number.isFinite).length,
      p50Ms,
      p95Ms,
      p50: p50Ms,
      p95: p95Ms,
    };
  }
  const summary = {
    schema: 'geometry-shell-hzb-timing-summary-v1',
    sampleCount: rows.length,
    stages,
  };
  for (const stage of HZB_TIMING_STAGES) {
    const label = `${stage[0].toUpperCase()}${stage.slice(1)}`;
    summary[`${stage}P50Ms`] = stages[stage].p50Ms;
    summary[`${stage}P95Ms`] = stages[stage].p95Ms;
    summary[`${label}P50Ms`] = stages[stage].p50Ms;
    summary[`${label}P95Ms`] = stages[stage].p95Ms;
  }
  return summary;
}

function timingSampleRecord(sample, round) {
  return {
    round,
    poseId: Number(sample.poseId),
    candidateCount: Number(sample.candidateCount),
    timings: sample.timings,
  };
}

function summarizeTimingRounds(roundResults) {
  if (!Array.isArray(roundResults) || roundResults.length === 0) {
    throw new Error('at least one timing round is required.');
  }
  const rounds = roundResults.map((roundResult, index) => {
    const samples = Array.isArray(roundResult?.samples) ? roundResult.samples : [];
    return {
      round: index + 1,
      sampleCount: samples.length,
      summary: summarizeTimingSamples(samples),
      samples: samples.map((sample) => timingSampleRecord(sample, index + 1)),
    };
  });
  const allSamples = roundResults.flatMap((roundResult) => roundResult?.samples || []);
  const aggregate = summarizeTimingSamples(allSamples);
  return {
    schema: 'geometry-shell-hzb-five-round-timing-v1',
    roundCount: rounds.length,
    samplesPerRound: rounds.map((round) => round.sampleCount),
    sampleCount: aggregate.sampleCount,
    stages: aggregate.stages,
    rounds,
    ...Object.fromEntries(HZB_TIMING_STAGES.flatMap((stage) => [
      [`${stage}P50Ms`, aggregate.stages[stage].p50Ms],
      [`${stage}P95Ms`, aggregate.stages[stage].p95Ms],
    ])),
  };
}

function attachWorkloadProvenance(browserResult, workload) {
  const fields = [
    'scene', 'split', 'mode', 'poseCount', 'candidateCount', 'width', 'height', 'resolution', 'aspect',
    'fallbackAspect', 'aspectSource', 'fovYDeg', 'near', 'far', 'depthBiasM', 'timingRounds',
    'warmupCount', 'timingDefinition', 'poseSelection', 'poseIndexPlan', 'provenance',
  ];
  const workloadFields = Object.fromEntries(fields
    .filter((key) => workload[key] !== undefined)
    .map((key) => [key, workload[key]]));
  return {
    ...browserResult,
    workload: { ...browserResult.workload, ...workloadFields },
    provenance: workload.provenance || null,
  };
}

function combineTimingResults(roundResults, workload) {
  if (!Array.isArray(roundResults) || roundResults.length === 0) {
    throw new Error('at least one browser timing result is required.');
  }
  const first = roundResults[0];
  const expected = (first.samples || []).map((sample) => [
    Number(sample.poseId),
    Number(sample.candidateCount),
  ]);
  for (let roundIndex = 1; roundIndex < roundResults.length; roundIndex += 1) {
    const actual = (roundResults[roundIndex].samples || []).map((sample) => [
      Number(sample.poseId),
      Number(sample.candidateCount),
    ]);
    if (actual.length !== expected.length || actual.some((value, index) => (
      value[0] !== expected[index][0] || value[1] !== expected[index][1]
    ))) {
      throw new Error(`timing round ${roundIndex + 1} did not execute the same pose-index plan.`);
    }
  }
  const timingSummary = summarizeTimingRounds(roundResults);
  const gpuValidationErrors = [...new Set(roundResults.flatMap((roundResult) => (
    roundResult.gpuValidationErrors || []
  )))];
  const timingFields = Object.fromEntries(HZB_TIMING_STAGES.flatMap((stage) => [
    [`${stage}P50Ms`, timingSummary.stages[stage].p50Ms],
    [`${stage}P95Ms`, timingSummary.stages[stage].p95Ms],
  ]));
  const summary = {
    ...(first.summary || {}),
    roundCount: timingSummary.roundCount,
    timingSampleCount: timingSummary.sampleCount,
    stages: timingSummary.stages,
    ...timingFields,
    totalP50Ms: timingSummary.stages.total.p50Ms,
    totalP95Ms: timingSummary.stages.total.p95Ms,
  };
  const provenance = workload.provenance || first.provenance || null;
  return {
    ...first,
    gpuValidationErrors,
    summary,
    timingSummary,
    timingRounds: timingSummary.rounds,
    provenance: provenance
      ? {
        ...provenance,
        timing: {
          schema: 'geometry-shell-hzb-five-round-timing-v1',
          roundCount: timingSummary.roundCount,
          sampleCount: timingSummary.sampleCount,
          samplesPerRound: timingSummary.samplesPerRound,
          stages: timingSummary.stages,
        },
      }
      : null,
    workload: {
      ...first.workload,
      provenance: provenance
        ? {
          ...provenance,
          timing: {
            schema: 'geometry-shell-hzb-five-round-timing-v1',
            roundCount: timingSummary.roundCount,
            sampleCount: timingSummary.sampleCount,
            samplesPerRound: timingSummary.samplesPerRound,
            stages: timingSummary.stages,
          },
        }
        : first.workload?.provenance || null,
    },
  };
}

function attachRegionSamplingToResult(browserResult, workload) {
  const regionSampling = workload.regionSampling || null;
  const rowsByPoseId = new Map(
    workload.poses.map((row) => [Number(row.poseId), row]),
  );
  return {
    ...browserResult,
    regionSampling,
    workload: { ...browserResult.workload, regionSampling },
    samples: (browserResult.samples || []).map((sample) => ({
      ...sample,
      regionSampling: rowsByPoseId.get(Number(sample.poseId))?.regionSampling || null,
    })),
  };
}

function buildPointRows(
  options,
  meta,
  poseBuffer,
  candidateOffsets,
  candidateIds,
  queryCenters,
  aspectInfo = null,
  poseIndexPlan = null,
) {
  const splitId = Number(meta.splitIds?.[options.split]);
  if (!Number.isInteger(splitId)) throw new Error(`dataset_meta.json has no split ${options.split}.`);
  const resolvedAspectInfo = aspectInfo || inspectPoseAspectData(meta, poseBuffer, Number(meta.poseCount));
  const poseIds = poseIndexPlan
    ? poseIndexPlan.poseIndices
    : Array.from({ length: Number(meta.poseCount) }, (_, poseId) => poseId);
  const rows = [];
  for (const poseId of poseIds) {
    if (!Number.isInteger(poseId) || poseId < 0 || poseId >= Number(meta.poseCount)) {
      throw new Error(`pose index ${poseId} is outside dataset poseCount.`);
    }
    const pose = readPoseRecord(poseBuffer, poseId);
    if (poseIndexPlan && pose.split !== splitId) {
      throw new Error(`pose index plan selected pose ${poseId}, which is not in split ${options.split}.`);
    }
    if (!poseIndexPlan && pose.split !== splitId) continue;
    const sourceStart = Number(candidateOffsets[poseId]);
    const sourceEnd = Number(candidateOffsets[poseId + 1]);
    const source = candidateIds.subarray(sourceStart, sourceEnd);
    const position = queryCenters
      ? Array.from(queryCenters.subarray(poseId * 3, poseId * 3 + 3))
      : pose.position;
    const resolvedAspect = poseAspect(pose, resolvedAspectInfo, options.aspect);
    rows.push({
      poseId,
      position,
      forward: pose.forward,
      fovYDeg: options.fovYDeg,
      aspect: resolvedAspect.aspect,
      cameraView: resolvedAspect.cameraView,
      near: options.near,
      far: options.far,
      sourceCandidateIds: source,
      category: pose.category,
    });
    if (options.limit > 0 && rows.length >= options.limit) break;
  }
  return rows;
}

function buildRegionRows(
  options,
  meta,
  poseBuffer,
  candidateOffsets,
  candidateIds,
  queryCenters,
  regionDir,
  aspectInfo = null,
  poseIndexPlan = null,
) {
  const regionSampleCount = options.regionSampleCount ?? 0;
  const pointRows = buildPointRows(
    { ...options, limit: 0 },
    meta,
    poseBuffer,
    candidateOffsets,
    candidateIds,
    queryCenters,
    aspectInfo,
    poseIndexPlan,
  );
  const regionMeta = readJson(requireFile(regionDir, 'dataset_meta.json'));
  const regionPoseCount = Number(regionMeta.viewcellCount ?? regionMeta.poseCount);
  const offsetsPath = path.join(regionDir, 'subpose_offsets.bin');
  const positionsPath = path.join(regionDir, 'subpose_camera_pos.bin');
  const forwardsPath = path.join(regionDir, 'subpose_camera_forward.bin');
  const centersPath = path.join(regionDir, 'viewcell_centers.bin');
  const sourcePoseIndicesPath = path.join(regionDir, 'subpose_pose_indices.bin');
  if (!fs.existsSync(offsetsPath) || !fs.existsSync(positionsPath) || !fs.existsSync(forwardsPath)
      || !fs.existsSync(centersPath) || !fs.existsSync(sourcePoseIndicesPath)) {
    throw new Error('Region66 requires subpose offsets/positions/forwards, viewcell centers and source pose indices.');
  }
  const offsets = readU64File(offsetsPath, regionPoseCount + 1);
  if (offsets[0] !== 0n || offsets.some((value, index) => index > 0 && value < offsets[index - 1])) {
    throw new Error('Region66 subpose offsets must be monotonic and start at zero.');
  }
  const subposeCount = Number(offsets[regionPoseCount]);
  const positions = readVec3File(positionsPath, subposeCount);
  const forwards = readVec3File(forwardsPath, subposeCount);
  const centers = readVec3File(centersPath, regionPoseCount);
  const sourcePoseIndices = readU32File(sourcePoseIndicesPath);
  if (sourcePoseIndices.length !== subposeCount) {
    throw new Error('Region66 subpose_pose_indices.bin length does not match subpose_offsets.bin.');
  }
  const rows = [];
  for (const row of pointRows) {
    if (row.poseId >= regionPoseCount) throw new Error(`Region66 pose ${row.poseId} is outside region dataset.`);
    const start = Number(offsets[row.poseId]);
    const end = Number(offsets[row.poseId + 1]);
    const availableSubposes = [];
    for (let index = start; index < end; index += 1) {
      availableSubposes.push({
        ordinal: index - start,
        sourcePoseIndex: Number(sourcePoseIndices[index]),
        position: Array.from(positions.subarray(index * 3, index * 3 + 3)),
        forward: Array.from(forwards.subarray(index * 3, index * 3 + 3)),
        fovYDeg: options.regionFovYDeg,
        aspect: row.aspect,
        cameraView: row.cameraView,
        near: options.near,
        far: options.far,
      });
    }
    if (availableSubposes.length === 0) {
      throw new Error(`Region66 view-cell ${row.poseId} has no successful subposes.`);
    }
    const canonicalCenter = Array.from(centers.subarray(row.poseId * 3, row.poseId * 3 + 3));
    const subposes = selectRegionSubposes(availableSubposes, canonicalCenter, regionSampleCount);
    rows.push({
      ...row,
      subposes,
      fovYDeg: options.regionFovYDeg,
      regionSampling: makeRegionSamplingRecord(regionSampleCount, availableSubposes.length, subposes),
    });
    if (options.limit > 0 && rows.length >= options.limit) break;
  }
  return rows;
}

function buildWorkload(options) {
  const regionSampleCount = options.regionSampleCount ?? 0;
  const datasetMeta = readJson(requireFile(options.datasetDir, 'dataset_meta.json'));
  const shellMeta = readJson(requireFile(options.shellDir, 'shell_meta.json'));
  if (shellMeta.schema !== 'geometry-shell-hzb-v2') throw new Error('shell directory has an unsupported schema.');
  if (Number(datasetMeta.numInstances) !== Number(shellMeta.instanceCount)) {
    throw new Error('dataset and geometry shell instance counts disagree.');
  }
  const poseCount = Number(datasetMeta.poseCount);
  const poseBytes = fs.readFileSync(requireFile(options.datasetDir, 'poses.bin'));
  if (poseBytes.byteLength !== poseCount * POSE_STRIDE_BYTES) throw new Error('poses.bin length does not match dataset_meta.json.');
  const poseBuffer = new Uint8Array(poseBytes.buffer, poseBytes.byteOffset, poseBytes.byteLength);
  const poseIndexPlan = options.poseIndexPlan ? readPoseIndexPlan(options.poseIndexPlan) : null;
  if (poseIndexPlan?.split && poseIndexPlan.split !== String(options.split)) {
    throw new Error(`pose index plan split ${poseIndexPlan.split} does not match --split ${options.split}.`);
  }
  const splitId = Number(datasetMeta.splitIds?.[options.split]);
  if (!Number.isInteger(splitId)) throw new Error(`dataset_meta.json has no split ${options.split}.`);
  const poseView = new DataView(poseBuffer.buffer, poseBuffer.byteOffset, poseBuffer.byteLength);
  const aspectPoseIndices = poseIndexPlan?.poseIndices || Array.from(
    { length: poseCount },
    (_, poseIndex) => poseIndex,
  ).filter((poseIndex) => poseView.getUint8(poseIndex * POSE_STRIDE_BYTES + 44) === splitId);
  const aspectInfo = inspectPoseAspectData(datasetMeta, poseBuffer, poseCount, aspectPoseIndices);
  if (!aspectInfo.hasPerPoseAspect && isIfcBenchDataset(options, datasetMeta, shellMeta)) {
    throw new Error('IFCBench formal workloads require per-pose dataset camera_view aspect; fixed --aspect is not allowed.');
  }
  const candidateOffsets = readU64File(requireFile(options.datasetDir, 'candidate_offsets.bin'), poseCount + 1);
  const candidateIds = readU32File(requireFile(options.datasetDir, 'candidate_ids.bin'));
  if (Number(candidateOffsets[0]) !== 0 || Number(candidateOffsets.at(-1)) !== candidateIds.length) {
    throw new Error('candidate CSR offsets are invalid.');
  }
  const queryCentersPath = path.join(options.datasetDir, 'query_center_world.bin');
  const queryCenters = fs.existsSync(queryCentersPath) ? readVec3File(queryCentersPath, poseCount) : null;
  const rows = options.mode === 'Region66'
    ? buildRegionRows(
      options,
      datasetMeta,
      poseBuffer,
      candidateOffsets,
      candidateIds,
      queryCenters,
      options.regionDatasetDir,
      aspectInfo,
      poseIndexPlan,
    )
    : buildPointRows(
      options,
      datasetMeta,
      poseBuffer,
      candidateOffsets,
      candidateIds,
      queryCenters,
      aspectInfo,
      poseIndexPlan,
    );
  if (rows.length === 0) throw new Error('selected split has no poses.');
  const outputDir = path.dirname(options.output);
  fs.mkdirSync(outputDir, { recursive: true });
  const poseSelection = {
    schema: 'geometry-shell-hzb-pose-selection-v1',
    source: poseIndexPlan ? 'explicit-pose-index-plan' : 'dataset-split-order',
    planFile: poseIndexPlan?.source || null,
    planSchema: poseIndexPlan?.schema || null,
    planMetadata: poseIndexPlan?.metadata || null,
    split: options.split,
    requestedPoseCount: poseIndexPlan ? poseIndexPlan.poseIndices.length : null,
    selectedPoseCount: rows.length,
    selectedPoseIndices: rows.map((row) => row.poseId),
    limit: options.limit || null,
    representative: options.limit === 0
      || Boolean(poseIndexPlan && rows.length === poseIndexPlan.poseIndices.length),
  };
  const workloadCandidates = [];
  let cursor = 0;
  const workloadRows = rows.map((row, ordinal) => {
    const ids = row.sourceCandidateIds;
    workloadCandidates.push(ids);
    const output = {
      ordinal,
      poseId: row.poseId,
      position: row.position,
      forward: row.forward,
      fovYDeg: row.fovYDeg,
      aspect: row.aspect,
      cameraView: row.cameraView,
      near: row.near,
      far: row.far,
      candidateOffset: cursor,
      candidateCount: ids.length,
      category: row.category,
    };
    if (row.subposes) output.subposes = row.subposes;
    if (row.regionSampling) output.regionSampling = row.regionSampling;
    cursor += ids.length;
    return output;
  });
  const flat = new Uint32Array(cursor);
  let flatOffset = 0;
  for (const ids of workloadCandidates) {
    flat.set(ids, flatOffset);
    flatOffset += ids.length;
  }
  const candidateFile = 'geometry_shell_hzb_candidates_uint32.bin';
  fs.writeFileSync(path.join(outputDir, candidateFile), Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength));
  const workload = {
    schema: 'geometry-shell-hzb-browser-workload-v1',
    scene: shellMeta.sceneName,
    split: options.split,
    mode: options.mode,
    poseCount: workloadRows.length,
    candidateCount: flat.length,
    candidateFile,
    candidateDtype: 'uint32-little-endian',
    width: options.width,
    height: options.height,
    resolution: [options.width, options.height],
    aspect: aspectInfo.hasPerPoseAspect ? null : options.aspect,
    fallbackAspect: options.aspect,
    aspectSource: aspectInfo.source,
    fovYDeg: options.mode === 'Region66' ? options.regionFovYDeg : options.fovYDeg,
    near: options.near,
    far: options.far,
    depthBiasM: options.depthBiasM,
    timingRounds: options.timingRounds,
    warmupCount: options.warmupCount,
    timingDefinition: HZB_TIMING_DEFINITION,
    poseSelection,
    poseIndexPlan: poseIndexPlan
      ? {
        schema: poseIndexPlan.schema,
        source: poseIndexPlan.source,
        split: options.split,
        poseIndices: poseIndexPlan.poseIndices,
        metadata: poseIndexPlan.metadata,
      }
      : null,
    provenance: {
      schema: 'geometry-shell-hzb-provenance-v1',
      configuration: {
        mode: options.mode,
        resolution: [options.width, options.height],
        width: options.width,
        height: options.height,
        fovYDeg: options.fovYDeg,
        regionFovYDeg: options.regionFovYDeg,
        near: options.near,
        far: options.far,
        depthBiasM: options.depthBiasM,
        timingRounds: options.timingRounds,
        warmupCount: options.warmupCount,
      },
      timingDefinition: HZB_TIMING_DEFINITION,
      aspect: {
        source: aspectInfo.source,
        fixedAspect: options.aspect,
        fixedAspectProvided: options.aspectProvided,
        fixedAspectUsed: !aspectInfo.hasPerPoseAspect,
        cameraViewEncoding: aspectInfo.cameraView,
      },
      poseSelection,
      poseAspects: workloadRows.map((row) => ({
        ordinal: row.ordinal,
        poseId: row.poseId,
        aspect: row.aspect,
        cameraView: row.cameraView,
        candidateCount: row.candidateCount,
      })),
    },
    regionSampling: options.mode === 'Region66'
      ? summarizeRegionSampling(workloadRows, regionSampleCount)
      : null,
    source: {
      datasetDirName: path.basename(options.datasetDir),
      regionDatasetDirName: options.regionDatasetDir ? path.basename(options.regionDatasetDir) : null,
      candidateSemantics: datasetMeta.candidateSemantics || null,
    },
    poses: workloadRows,
  };
  const workloadFile = path.join(outputDir, 'geometry_shell_hzb_workload.json');
  fs.writeFileSync(workloadFile, `${JSON.stringify(workload, null, 2)}\n`, 'utf8');
  return { workload, workloadFile, outputDir, shellMeta };
}

function findChrome(explicit) {
  const candidates = [
    explicit,
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const candidate of candidates) if (fs.existsSync(candidate)) return candidate;
  throw new Error('Chrome/Chromium executable not found.');
}

async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch {
    return import(`file://${path.join(VIEWER_ROOT, 'node_modules', 'playwright', 'index.mjs')}`);
  }
}

function runNvidiaSmi(args) {
  try {
    return { available: true, text: execFileSync('nvidia-smi', args, { encoding: 'utf8', timeout: 10000 }) };
  } catch (error) {
    return { available: false, error: String(error?.message || error) };
  }
}

function hostGpuEvidence() {
  return {
    nvidiaSmi: runNvidiaSmi([
      '--query-gpu=index,name,driver_version,utilization.gpu,memory.used',
      '--format=csv,noheader,nounits',
    ]),
    nvidiaSmiPmon: runNvidiaSmi(['pmon', '-c', '1']),
  };
}

function competingComputeProcesses(...evidence) {
  const processes = [];
  const seen = new Set();
  for (const sample of evidence) {
    const text = sample?.nvidiaSmiPmon?.text || '';
    for (const line of text.split(/\r?\n/)) {
      const fields = line.trim().split(/\s+/);
      if (fields.length < 8 || !/^\d+$/.test(fields[0]) || !/^\d+$/.test(fields[1])) continue;
      const type = fields[2];
      if (!type.includes('C')) continue;
      const command = fields.slice(fields.length >= 10 ? 9 : 7).join(' ');
      if (/chrome|chromium|xorg|nvidia-smi/i.test(command)) continue;
      const key = `${fields[0]}:${fields[1]}:${command}`;
      if (seen.has(key)) continue;
      seen.add(key);
      processes.push({ gpu: Number(fields[0]), pid: Number(fields[1]), type, command });
    }
  }
  return processes;
}

function startPmonSampler() {
  let output = '';
  let errorOutput = '';
  let spawnError = null;
  let closed = false;
  const child = spawn('nvidia-smi', ['pmon', '-s', 'um', '-d', '1'], { stdio: ['ignore', 'pipe', 'pipe'] });
  child.stdout.on('data', (chunk) => { output += chunk.toString(); });
  child.stderr.on('data', (chunk) => { errorOutput += chunk.toString(); });
  child.on('error', (error) => { spawnError = String(error?.message || error); });
  child.on('close', () => { closed = true; });
  return {
    stop: async () => new Promise((resolve) => {
      if (closed) {
        resolve({ available: !spawnError, error: spawnError, text: output, errorText: errorOutput, sampleCount: output.split(/\r?\n/).filter((line) => /^\s*\d+\s+\S+/.test(line)).length });
        return;
      }
      child.once('close', () => resolve({
        available: !spawnError,
        error: spawnError,
        text: output,
        errorText: errorOutput,
        sampleCount: output.split(/\r?\n/).filter((line) => /^\s*\d+\s+\S+/.test(line)).length,
      }));
      child.kill('SIGTERM');
      setTimeout(() => { if (!closed) child.kill('SIGKILL'); }, 2000);
    }),
  };
}

function captureChromeProcess(browserServer) {
  const child = browserServer?.process?.() || null;
  let stdout = '';
  let stderr = '';
  child?.stdout?.on('data', (chunk) => { stdout += chunk.toString(); });
  child?.stderr?.on('data', (chunk) => { stderr += chunk.toString(); });
  return {
    available: Boolean(child?.stdout && child?.stderr),
    pid: child?.pid || null,
    snapshot: () => ({
      available: Boolean(child?.stdout && child?.stderr),
      pid: child?.pid || null,
      exitCode: child?.exitCode ?? null,
      signalCode: child?.signalCode ?? null,
      stdout,
      stderr,
    }),
  };
}

function mimeType(file) {
  return {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.bin': 'application/octet-stream',
  }[path.extname(file).toLowerCase()] || 'application/octet-stream';
}

function startStaticServer(bundleDir, mappings, requestedPort) {
  const resolvedBundle = path.resolve(bundleDir);
  const routes = mappings.map(([prefix, root]) => [prefix, path.resolve(root)]);
  const server = http.createServer((request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      if (url.pathname === '/favicon.ico') {
        response.writeHead(204);
        response.end();
        return;
      }
      let root = resolvedBundle;
      let relative = decodeURIComponent(url.pathname).replace(/^\/+/, '') || 'geometry_shell_hzb_benchmark.html';
      for (const [prefix, mappedRoot] of routes) {
        if (url.pathname === prefix.slice(0, -1) || url.pathname.startsWith(prefix)) {
          root = mappedRoot;
          relative = decodeURIComponent(url.pathname.slice(prefix.length)).replace(/^\/+/, '');
          break;
        }
      }
      const file = path.resolve(root, relative);
      if (file !== root && !file.startsWith(`${root}${path.sep}`)) throw new Error('path traversal');
      const stat = fs.statSync(file);
      const served = stat.isDirectory() ? path.join(file, 'index.html') : file;
      response.writeHead(200, { 'Content-Type': mimeType(served), 'Cache-Control': 'no-store' });
      fs.createReadStream(served).pipe(response);
    } catch {
      response.writeHead(404);
      response.end('Not found');
    }
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(requestedPort, '127.0.0.1', () => resolve(server));
  });
}

function buildBundle(bundleDir) {
  const parcel = path.join(VIEWER_ROOT, 'node_modules', '.bin', 'parcel');
  if (!fs.existsSync(parcel)) throw new Error(`${parcel} is missing; run npm ci in slm2viewer first.`);
  const result = spawnSync(parcel, [
    'build', 'geometry_shell_hzb_benchmark.html', '--no-cache', '--no-source-maps', '--no-minify', '--out-dir', bundleDir,
  ], { cwd: VIEWER_ROOT, encoding: 'utf8', timeout: 10 * 60 * 1000 });
  if (result.status !== 0) throw new Error(`Parcel HZB benchmark build failed:\n${result.stdout}\n${result.stderr}`);
}

function classifyBackend(info) {
  const values = [info?.vendor, info?.architecture, info?.device, info?.description, info?.renderer, info?.version];
  const gate = classifyGpuText(values);
  return { ...info, hardware: Boolean(values.join(' ').trim()) && !SOFTWARE_PATTERN.test(values.join(' ')), ...gate };
}

function formalGpuFailure(options, formalReady, failure = null) {
  if (failure) return failure;
  if (options.requireHardwareGpu && !formalReady) {
    return 'Geometry-shell HZB hardware GPU gate or execution evidence failed.';
  }
  return null;
}

function formalProtocolReady(options, workload) {
  if (!workload?.poseSelection?.representative) return false;
  if (options.poseIndexPlan) return options.timingRounds === 5;
  return options.timingRounds >= 1;
}

async function main() {
  const options = parseArgs(process.argv);
  const workloadInfo = buildWorkload(options);
  const bundleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'geometry-shell-hzb-bundle-'));
  buildBundle(bundleDir);
  const server = await startStaticServer(bundleDir, [
    ['/shell/', options.shellDir],
    ['/workload/', workloadInfo.outputDir],
  ], options.port);
  const port = server.address().port;
  const chrome = findChrome(options.chromeExe);
  const { chromium } = await loadPlaywright();
  const launchArgs = headlessWebGpuArgs({ screenSize: '1280,720', quietBrowser: true });
  const launchEnvironment = resolveVulkanEnvironment();
  const hostGpuBefore = hostGpuEvidence();
  const pmon = startPmonSampler();
  let browser = null;
  let browserServer = null;
  let chromeProcess = null;
  let pageConsole = [];
  let pageErrors = [];
  let result = null;
  let roundResults = [];
  let failure = null;
  try {
    browserServer = await chromium.launchServer({
      executablePath: chrome,
      headless: true,
      args: launchArgs,
      env: launchEnvironment,
      dumpio: false,
    });
    chromeProcess = captureChromeProcess(browserServer);
    browser = await chromium.connect(browserServer.wsEndpoint());
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    page.on('console', (message) => {
      const entry = { type: message.type(), text: message.text() };
      pageConsole.push(entry);
      console.log(`[geometry-shell-hzb-page:${entry.type}] ${entry.text}`);
    });
    page.on('pageerror', (error) => {
      const entry = String(error.stack || error);
      pageErrors.push(entry);
      console.error(`[geometry-shell-hzb-pageerror] ${entry}`);
    });
    await page.goto(`http://127.0.0.1:${port}/geometry_shell_hzb_benchmark.html`, {
      waitUntil: 'domcontentloaded',
      timeout: options.timeoutMs,
    });
    await page.waitForFunction(() => window.__geometryShellHZBReady === true, null, { timeout: options.timeoutMs });
    const roundConfig = {
      assetBaseUrl: `http://127.0.0.1:${port}/shell/`,
      workloadUrl: `http://127.0.0.1:${port}/workload/geometry_shell_hzb_workload.json`,
      mode: options.mode,
      width: options.width,
      height: options.height,
      fovYDeg: options.fovYDeg,
      near: options.near,
      far: options.far,
      depthBiasM: options.depthBiasM,
      warmupCount: options.warmupCount,
    };
    for (let round = 0; round < options.timingRounds; round += 1) {
      const browserResult = await page.evaluate(
        async (config) => window.runGeometryShellHZBBenchmark(config),
        roundConfig,
      );
      roundResults.push(attachWorkloadProvenance(
        attachRegionSamplingToResult(browserResult, workloadInfo.workload),
        workloadInfo.workload,
      ));
    }
    result = combineTimingResults(roundResults, workloadInfo.workload);
  } catch (error) {
    failure = String(error?.stack || error);
  } finally {
    const hostGpuDuring = {
      nvidiaSmi: hostGpuEvidence().nvidiaSmi,
      nvidiaSmiPmon: await pmon.stop(),
    };
    await browser?.close().catch(() => {});
    await browserServer?.close().catch(() => {});
    const hostGpuAfter = hostGpuEvidence();
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(bundleDir, { recursive: true, force: true });
    const chromeProcessEvidence = chromeProcess?.snapshot() || {
      available: false,
      pid: null,
      exitCode: null,
      signalCode: null,
      stdout: '',
      stderr: '',
    };
    const competingProcesses = competingComputeProcesses(hostGpuBefore, hostGpuDuring);
    const concurrentGpuWork = competingProcesses.length > 0;
    const adapterGate = classifyBackend(result?.gpuBackend || result?.adapterInfo || {});
    const webglGate = classifyBackend(result?.webglInfo || {});
    const gpuGate = {
      required: options.requireHardwareGpu,
      hardware: adapterGate.hardware && webglGate.hardware,
      adapter: adapterGate,
      webgl: webglGate,
    };
    const protocolReady = formalProtocolReady(options, workloadInfo.workload);
    const runtimeHealthy = Boolean(
      result && pageErrors.length === 0 && (result.gpuValidationErrors || []).length === 0,
    );
    const evidenceReady = hostGpuBefore.nvidiaSmi.available
      && hostGpuBefore.nvidiaSmiPmon.available
      && hostGpuDuring.nvidiaSmi.available
      && hostGpuDuring.nvidiaSmiPmon.available
      && hostGpuDuring.nvidiaSmiPmon.sampleCount > 0
      && hostGpuAfter.nvidiaSmi.available
      && hostGpuAfter.nvidiaSmiPmon.available
      && chromeProcessEvidence.available;
    const formalReady = Boolean(
      options.requireHardwareGpu && !failure && protocolReady && runtimeHealthy
        && gpuGate.hardware && evidenceReady && !concurrentGpuWork,
    );
    if (!failure && options.requireHardwareGpu && !gpuGate.hardware) {
      failure = 'Geometry-shell HZB WebGPU/WebGL hardware GPU gate failed.';
    } else if (!failure && options.requireHardwareGpu && !runtimeHealthy) {
      failure = 'Geometry-shell HZB browser runtime reported a page or WebGPU validation error.';
    }
    failure = formalGpuFailure(options, formalReady, failure);
    const executionClass = !options.requireHardwareGpu
      ? 'software-debug-only'
      : !gpuGate.hardware
        ? 'hardware-gate-failed'
        : failure
          ? 'hardware-gate-failed'
          : concurrentGpuWork
            ? 'hardware-smoke-concurrent'
            : formalReady
              ? 'formal-hardware-gpu'
              : 'hardware-gate-failed';
    const capture = {
      ...(result || {
        schema: 'geometry-shell-hzb-browser-result-v2',
        mode: options.mode,
        workload: workloadInfo.workload,
        provenance: workloadInfo.workload.provenance,
      }),
      capturedAt: new Date().toISOString(),
      shellDir: options.shellDir,
      datasetDir: options.datasetDir,
      regionDatasetDir: options.regionDatasetDir || null,
      output: options.output,
      regionSampling: workloadInfo.workload.regionSampling || null,
      protocolReady,
      gpuGate,
      formalReady,
      executionClass,
      gpuConcurrency: {
        concurrentComputeDetected: concurrentGpuWork,
        competingProcesses,
      },
      browserPageConsole: pageConsole,
      browserPageErrors: pageErrors,
      chromeProcess: chromeProcessEvidence,
      hostGpuBefore,
      hostGpuDuring,
      hostGpuAfter,
      browserLaunch: {
        executablePath: chrome,
        headless: true,
        dumpio: false,
        transport: 'launchServer+connect',
        args: launchArgs,
        vulkanIcd: launchEnvironment.VK_ICD_FILENAMES || null,
      },
      error: failure,
    };
    fs.mkdirSync(path.dirname(options.output), { recursive: true });
    fs.writeFileSync(options.output, `${JSON.stringify(capture, null, 2)}\n`, 'utf8');
    fs.writeFileSync(path.join(path.dirname(options.output), 'geometry_shell_hzb_gpu_evidence.json'), `${JSON.stringify({
      schema: 'geometry-shell-hzb-gpu-evidence-v1',
      gpuBackend: result?.gpuBackend || null,
      protocolReady,
      formalReady,
      gpuGate,
      browserLaunch: capture.browserLaunch,
      chromeProcess: chromeProcessEvidence,
      gpuConcurrency: capture.gpuConcurrency,
      hostGpuBefore,
      hostGpuDuring,
      hostGpuAfter,
      error: failure,
    }, null, 2)}\n`, 'utf8');
  }
  if (failure) throw new Error(failure);
  console.log(JSON.stringify({
    output: options.output,
    mode: result.mode,
    poseCount: result.samples.length,
    candidateCount: result.workload.candidateCount,
    summary: result.summary,
    gpuBackend: result.gpuBackend,
    adapterInfo: result.adapterInfo,
    webglInfo: result.webglInfo,
  }, null, 2));
}

export {
  buildRegionRows,
  buildWorkload,
  attachRegionSamplingToResult,
  attachWorkloadProvenance,
  combineTimingResults,
  formalGpuFailure,
  formalProtocolReady,
  makeRegionSamplingRecord,
  parseArgs,
  readPoseIndexPlan,
  selectRegionSubposes,
  summarizeRegionSampling,
  summarizeTimingRounds,
  summarizeTimingSamples,
};

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error?.stack || error);
    process.exitCode = 1;
  });
}
