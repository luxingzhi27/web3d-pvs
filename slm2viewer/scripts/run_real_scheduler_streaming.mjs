#!/usr/bin/env node
/*
 * Replay a fixed scheduler plan against the real GLB files.
 *
 * The harness deliberately uses the production GlbResourceScheduler.  The
 * local HTTP server only supplies reproducible bandwidth and real file bytes;
 * it does not change scheduler ordering or provide a cache between poses.
 */
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { once } from 'node:events';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

import {
  classifyGlbSchedule,
  GlbResourceScheduler,
} from '../src/GlbResourceScheduler.js';

const SCHEDULER_TIERS = Object.freeze(['urgent', 'warm', 'speculative']);

function option(name, fallback = null) {
  const prefix = `--${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

function hasFlag(name) {
  return process.argv.includes(`--${name}`);
}

function parsePositiveInt(value, label) {
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${label} must be a positive integer.`);
  return parsed;
}

function parseNumberList(value, label) {
  const result = String(value).split(',').filter((item) => item.trim()).map((item) => Number(item.trim()));
  if (!result.length || result.some((item) => !Number.isFinite(item) || item <= 0)) {
    throw new Error(`${label} must contain positive numbers.`);
  }
  return result;
}

function parseMethods(value) {
  const methods = String(value).split(',').map((item) => item.trim()).filter(Boolean);
  if (!methods.length) throw new Error('--methods selected no methods.');
  return methods;
}

async function readJson(filePath, label) {
  const resolved = path.resolve(filePath);
  try {
    return JSON.parse(await fsp.readFile(resolved, 'utf8'));
  } catch (error) {
    throw new Error(`Cannot read ${label} ${resolved}: ${error.message}`);
  }
}

function resolveInside(root, relativePath, label) {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(resolvedRoot, String(relativePath));
  if (resolved !== resolvedRoot && !resolved.startsWith(`${resolvedRoot}${path.sep}`)) {
    throw new Error(`${label} escapes the supplied root: ${relativePath}`);
  }
  return resolved;
}

async function loadInventory(runtimeMetaPath, glbIndexPath, glbRoot) {
  const [runtime, index] = await Promise.all([
    readJson(runtimeMetaPath, 'runtime metadata'),
    readJson(glbIndexPath, 'GLB index'),
  ]);
  if (!Array.isArray(index.entries) || !index.entries.length) {
    throw new Error('GLB index has no entries.');
  }
  const expectedCount = Number(runtime.globalGlbCount ?? index.entries.length);
  if (!Number.isInteger(expectedCount) || expectedCount !== index.entries.length) {
    throw new Error('Runtime metadata and GLB index GLB counts disagree.');
  }
  const assets = new Map();
  const seenPaths = new Set();
  for (const [ordinal, entry] of index.entries.entries()) {
    const id = Number(entry.globalId);
    if (!Number.isInteger(id) || id < 0 || assets.has(id)) {
      throw new Error(`Invalid or duplicate GLB ID at index row ${ordinal}.`);
    }
    const filePath = resolveInside(glbRoot, entry.path, `GLB ${id}`);
    const stat = await fsp.stat(filePath);
    if (!stat.isFile() || stat.size <= 0) throw new Error(`GLB ${id} is missing or empty: ${filePath}`);
    if (seenPaths.has(filePath)) throw new Error(`Two GLB IDs refer to one file: ${filePath}`);
    seenPaths.add(filePath);
    assets.set(id, { id, filePath, byteSize: Number(stat.size), originalRank: ordinal });
  }
  if (assets.size !== expectedCount || Array.from({ length: expectedCount }, (_, id) => id).some((id) => !assets.has(id))) {
    throw new Error('GLB index must cover a dense global ID inventory.');
  }
  return { runtime, index, assets };
}

function validatePosePlan(plan, methodName, assets) {
  const methodPlan = plan.methods?.[methodName];
  if (!methodPlan || methodPlan.status !== 'available') {
    throw new Error(`Method ${methodName} is unavailable in the scheduler plan.`);
  }
  if (Number(methodPlan.poseCount) !== 12 || !Array.isArray(methodPlan.poses) || methodPlan.poses.length !== 12) {
    throw new Error(`${methodName} must contain exactly 12 fixed test poses.`);
  }
  return methodPlan.poses.map((pose, poseIndex) => {
    const candidate = Array.from(pose.candidateGlbIds || []).map(Number);
    const ordered = Array.from(pose.orderedGlbIds || []).map(Number);
    const gt = Array.from(pose.gtGlbIds || []).map(Number);
    if (new Set(candidate).size !== candidate.length) {
      throw new Error(`${methodName} pose ${poseIndex} has an invalid candidate GLB set.`);
    }
    if (new Set(ordered).size !== ordered.length || ordered.length !== candidate.length
      || new Set(ordered).size !== new Set(candidate).size
      || ordered.some((id) => !new Set(candidate).has(id))) {
      throw new Error(`${methodName} pose ${poseIndex} changed the candidate GLB set.`);
    }
    if (new Set(gt).size !== gt.length) throw new Error(`${methodName} pose ${poseIndex} has duplicate GT GLBs.`);
    for (const id of candidate.concat(gt)) {
      if (!assets.has(id)) throw new Error(`${methodName} pose ${poseIndex} references unknown GLB ${id}.`);
    }
    const tiers = pose.tiers || {};
    const tierIds = [];
    for (const tier of SCHEDULER_TIERS) {
      if (!Array.isArray(tiers[tier])) throw new Error(`${methodName} pose ${poseIndex} is missing ${tier} tier.`);
      tierIds.push(...tiers[tier].map(Number));
    }
    if (tierIds.length !== candidate.length || new Set(tierIds).size !== candidate.length
      || tierIds.some((id) => !new Set(candidate).has(id))) {
      throw new Error(`${methodName} pose ${poseIndex} has invalid scheduler tier coverage.`);
    }
    return {
      poseId: Number(pose.poseId),
      ordinal: Number(pose.ordinal ?? poseIndex),
      candidateGlbIds: candidate,
      gtGlbIds: gt,
      orderedGlbIds: ordered,
      tiers: Object.fromEntries(SCHEDULER_TIERS.map((tier) => [tier, tiers[tier].map(Number)])),
    };
  });
}

function modelInfoFor(id, score) {
  return {
    id,
    weight: score,
    score,
    idMode: 'global-glb',
  };
}

function schedulerGroupsForPose(pose, assets) {
  const orderIndex = new Map(pose.orderedGlbIds.map((id, index) => [id, index]));
  const scoreFor = (id) => pose.orderedGlbIds.length - orderIndex.get(id);
  const modelInfos = pose.tiers.urgent.concat(pose.tiers.warm)
    .map((id) => modelInfoFor(id, scoreFor(id)));
  const prefetchInfos = pose.tiers.speculative.map((id) => modelInfoFor(id, scoreFor(id)));
  const classified = classifyGlbSchedule(modelInfos, prefetchInfos, pose.tiers.urgent);
  for (const tier of SCHEDULER_TIERS) {
    const expected = new Set(pose.tiers[tier]);
    const actual = classified[tier].map((item) => Number(item.id));
    if (actual.length !== expected.size || actual.some((id) => !expected.has(id))) {
      throw new Error(`Production scheduler classifier changed the ${tier} candidate set.`);
    }
  }
  return classified;
}

function scheduleEntry(item, assets) {
  const id = Number(item.id);
  const asset = assets.get(id);
  if (!asset) throw new Error(`Cannot schedule unknown GLB ${id}.`);
  const score = Number(item.score ?? item.weight ?? 0);
  /* The production scheduler requires an opaque hash field. This is a
   * resource key, not a content hash and is never computed or reported as one. */
  return {
    hash: `resource-${id}`,
    glbId: id,
    score,
    modelInfo: {
      ...item,
      id,
      glbId: id,
      byteSize: asset.byteSize,
      prefetch: item.prefetch !== false,
    },
  };
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, milliseconds)));
}

function parseGlbContainer(buffer) {
  if (buffer.byteLength < 20) throw new Error('GLB payload is shorter than its header and JSON chunk.');
  const magic = buffer.readUInt32LE(0);
  const version = buffer.readUInt32LE(4);
  const declaredLength = buffer.readUInt32LE(8);
  if (magic !== 0x46546c67 || version !== 2) throw new Error('GLB header is invalid.');
  if (declaredLength > buffer.byteLength) throw new Error('GLB is truncated.');
  let offset = 12;
  let jsonChunkBytes = 0;
  while (offset + 8 <= buffer.byteLength) {
    const chunkLength = buffer.readUInt32LE(offset);
    const chunkType = buffer.readUInt32LE(offset + 4);
    offset += 8;
    if (offset + chunkLength > buffer.byteLength) throw new Error('GLB chunk is truncated.');
    if (chunkType === 0x4e4f534a) {
      const text = new TextDecoder().decode(buffer.subarray(offset, offset + chunkLength)).replace(/\0+$/g, '').trim();
      JSON.parse(text);
      jsonChunkBytes = chunkLength;
    }
    offset += chunkLength;
  }
  if (!jsonChunkBytes) throw new Error('GLB has no JSON chunk.');
  return { declaredLength, jsonChunkBytes };
}

function createBandwidthLimiter() {
  let bytesPerMillisecond = 1;
  let nextAvailableAt = performance.now();
  return {
    setMbps(mbps) {
      bytesPerMillisecond = Number(mbps) * 1_000_000 / 8 / 1_000;
      nextAvailableAt = performance.now();
    },
    async reserve(byteCount) {
      const now = performance.now();
      const start = Math.max(now, nextAvailableAt);
      nextAvailableAt = start + Number(byteCount) / bytesPerMillisecond;
      await sleep(start - now);
    },
  };
}

async function startAssetServer(assets) {
  const limiter = createBandwidthLimiter();
  const server = http.createServer(async (request, response) => {
    const match = /^\/asset\/(\d+)$/.exec(new URL(request.url, 'http://127.0.0.1').pathname);
    const id = match ? Number(match[1]) : -1;
    const asset = assets.get(id);
    if (request.method !== 'GET' || !asset) {
      response.statusCode = 404;
      response.end();
      return;
    }
    response.writeHead(200, {
      'Content-Length': asset.byteSize,
      'Content-Type': 'model/gltf-binary',
      'Cache-Control': 'no-store',
      Connection: 'close',
    });
    const stream = fs.createReadStream(asset.filePath, { highWaterMark: 64 * 1024 });
    try {
      for await (const chunk of stream) {
        if (response.destroyed) break;
        await limiter.reserve(chunk.length);
        if (!response.write(chunk)) await once(response, 'drain');
      }
      if (!response.destroyed) response.end();
    } catch (error) {
      stream.destroy();
      if (!response.destroyed) response.destroy(error);
    }
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  return {
    server,
    setMbps: (mbps) => limiter.setMbps(mbps),
    baseUrl: `http://127.0.0.1:${address.port}`,
  };
}

function summarizeNumbers(values) {
  const finite = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!finite.length) return { count: 0, mean: null, median: null, p95: null };
  const quantile = (fraction) => finite[Math.min(finite.length - 1, Math.floor(fraction * (finite.length - 1)))];
  return {
    count: finite.length,
    mean: finite.reduce((sum, value) => sum + value, 0) / finite.length,
    median: quantile(0.5),
    p95: quantile(0.95),
  };
}

function currentCoverage(loaded, gt, assets) {
  const required = gt.filter((id) => assets.has(id));
  const loadedRequired = required.filter((id) => loaded.has(id));
  const totalBytes = loadedRequired.reduce((sum, id) => sum + assets.get(id).byteSize, 0);
  return {
    coverage: required.length ? loadedRequired.length / required.length : 1,
    requiredCount: required.length,
    loadedRequiredCount: loadedRequired.length,
    loadedRequiredBytes: totalBytes,
  };
}

async function runPose({ pose, assets, server, bandwidthMbps, concurrency, parseMsPerMiB, mountMs, stopAfterFirstFrame, runId }) {
  const groups = schedulerGroupsForPose(pose, assets);
  const scheduler = new GlbResourceScheduler({
    clock: () => performance.now(),
    maxRetries: 0,
  });
  scheduler.setPlan(Object.fromEntries(
    SCHEDULER_TIERS.map((tier) => [tier, groups[tier].map((item) => scheduleEntry(item, assets))]),
  ), 1);
  server.setMbps(bandwidthMbps);
  const start = performance.now();
  const loaded = new Set();
  const active = new Map();
  const controllers = new Map();
  const completeEvents = [];
  let scheduledCount = 0;
  let downloadedBytes = 0;
  let firstFrame = null;
  let failureCount = 0;
  let cancellationCount = 0;

  if (pose.gtGlbIds.length === 0) {
    firstFrame = {
      reached: true,
      elapsedMs: 0,
      at: start,
      downloadedBytes: 0,
      loadedGlbCount: 0,
      coverage: 1,
    };
  }

  const onComplete = (result) => {
    if (!result.ok) {
      if (result.cancelled) cancellationCount += 1;
      else failureCount += 1;
      return;
    }
    loaded.add(result.glbId);
    downloadedBytes += result.byteSize;
    completeEvents.push(result);
    const coverage = currentCoverage(loaded, pose.gtGlbIds, assets);
    if (firstFrame == null && coverage.loadedRequiredCount === coverage.requiredCount) {
      firstFrame = {
        reached: true,
        elapsedMs: result.residentAt - start,
        at: result.residentAt,
        downloadedBytes,
        loadedGlbCount: loaded.size,
        coverage: coverage.coverage,
      };
    }
  };

  async function processEntry(entry) {
    const glbId = Number(entry.glbId);
    const controller = new AbortController();
    controllers.set(entry.hash, controller);
    const fetchStartedAt = performance.now();
    try {
      const response = await fetch(`${server.baseUrl}/asset/${glbId}?run=${encodeURIComponent(runId)}`, {
        cache: 'no-store',
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = Buffer.from(await response.arrayBuffer());
      if (payload.byteLength !== assets.get(glbId).byteSize) {
        throw new Error(`GLB ${glbId} response has ${payload.byteLength} bytes, expected ${assets.get(glbId).byteSize}`);
      }
      const fetchCompletedAt = performance.now();
      scheduler.markParsing(entry.hash);
      const parseStartedAt = performance.now();
      const parseInfo = parseGlbContainer(payload);
      if (parseMsPerMiB > 0) await sleep(parseMsPerMiB * payload.byteLength / (1024 * 1024));
      const parseCompletedAt = performance.now();
      scheduler.markMounting(entry.hash);
      const mountStartedAt = performance.now();
      if (mountMs > 0) await sleep(mountMs);
      scheduler.markResident(entry.hash);
      const residentAt = performance.now();
      return {
        ok: true,
        glbId,
        byteSize: payload.byteLength,
        fetchMs: fetchCompletedAt - fetchStartedAt,
        parseMs: parseCompletedAt - parseStartedAt,
        mountMs: residentAt - mountStartedAt,
        residentAt,
        parseInfo,
        tier: entry.tier,
      };
    } catch (error) {
      scheduler.markFailed(entry.hash, error);
      return {
        ok: false,
        glbId,
        error: String(error?.message || error),
        cancelled: Boolean(controller.signal.aborted),
      };
    } finally {
      controllers.delete(entry.hash);
    }
  }

  function launch(entry) {
    scheduledCount += 1;
    const task = processEntry(entry).then((result) => {
      active.delete(entry.hash);
      onComplete(result);
      return result;
    });
    active.set(entry.hash, task);
  }

  let stoppedAfterFirstFrame = false;
  while (scheduler.hasPendingWork() || active.size > 0) {
    while (active.size < concurrency && scheduler.hasReadyFetch()) {
      const entry = scheduler.takeNext();
      if (!entry) break;
      launch(entry);
    }
    if (firstFrame && stopAfterFirstFrame) {
      stoppedAfterFirstFrame = true;
      for (const controller of controllers.values()) controller.abort();
      await Promise.allSettled(Array.from(active.values()));
      break;
    }
    if (active.size > 0) {
      await Promise.race(Array.from(active.values()));
      continue;
    }
    const wakeDelay = scheduler.nextWakeDelay();
    if (wakeDelay == null) break;
    await sleep(Math.max(1, wakeDelay));
  }
  const end = performance.now();
  const coverage = currentCoverage(loaded, pose.gtGlbIds, assets);
  const requiredBytes = pose.gtGlbIds.filter((id) => assets.has(id)).reduce((sum, id) => sum + assets.get(id).byteSize, 0);
  return {
    poseId: pose.poseId,
    ordinal: pose.ordinal,
    bandwidthMbps,
    cacheMode: 'strict_cold_cache_per_pose',
    startup100Enabled: false,
    schedulerTiers: SCHEDULER_TIERS,
    candidateGlbCount: pose.candidateGlbIds.length,
    candidateGlbBytes: pose.candidateGlbIds.reduce((sum, id) => sum + assets.get(id).byteSize, 0),
    gtGlbCount: pose.gtGlbIds.length,
    candidateGtCount: pose.gtGlbIds.filter((id) => assets.has(id)).length,
    coverageCeiling: pose.gtGlbIds.length ? pose.gtGlbIds.filter((id) => pose.candidateGlbIds.includes(id)).length / pose.gtGlbIds.length : 1,
    firstFrame: firstFrame || {
      reached: false,
      elapsedMs: null,
      downloadedBytes: null,
      loadedGlbCount: null,
      coverage: coverage.coverage,
    },
    downloadedBytes,
    completedGlbCount: loaded.size,
    scheduledGlbCount: scheduledCount,
    stoppedAfterFirstFrame,
    elapsedMs: end - start,
    coverageAtStop: coverage.coverage,
    wasteBeforeFirstFrameBytes: firstFrame ? Math.max(0, firstFrame.downloadedBytes - requiredBytes) : null,
    failureCount,
    cancellationCount,
    errors: completeEvents.length + failureCount === scheduledCount
      ? []
      : ['scheduler run ended with inconsistent event accounting'],
    schedulerSnapshot: scheduler.getSnapshot(),
    phaseMs: {
      fetchMean: summarizeNumbers(completeEvents.map((item) => item.fetchMs)).mean,
      parseMean: summarizeNumbers(completeEvents.map((item) => item.parseMs)).mean,
      mountMean: summarizeNumbers(completeEvents.map((item) => item.mountMs)).mean,
    },
  };
}

function parseMethodAssets(values) {
  const assets = new Map();
  for (const value of values) {
    const separator = String(value).indexOf('=');
    if (separator <= 0) throw new Error('--method-asset must use method=path.');
    const method = String(value).slice(0, separator).trim();
    const filePath = path.resolve(String(value).slice(separator + 1).trim());
    if (!method || !filePath) throw new Error('--method-asset contains an empty method or path.');
    if (!assets.has(method)) assets.set(method, []);
    assets.get(method).push(filePath);
  }
  return assets;
}

async function loadMethodAssets(method, paths) {
  const startedAt = performance.now();
  let byteCount = 0;
  for (const filePath of paths || []) {
    const payload = await fsp.readFile(filePath);
    byteCount += payload.byteLength;
  }
  return {
    files: (paths || []).map((filePath) => path.resolve(filePath)),
    byteCount,
    loadMs: performance.now() - startedAt,
    status: 'loaded',
    method,
  };
}

function summarizeRuns(runs) {
  const keys = new Map();
  for (const run of runs) {
    const key = `${run.method}|${run.bandwidthMbps}`;
    if (!keys.has(key)) keys.set(key, []);
    keys.get(key).push(run);
  }
  return Array.from(keys.entries()).map(([key, group]) => {
    const [method, bandwidthText] = key.split('|');
    return {
      method,
      bandwidthMbps: Number(bandwidthText),
      poseCount: group.length,
      firstFrameReachedRatio: group.filter((run) => run.firstFrame.reached).length / group.length,
      firstFrameMs: summarizeNumbers(group.map((run) => run.firstFrame.elapsedMs)),
      firstFrameBytes: summarizeNumbers(group.map((run) => run.firstFrame.downloadedBytes)),
      coverageAtStop: summarizeNumbers(group.map((run) => run.coverageAtStop)),
      coverageCeiling: summarizeNumbers(group.map((run) => run.coverageCeiling)),
      candidateGlbCount: summarizeNumbers(group.map((run) => run.candidateGlbCount)),
      candidateGlbBytes: summarizeNumbers(group.map((run) => run.candidateGlbBytes)),
      downloadedBytes: summarizeNumbers(group.map((run) => run.downloadedBytes)),
      completedGlbCount: summarizeNumbers(group.map((run) => run.completedGlbCount)),
      wasteBeforeFirstFrameBytes: summarizeNumbers(group.map((run) => run.wasteBeforeFirstFrameBytes)),
      failureCount: group.reduce((sum, run) => sum + run.failureCount, 0),
      cancellationCount: group.reduce((sum, run) => sum + run.cancellationCount, 0),
    };
  });
}

async function main() {
  if (process.argv.some((value) => value === '--startup-100' || value.startsWith('--startup-100=')
    || value === '--startup100' || value.startsWith('--startup100='))) {
    throw new Error('The formal scheduler replay forbids startup-100.');
  }
  const planPath = option('plan');
  const runtimeMetaPath = option('runtime-meta');
  const glbIndexPath = option('glb-index');
  const glbRoot = option('glb-root');
  const outputDir = option('output-dir');
  if (!planPath || !runtimeMetaPath || !glbIndexPath || !glbRoot || !outputDir) {
    throw new Error('Required: --plan, --runtime-meta, --glb-index, --glb-root, --output-dir.');
  }
  const concurrency = parsePositiveInt(option('concurrency', '4'), '--concurrency');
  const repeats = parsePositiveInt(option('repeats', '3'), '--repeats');
  const bandwidths = parseNumberList(option('bandwidths', '25,50'), '--bandwidths');
  const parseMsPerMiB = Number(option('parse-ms-per-mib', '0'));
  const mountMs = Number(option('mount-ms', '0'));
  if (!Number.isFinite(parseMsPerMiB) || parseMsPerMiB < 0 || !Number.isFinite(mountMs) || mountMs < 0) {
    throw new Error('Parse and mount phase delays must be finite and non-negative.');
  }
  const plan = await readJson(planPath, 'scheduler plan');
  if (plan.schema !== 'pvs-real-scheduler-plan-v1') throw new Error('Unsupported scheduler plan schema.');
  if (plan.split !== 'test' || plan.testRead !== true) throw new Error('Real scheduler replay requires a test plan.');
  if (Number(plan.poseCount) !== 12 || !Array.isArray(plan.poseIds) || plan.poseIds.length !== 12) {
    throw new Error('Real scheduler replay requires exactly 12 fixed test poses.');
  }
  if (plan.startup100?.enabled !== false || plan.startup100?.startupTierUsed !== false) {
    throw new Error('Scheduler plan does not explicitly disable startup-100.');
  }
  if (!Array.isArray(plan.tiering?.schedulerTiers)
    || JSON.stringify(plan.tiering.schedulerTiers) !== JSON.stringify(SCHEDULER_TIERS)) {
    throw new Error('Scheduler plan does not declare urgent/warm/speculative tiers.');
  }
  const inventory = await loadInventory(runtimeMetaPath, glbIndexPath, glbRoot);
  const requestedMethods = option('methods', '')
    ? parseMethods(option('methods'))
    : Object.keys(plan.methods || {});
  const methodAssets = parseMethodAssets(process.argv
    .filter((value) => value.startsWith('--method-asset='))
    .map((value) => value.slice('--method-asset='.length)));
  const server = await startAssetServer(inventory.assets);
  const runs = [];
  const methodAssetResults = {};
  try {
    for (const method of requestedMethods) {
      const poses = validatePosePlan(plan, method, inventory.assets);
      methodAssetResults[method] = await loadMethodAssets(method, methodAssets.get(method) || []);
      for (const bandwidthMbps of bandwidths) {
        for (let repeat = 0; repeat < repeats; repeat += 1) {
          for (const pose of poses) {
            const result = await runPose({
              pose,
              assets: inventory.assets,
              server,
              bandwidthMbps,
              concurrency,
              parseMsPerMiB,
              mountMs,
              stopAfterFirstFrame: !hasFlag('continue-after-first-frame'),
              runId: `${method}-${bandwidthMbps}-${repeat}-${pose.ordinal}`,
            });
            runs.push({ method, repeat: repeat + 1, ...result });
            console.log(`[real-streaming] method=${method} bandwidth=${bandwidthMbps}Mbps repeat=${repeat + 1} pose=${pose.poseId} firstFrame=${result.firstFrame.reached ? result.firstFrame.elapsedMs.toFixed(1) : 'unreached'}ms downloaded=${result.downloadedBytes}`);
          }
        }
      }
    }
  } finally {
    await new Promise((resolve) => server.server.close(resolve));
  }

  const resolvedOutput = path.resolve(outputDir);
  await fsp.mkdir(resolvedOutput, { recursive: true });
  const runsPath = path.join(resolvedOutput, 'real_scheduler_runs.jsonl');
  await fsp.writeFile(runsPath, `${runs.map((run) => JSON.stringify(run)).join('\n')}\n`);
  const summary = {
    schema: 'pvs-real-scheduler-streaming-summary-v1',
    scheduler: 'GlbResourceScheduler',
    schedulerTiers: SCHEDULER_TIERS,
    split: plan.split,
    testRead: true,
    poseCount: 12,
    poseIds: plan.poseIds.map(Number),
    methods: requestedMethods,
    bandwidthsMbps: bandwidths,
    repeats,
    concurrency,
    stopAfterFirstFrame: !hasFlag('continue-after-first-frame'),
    parseMsPerMiB,
    mountMs,
    startup100Enabled: false,
    startupTierUsed: false,
    cacheMode: 'strict_cold_cache_per_pose',
    arrivalSemantics: 'a GLB becomes resident and contributes only after the complete HTTP response, container parse, and mount phase',
    source: {
      plan: path.resolve(planPath),
      runtimeMeta: path.resolve(runtimeMetaPath),
      glbIndex: path.resolve(glbIndexPath),
      glbRoot: path.resolve(glbRoot),
    },
    methodAssets: methodAssetResults,
    runCount: runs.length,
    runsFile: path.basename(runsPath),
    summaries: summarizeRuns(runs),
    limitations: [
      'The driver uses real GLB bytes and the production scheduler, but its parse/mount phase validates the GLB container rather than constructing a Three.js scene.',
      'The fixed plan is threshold-free; frozen threshold filtering is evaluated only by the separate offline filtering summary.',
      'No content hashes are computed or reported; scheduler resource keys are opaque per-GLB identifiers required by its existing API.',
    ],
  };
  await fsp.writeFile(path.join(resolvedOutput, 'real_scheduler_summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
  const manifest = {
    schema: 'pvs-real-scheduler-streaming-manifest-v1',
    summary: 'real_scheduler_summary.json',
    runs: 'real_scheduler_runs.jsonl',
    plan: path.resolve(planPath),
    startup100Enabled: false,
    schedulerTiers: SCHEDULER_TIERS,
    poseCount: 12,
    bandwidthsMbps: bandwidths,
    repeats,
  };
  await fsp.writeFile(path.join(resolvedOutput, 'real_scheduler_manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(JSON.stringify({ outputDir: resolvedOutput, runCount: runs.length, summaries: summary.summaries }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(`[real-streaming] ${error.stack || error.message || error}`);
    process.exitCode = 1;
  });
}

export {
  parseGlbContainer,
  schedulerGroupsForPose,
  summarizeNumbers,
};
