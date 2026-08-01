import { Box3, Frustum, Matrix4, PerspectiveCamera, Quaternion, Vector3 } from 'three';
import { InstancePVS } from './InstancePVS.js';
import { FRONTEND_RENDER_FOV_Y_DEG, MODEL_INPUT_FOV_Y_DEG } from './neuralPvsFovProtocol.js';

const _box = new Box3();
const _matrix = new Matrix4();
const _frustum = new Frustum();
const _quat = new Quaternion();
const _forward = new Vector3();

let state = {
  assetBaseUrl: '',
  runtimeMetaUrl: '',
  meta: null,
  runtimeMeta: null,
  pvs: null,
  m12Pvs: null,
  componentAabbs: null,
  instanceToGlobalGlb: null,
  localToGlobalInstance: null,
  globalToLocalInstance: null,
  ready: false,
  backend: 'uninitialized',
  initTimings: null,
  forceFallback: false,
  maxImmediate: 384,
  maxPrefetch: 2048,
  prefetchThreshold: 0.04,
  downloadPlanMode: 'viewcell-priority',
  assetVersion: null,
};

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function clampNumber(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const text = await response.text();
  if (text.trimStart().charAt(0) === '<') {
    throw new Error(`Expected JSON from ${url}, but received HTML. Check the asset path or server fallback.`);
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`Failed to parse JSON ${url}: ${error && error.message ? error.message : String(error)}`);
  }
}

function versionedAssetUrl(path) {
  if (!state.assetVersion) return path;
  return `${path}${path.indexOf('?') >= 0 ? '&' : '?'}v=${encodeURIComponent(state.assetVersion)}`;
}

function buildFallbackTables(meta, runtimeMeta) {
  const records = Array.isArray(runtimeMeta?.componentRecords) ? runtimeMeta.componentRecords : [];
  const numInstances = Number(meta?.numInstances || records.length || 0);
  const aabbs = new Float32Array(numInstances * 6);
  const instanceToGlb = new Uint32Array(numInstances);
  const globalIds = Array.isArray(meta?.globalInstanceIds) ? meta.globalInstanceIds : null;
  const globalToLocal = new Map();
  for (let localId = 0; localId < numInstances; localId += 1) {
    const globalId = globalIds && globalIds[localId] != null ? Number(globalIds[localId]) : localId;
    if (Number.isFinite(globalId) && globalId >= 0) globalToLocal.set(globalId, localId);
  }
  for (let i = 0; i < numInstances; i += 1) {
    instanceToGlb[i] = 0xffffffff;
    const base = i * 6;
    aabbs[base + 0] = Number.POSITIVE_INFINITY;
    aabbs[base + 1] = Number.POSITIVE_INFINITY;
    aabbs[base + 2] = Number.POSITIVE_INFINITY;
    aabbs[base + 3] = Number.NEGATIVE_INFINITY;
    aabbs[base + 4] = Number.NEGATIVE_INFINITY;
    aabbs[base + 5] = Number.NEGATIVE_INFINITY;
  }

  if (Array.isArray(meta?.instanceToGlobalGlb)) {
    const limit = Math.min(numInstances, meta.instanceToGlobalGlb.length);
    for (let i = 0; i < limit; i += 1) {
      const glb = Number(meta.instanceToGlobalGlb[i]);
      instanceToGlb[i] = Number.isFinite(glb) && glb >= 0 ? glb >>> 0 : 0xffffffff;
    }
  }

  for (const record of records) {
    const globalId = Number(record?.componentGlobalId);
    if (!Number.isFinite(globalId)) continue;
    const id = globalToLocal.has(globalId)
      ? globalToLocal.get(globalId)
      : (globalIds ? null : globalId);
    if (id == null || id < 0 || id >= numInstances) continue;
    if (record.globalGlbId != null) {
      const glb = Number(record.globalGlbId);
      if (Number.isFinite(glb) && glb >= 0) instanceToGlb[id] = glb >>> 0;
    }
    const bounds = record.bounds || null;
    if (!bounds) continue;
    let min = null;
    let max = null;
    if (Array.isArray(bounds.min) && Array.isArray(bounds.max)) {
      min = bounds.min;
      max = bounds.max;
    } else if (Array.isArray(bounds.center) && Array.isArray(bounds.size)) {
      min = [
        Number(bounds.center[0]) - Number(bounds.size[0]) * 0.5,
        Number(bounds.center[1]) - Number(bounds.size[1]) * 0.5,
        Number(bounds.center[2]) - Number(bounds.size[2]) * 0.5,
      ];
      max = [
        Number(bounds.center[0]) + Number(bounds.size[0]) * 0.5,
        Number(bounds.center[1]) + Number(bounds.size[1]) * 0.5,
        Number(bounds.center[2]) + Number(bounds.size[2]) * 0.5,
      ];
    }
    if (!min || !max) continue;
    const base = id * 6;
    aabbs[base + 0] = Number(min[0]);
    aabbs[base + 1] = Number(min[1]);
    aabbs[base + 2] = Number(min[2]);
    aabbs[base + 3] = Number(max[0]);
    aabbs[base + 4] = Number(max[1]);
    aabbs[base + 5] = Number(max[2]);
  }

  return { aabbs, instanceToGlb, globalToLocal };
}

function buildLocalToGlobal(meta) {
  const count = Number(meta?.numInstances || 0);
  const configured = Array.isArray(meta?.globalInstanceIds) ? meta.globalInstanceIds : null;
  const localToGlobal = new Array(count);
  const globalToLocal = new Map();
  for (let localId = 0; localId < count; localId += 1) {
    const value = configured && configured[localId] != null ? Number(configured[localId]) : localId;
    const globalId = Number.isFinite(value) && value >= 0 ? value : localId;
    localToGlobal[localId] = globalId;
    globalToLocal.set(globalId, localId);
  }
  return { localToGlobal, globalToLocal };
}

function globalIdsToLocal(ids) {
  const result = [];
  const map = state.globalToLocalInstance || new Map();
  for (const value of ids || []) {
    const localId = map.get(Number(value));
    if (localId != null) result.push(localId);
  }
  return Array.from(new Set(result));
}

function localIdToGlobal(localId) {
  const value = state.localToGlobalInstance?.[Number(localId)];
  return value == null ? Number(localId) : Number(value);
}

function buildCamera(snapshot, options = {}) {
  const fov = clampNumber(options.fov, FRONTEND_RENDER_FOV_Y_DEG);
  const aspect = Math.max(1e-4, clampNumber(options.aspect, clampNumber(snapshot.aspect, 16 / 9)));
  const near = Math.max(1e-4, clampNumber(snapshot.near, 0.1));
  const far = Math.max(near + 1, clampNumber(snapshot.far, 20000));
  const camera = new PerspectiveCamera(fov, aspect, near, far);
  camera.position.set(
    clampNumber(snapshot.position?.[0], 0),
    clampNumber(snapshot.position?.[1], 0),
    clampNumber(snapshot.position?.[2], 0),
  );
  camera.quaternion.set(
    clampNumber(snapshot.quaternion?.[0], 0),
    clampNumber(snapshot.quaternion?.[1], 0),
    clampNumber(snapshot.quaternion?.[2], 0),
    clampNumber(snapshot.quaternion?.[3], 1),
  );
  if (options.position) {
    camera.position.set(options.position[0], options.position[1], options.position[2]);
  }
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return camera;
}

function buildPredictionCamera(snapshot, meta) {
  const mode = String(meta?.predictionCameraMode || '').toLowerCase();
  // The runtime has exactly two FOVs: 60° for the displayed camera and 66°
  // for the model query camera. Do not reintroduce an exported PVS/buffer FOV.
  const fovY = MODEL_INPUT_FOV_Y_DEG;
  const aspect = clampNumber(snapshot.aspect, 16 / 9);
  const backOffset = mode === 'viewcell-back-camera'
    ? Math.max(0, clampNumber(meta?.pvsBackOffsetM, 0))
    : 0;
  _quat.set(
    clampNumber(snapshot.quaternion?.[0], 0),
    clampNumber(snapshot.quaternion?.[1], 0),
    clampNumber(snapshot.quaternion?.[2], 0),
    clampNumber(snapshot.quaternion?.[3], 1),
  );
  _forward.set(0, 0, -1).applyQuaternion(_quat).normalize();
  const position = [
    clampNumber(snapshot.position?.[0], 0) - _forward.x * backOffset,
    clampNumber(snapshot.position?.[1], 0) - _forward.y * backOffset,
    clampNumber(snapshot.position?.[2], 0) - _forward.z * backOffset,
  ];
  return buildCamera(snapshot, { fov: fovY, aspect, position });
}

function frustumForCamera(camera) {
  camera.updateMatrixWorld(true);
  _matrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  return _frustum.setFromProjectionMatrix(_matrix);
}

function intersectsComponent(componentId, frustum) {
  const aabbs = state.componentAabbs;
  const base = componentId * 6;
  if (!aabbs || base + 5 >= aabbs.length) return true;
  const minX = aabbs[base + 0];
  const minY = aabbs[base + 1];
  const minZ = aabbs[base + 2];
  const maxX = aabbs[base + 3];
  const maxY = aabbs[base + 4];
  const maxZ = aabbs[base + 5];
  if (![minX, minY, minZ, maxX, maxY, maxZ].every(Number.isFinite)) return true;
  _box.min.set(minX, minY, minZ);
  _box.max.set(maxX, maxY, maxZ);
  return frustum.intersectsBox(_box);
}

function filterComponentsByFrustum(componentIds, camera) {
  const frustum = frustumForCamera(camera);
  const out = [];
  for (const id of componentIds || []) {
    const componentId = Number(id);
    if (Number.isFinite(componentId) && componentId >= 0 && intersectsComponent(componentId, frustum)) {
      out.push(componentId >>> 0);
    }
  }
  return out;
}

function componentIdsToGlbIds(componentIds) {
  const set = new Set();
  const map = state.instanceToGlobalGlb;
  for (const id of componentIds || []) {
    const glb = map && id < map.length ? map[id] : 0xffffffff;
    if (glb !== 0xffffffff) set.add(glb >>> 0);
  }
  return Array.from(set).sort((a, b) => a - b);
}

function normalizeCandidateSelection(value, fallbackCount = null) {
  if (!value || typeof value !== 'object') return null;
  const numberOrNull = (item) => {
    const number = Number(item);
    return Number.isFinite(number) ? number : null;
  };
  return {
    source: value.source ? String(value.source) : null,
    candidateCount: numberOrNull(value.candidateCount) ?? numberOrNull(fallbackCount),
    queryCellCount: numberOrNull(value.queryCellCount),
    indexedInstanceCount: numberOrNull(value.indexedInstanceCount),
    overflowInstanceCount: numberOrNull(value.overflowInstanceCount),
  };
}

function candidatePriority(item, fallback = 0) {
  const downloadPriority = Number(item?.downloadPriority);
  if (Number.isFinite(downloadPriority)) return downloadPriority;
  const confidence = Number(item?.confidence);
  if (Number.isFinite(confidence)) return confidence;
  const importance = Number(item?.importance);
  if (Number.isFinite(importance)) return importance;
  return fallback;
}

function compareCandidatesByPriority(a, b) {
  return candidatePriority(b) - candidatePriority(a)
    || Number(a?.globalGlbId || 0) - Number(b?.globalGlbId || 0);
}

function candidateFromVisibleComponents(componentIds) {
  const byGlb = new Map();
  const map = state.instanceToGlobalGlb;
  for (const id of componentIds || []) {
    const glb = map && id < map.length ? map[id] : 0xffffffff;
    if (glb === 0xffffffff) continue;
    let item = byGlb.get(glb);
    if (!item) {
      item = {
        globalGlbId: glb >>> 0,
        confidence: 1,
        importance: 1,
        downloadPriority: 1,
        visibilityScore: 1,
        prioritySource: 'aabb-fallback',
        sourceComponentIds: [],
        sourceComponents: [],
      };
      byGlb.set(glb, item);
    }
    item.sourceComponentIds.push(id >>> 0);
    item.sourceComponents.push({
      componentId: id >>> 0,
      confidence: 1,
      importance: 1,
      downloadPriority: 1,
      visibilityScore: 1,
      visible: true,
    });
  }
  return Array.from(byGlb.values()).sort(compareCandidatesByPriority);
}

function normalizeDownloadPlanMode(value) {
  return value === 'raw-visible' ? 'raw-visible' : 'viewcell-priority';
}

function splitDownloadPlan(visibleCandidates, prefetchCandidates, activeGlbIds = [], options = {}) {
  const immediate = [];
  const prefetch = [];
  const seen = new Set();
  const activeSet = activeGlbIds instanceof Set ? activeGlbIds : new Set(activeGlbIds || []);
  const maxImmediate = Math.max(0, Number(state.maxImmediate || 0));
  const maxPrefetch = Math.max(0, Number(state.maxPrefetch || 0));
  const modelPriority = Boolean(options.modelPriority);
  const sortedVisible = (visibleCandidates || []).slice().sort(compareCandidatesByPriority);
  const activeVisible = [];
  const viewcellPrefetchVisible = [];
  for (const item of sortedVisible) {
    const id = Number(item.globalGlbId);
    if (!Number.isFinite(id)) continue;
    if (activeSet.has(id >>> 0)) activeVisible.push(item);
    else viewcellPrefetchVisible.push(item);
  }

  if (state.downloadPlanMode === 'raw-visible') {
    return {
      immediate: sortedVisible,
      prefetch: [],
      all: sortedVisible,
      activeVisibleCount: activeVisible.length,
      viewcellPrefetchVisibleCount: viewcellPrefetchVisible.length,
    };
  }

  if (modelPriority) {
    for (const item of activeVisible) {
      const id = Number(item.globalGlbId);
      if (!Number.isFinite(id) || seen.has(id)) continue;
      seen.add(id);
      if (immediate.length < maxImmediate) immediate.push(item);
      else if (prefetch.length < maxPrefetch) prefetch.push(item);
    }
    for (const item of viewcellPrefetchVisible) {
      const id = Number(item.globalGlbId);
      if (!Number.isFinite(id) || seen.has(id) || prefetch.length >= maxPrefetch) continue;
      seen.add(id);
      prefetch.push(item);
    }
    const sortedPrefetch = (prefetchCandidates || []).slice().sort(compareCandidatesByPriority);
    for (const item of sortedPrefetch) {
      const id = Number(item.globalGlbId);
      if (!Number.isFinite(id) || seen.has(id) || prefetch.length >= maxPrefetch) continue;
      seen.add(id);
      prefetch.push(item);
    }
    return {
      immediate,
      prefetch,
      all: sortedVisible,
      activeVisibleCount: activeVisible.length,
      viewcellPrefetchVisibleCount: viewcellPrefetchVisible.length,
    };
  }

  // Viewcell PVS is intentionally conservative. Download the current-camera
  // subset first; the rest of the viewcell union is prefetch, not immediate.
  for (const item of activeVisible) {
    const id = Number(item.globalGlbId);
    if (!Number.isFinite(id) || seen.has(id)) continue;
    seen.add(id);
    if (immediate.length < maxImmediate) immediate.push(item);
    else if (prefetch.length < maxPrefetch) prefetch.push(item);
  }
  for (const item of viewcellPrefetchVisible) {
    const id = Number(item.globalGlbId);
    if (!Number.isFinite(id) || seen.has(id) || prefetch.length >= maxPrefetch) continue;
    seen.add(id);
    prefetch.push(item);
  }
  const sortedPrefetch = (prefetchCandidates || []).slice().sort(compareCandidatesByPriority);
  for (const item of sortedPrefetch) {
    const id = Number(item.globalGlbId);
    if (!Number.isFinite(id) || seen.has(id) || prefetch.length >= maxPrefetch) continue;
    seen.add(id);
    prefetch.push(item);
  }
  return { immediate, prefetch, all: sortedVisible, activeVisibleCount: activeVisible.length, viewcellPrefetchVisibleCount: viewcellPrefetchVisible.length };
}

function idsFromCandidates(items) {
  return Uint32Array.from((items || []).map((item) => Number(item.globalGlbId) >>> 0));
}

function weightsFromCandidates(items) {
  return Float32Array.from((items || []).map((item) => Math.max(0.000001, candidatePriority(item, 1))));
}

function buildModelInfo(meta) {
  const workpoint = meta?.testWorkpoint || {};
  return {
    runtimeModelName: meta?.runtimeModelName || meta?.experimentName || meta?.modelName || '-',
    runtimeModelDisplayName: meta?.runtimeModelDisplayName || meta?.runtimeModelName || '-',
    runtimeSchema: meta?.runtimeSchema || '-',
    assetFile: meta?.assetFile || 'instance_pvs_assets.bin',
    thresholdSelection: meta?.thresholdSelection || meta?.selectionRule || '-',
    visibilityThreshold: Number(meta?.visibilityThreshold),
    prefetchThreshold: Number(meta?.prefetchThreshold),
    outputsDownloadPriority: meta?.outputsDownloadPriority === true || meta?.outputsGlbPriority === true,
    usesDynamicOcclusionPool: meta?.usesDynamicOcclusionPool === true,
    usesFixedInstanceFeatures: meta?.usesFixedInstanceFeatures !== false,
    testWorkpoint: {
      posePrecision: Number(workpoint.posePrecision),
      poseRecall: Number(workpoint.poseRecall),
      poseF1: Number(workpoint.poseF1),
      poseWeightedRecall: Number(workpoint.poseWeightedRecall),
      avgPred: Number(workpoint.avgPred),
      avgCandidate: Number(workpoint.avgCandidate),
      candidateReduction: Number(workpoint.candidateReduction),
      evalPoseCount: Number(workpoint.evalPoseCount),
    },
  };
}

function postResult(payload) {
  const transfers = [];
  for (const key of [
    'componentModelList',
    'modelList',
    'weightList',
    'immediateGlbIds',
    'immediateWeights',
    'prefetchGlbIds',
    'prefetchWeights',
    'renderComponentModelList',
    'renderModelList',
  ]) {
    if (payload[key] && payload[key].buffer) transfers.push(payload[key].buffer);
  }
  self.postMessage(payload, transfers);
}

async function initWorker(message) {
  const initStart = nowMs();
  state = {
    ...state,
    assetBaseUrl: String(message.assetBaseUrl || '').replace(/\/$/, ''),
    runtimeMetaUrl: String(message.runtimeMetaUrl || ''),
    // 主线程已加载的元数据 — 优先复用,避免重复 fetch 同一份 27MB+ 文件。
    preloadedModelMeta: message.preloadedModelMeta || null,
    preloadedRuntimeMeta: message.preloadedRuntimeMeta || null,
    forceFallback: Boolean(message.forceFallback),
    maxImmediate: Math.max(1, Number(message.maxImmediate || 384)),
    maxPrefetch: Math.max(0, Number(message.maxPrefetch || 2048)),
    prefetchThreshold: Number.isFinite(Number(message.prefetchThreshold))
      ? Math.max(0, Math.min(1, Number(message.prefetchThreshold)))
      : null,
    downloadPlanMode: normalizeDownloadPlanMode(message.downloadPlanMode),
    assetVersion: message.assetVersion || null,
    m12Pvs: null,
  };

  const metaStart = nowMs();
  // 优先用主线程 postMessage 传过来的 model_meta(已含二进制缓冲引用,零成本);
  // 缺省时才自己 fetch。
  state.meta = state.preloadedModelMeta
    || await fetchJson(versionedAssetUrl(`${state.assetBaseUrl}/instance_model_meta.json`));
  const idMaps = buildLocalToGlobal(state.meta);
  state.localToGlobalInstance = idMaps.localToGlobal;
  state.globalToLocalInstance = idMaps.globalToLocal;
  if (state.prefetchThreshold == null) {
    state.prefetchThreshold = Math.max(0, Math.min(1, Number(state.meta.prefetchThreshold ?? 0.04)));
  }
  // 同样:runtimeMeta 优先用主线程传的(避免 worker 再 fetch 27MB),
  // 缺省时才回退 URL。
  state.runtimeMeta = state.preloadedRuntimeMeta
    || (state.runtimeMetaUrl ? await fetchJson(state.runtimeMetaUrl) : { componentRecords: [] });
  const fallbackTables = buildFallbackTables(state.meta, state.runtimeMeta);
  state.componentAabbs = fallbackTables.aabbs;
  state.instanceToGlobalGlb = fallbackTables.instanceToGlb;
  const metaEnd = nowMs();

  if (!state.forceFallback) {
    state.backend = 'worker-aabb-fallback-warming';
    state.ready = true;
    state.initTimings = {
      totalMs: nowMs() - initStart,
      metaMs: metaEnd - metaStart,
      fallbackReason: 'webgpu-warming',
    };
    self.postMessage({ type: 'ready', backend: state.backend, timings: state.initTimings, fallbackReason: 'webgpu-warming', modelInfo: buildModelInfo(state.meta) });

    const warmupPvs = new InstancePVS(state.assetBaseUrl, {
        runtimeMetaUrl: state.runtimeMetaUrl,
        debugLogging: Boolean(message.debugLogging),
        assetVersion: message.assetVersion || null,
        inferenceFovYDeg: MODEL_INPUT_FOV_Y_DEG,
        deferRuntimeMeta: false,
    });
    state.pvs = warmupPvs;
    warmupPvs.init()
      .then(() => {
        if (state.pvs !== warmupPvs) return;
        state.backend = 'worker-webgpu';
        state.initTimings = {
          totalMs: nowMs() - initStart,
          metaMs: metaEnd - metaStart,
          pvs: warmupPvs.lastInitTimings || null,
        };
        self.postMessage({ type: 'upgraded', backend: state.backend, timings: state.initTimings, modelInfo: buildModelInfo(state.meta) });
      })
      .catch((error) => {
        if (state.pvs !== warmupPvs) return;
        state.pvs = null;
        state.backend = 'worker-aabb-fallback';
        state.initTimings = {
          totalMs: nowMs() - initStart,
          metaMs: metaEnd - metaStart,
          fallbackReason: error && error.message ? error.message : String(error),
        };
        self.postMessage({
          type: 'upgraded',
          backend: state.backend,
          timings: state.initTimings,
          fallbackReason: state.initTimings.fallbackReason,
          modelInfo: buildModelInfo(state.meta),
        });
      });
    return;
  }

  state.backend = 'worker-aabb-fallback';
  state.ready = true;
  state.initTimings = {
    totalMs: nowMs() - initStart,
    metaMs: metaEnd - metaStart,
    fallbackReason: 'forced-fallback',
  };
  self.postMessage({ type: 'ready', backend: state.backend, timings: state.initTimings, fallbackReason: 'forced-fallback', modelInfo: buildModelInfo(state.meta) });
}

async function predictWorker(message) {
  const start = nowMs();
  if (!state.ready) throw new Error('Lightweight PVS worker is not ready.');
  const snapshot = message.snapshot || {};
  const activeCamera = buildCamera(snapshot);
  const predictionCamera = buildPredictionCamera(snapshot, state.meta);
  let rawComponentIds = [];
  let visibleCandidates = [];
  let prefetchCandidates = [];
  let candidateCount = 0;
  let backend = state.backend;
  let fallbackReason = null;
  let inferenceMs = 0;
  let candidateMs = 0;
  let candidateSelection = null;
  let hasModelDownloadPriority = false;
  let pvsOutputUsesGlobalIds = false;

  if (state.pvs && state.backend === 'worker-webgpu') {
    const inferenceStart = nowMs();
    const pred = await state.pvs.predict(predictionCamera, {
      camera: predictionCamera,
      returnAllScores: true,
      prefetchThreshold: state.prefetchThreshold,
    }, predictionCamera);
    inferenceMs = nowMs() - inferenceStart;
    candidateMs = Number(pred?.timings?.candidateMs || 0);
    rawComponentIds = Array.isArray(pred?.componentModelList) ? pred.componentModelList : [];
    pvsOutputUsesGlobalIds = true;
    visibleCandidates = Array.isArray(pred?.candidates) ? pred.candidates : [];
    prefetchCandidates = Array.isArray(pred?.prefetchCandidates) ? pred.prefetchCandidates : [];
    candidateCount = Number(pred?.candidateCount || 0);
    candidateSelection = normalizeCandidateSelection(
      pred?.timings?.candidateSelection || pred?.candidateSelection,
      candidateCount,
    );
    hasModelDownloadPriority = Boolean(pred?.hasModelDownloadPriority);
  } else {
    fallbackReason = state.initTimings?.fallbackReason || 'webgpu-unavailable';
    backend = 'worker-aabb-fallback';
    const frustum = frustumForCamera(predictionCamera);
    const numInstances = Number(state.meta?.numInstances || 0);
    for (let id = 0; id < numInstances; id += 1) {
      if (intersectsComponent(id, frustum)) rawComponentIds.push(id >>> 0);
    }
    candidateCount = rawComponentIds.length;
    candidateSelection = normalizeCandidateSelection({
      source: 'worker_full_aabb_scan',
      candidateCount,
    }, candidateCount);
    visibleCandidates = candidateFromVisibleComponents(rawComponentIds);
    hasModelDownloadPriority = false;
  }

  const localRawComponentIds = pvsOutputUsesGlobalIds
    ? globalIdsToLocal(rawComponentIds)
    : rawComponentIds;
  const renderComponentIds = filterComponentsByFrustum(localRawComponentIds, activeCamera);
  const outputRawComponentIds = pvsOutputUsesGlobalIds
    ? rawComponentIds
    : rawComponentIds.map(localIdToGlobal);
  const outputRenderComponentIds = renderComponentIds.map(localIdToGlobal);
  const renderGlbIds = componentIdsToGlbIds(renderComponentIds);
  const modelList = idsFromCandidates(visibleCandidates);
  const weightList = weightsFromCandidates(visibleCandidates);
  const plan = splitDownloadPlan(visibleCandidates, prefetchCandidates, renderGlbIds, {
    modelPriority: hasModelDownloadPriority,
  });
  const end = nowMs();

  postResult({
    type: 'result',
    serial: message.serial,
    idMode: 'global-glb',
    backend,
    fallbackReason,
    componentModelList: Uint32Array.from(outputRawComponentIds),
    modelList,
    weightList,
    immediateGlbIds: idsFromCandidates(plan.immediate),
    immediateWeights: weightsFromCandidates(plan.immediate),
    prefetchGlbIds: idsFromCandidates(plan.prefetch),
    prefetchWeights: weightsFromCandidates(plan.prefetch),
    renderComponentModelList: Uint32Array.from(outputRenderComponentIds),
    renderModelList: Uint32Array.from(renderGlbIds),
    candidateCount,
    candidateSelection,
    visibleInstanceCount: rawComponentIds.length,
    hasModelDownloadPriority,
    executionTime: end - start,
    timings: {
      totalMs: end - start,
      inferenceMs,
      candidateMs,
      postMs: end - start - inferenceMs,
      candidateSelection,
      rawInstanceCount: outputRawComponentIds.length,
      rawGlbCount: modelList.length,
      immediateGlbCount: plan.immediate.length,
      prefetchGlbCount: plan.prefetch.length,
      downloadPlanMode: state.downloadPlanMode,
      activeVisibleGlbCount: plan.activeVisibleCount,
      viewcellPrefetchGlbCount: plan.viewcellPrefetchVisibleCount,
      renderInstanceCount: renderComponentIds.length,
      renderGlbCount: renderGlbIds.length,
      hasModelDownloadPriority,
      prioritySource: hasModelDownloadPriority ? 'model-download-priority' : 'visibility-fallback',
      modelInfo: buildModelInfo(state.meta),
    },
  });
}

async function m12ProbeWorker(message) {
  if (!state.ready) throw new Error('Lightweight PVS worker is not ready.');
  if (!state.m12Pvs) {
    const probe = new InstancePVS(state.assetBaseUrl, {
      runtimeMetaUrl: '',
      debugLogging: false,
      assetVersion: state.assetVersion,
      inferenceFovYDeg: MODEL_INPUT_FOV_Y_DEG,
      deferRuntimeMeta: true,
      benchmarkRawOutput: true,
    });
    try {
      await probe.init();
    } catch (error) {
      const detail = probe.lastWebGPUTimings?.error || probe.lastWebGPUTimings?.unavailableReason;
      if (detail && !String(error.message || '').includes(detail)) {
        error.message = `${error.message || error} (${detail})`;
      }
      throw error;
    }
    state.m12Pvs = probe;
  }
  const cases = Array.isArray(message.cases) ? message.cases : [];
  const results = [];
  for (const item of cases) {
    // M12 cases are generated offline with a `camera` field; accepting the
    // older `snapshot` spelling keeps captured probes backwards compatible.
    const snapshot = item && (item.snapshot || item.camera) ? (item.snapshot || item.camera) : {};
    const camera = buildPredictionCamera(snapshot, state.meta);
    const candidateIds = Array.isArray(item?.candidateIds)
      ? item.candidateIds.map((value) => Number(value) >>> 0)
      : [];
    const prediction = await state.m12Pvs.predict(camera, {
      camera,
      candidateIds,
      returnRawOutput: true,
    }, camera);
    results.push({
      caseId: item?.caseId == null ? results.length : item.caseId,
      backend: prediction?.backend || state.m12Pvs.backend,
      candidateIds: prediction?.rawCandidateIds || candidateIds,
      rawOutput: prediction?.rawOutput || [],
      timings: prediction?.timings || null,
    });
  }
  self.postMessage({
    type: 'm12-result',
    requestId: message.requestId,
    backend: state.m12Pvs.backend,
    results,
  });
}

self.onmessage = (event) => {
  const message = event.data || {};
  Promise.resolve()
    .then(async () => {
      if (message.type === 'init') {
        await initWorker(message);
      } else if (message.type === 'setDownloadPlanMode') {
        state.downloadPlanMode = normalizeDownloadPlanMode(message.downloadPlanMode);
      } else if (message.type === 'setRuntimeMeta') {
        // 主线程在 init 后注入 runtimeMeta(避免 worker 重复 fetch 27MB+ 同一份)
        state.runtimeMeta = message.runtimeMeta || state.runtimeMeta;
        const fallbackTables = buildFallbackTables(state.meta, state.runtimeMeta);
        state.componentAabbs = fallbackTables.aabbs;
        state.instanceToGlobalGlb = fallbackTables.instanceToGlb;
      } else if (message.type === 'setModelMeta') {
        // 主线程注入 model_meta(替换 placeholder,重建 fallback tables)
        if (message.modelMeta) {
          state.meta = message.modelMeta;
          const idMaps = buildLocalToGlobal(state.meta);
          state.localToGlobalInstance = idMaps.localToGlobal;
          state.globalToLocalInstance = idMaps.globalToLocal;
          if (state.prefetchThreshold == null) {
            state.prefetchThreshold = Math.max(0, Math.min(1, Number(state.meta.prefetchThreshold ?? 0.04)));
          }
          const fallbackTables = buildFallbackTables(state.meta, state.runtimeMeta);
          state.componentAabbs = fallbackTables.aabbs;
          state.instanceToGlobalGlb = fallbackTables.instanceToGlb;
        }
      } else if (message.type === 'predict') {
        await predictWorker(message);
      } else if (message.type === 'm12-probe') {
        await m12ProbeWorker(message);
      }
    })
    .catch((error) => {
      const payload = {
        type: message.type === 'm12-probe' ? 'm12-error' : 'error',
        serial: message.serial,
        requestId: message.requestId,
        message: error && error.message ? error.message : String(error),
        stack: error && error.stack ? error.stack : null,
      };
      self.postMessage(payload);
    });
};
