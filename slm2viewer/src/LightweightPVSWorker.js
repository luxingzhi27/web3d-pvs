import { Box3, Frustum, Matrix4, PerspectiveCamera, Quaternion, Vector3 } from 'three';
import { InstancePVS } from './InstancePVS.js';
import { FRONTEND_RENDER_FOV_Y_DEG, MODEL_INPUT_FOV_Y_DEG } from './neuralPvsFovProtocol.js';

const RUNTIME_SCHEMA = 'pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4';
const _box = new Box3();
const _matrix = new Matrix4();
const _frustum = new Frustum();
const _quaternion = new Quaternion();
const _forward = new Vector3();

let state = {
  assetBaseUrl: '',
  assetVersion: null,
  meta: null,
  pvs: null,
  componentAabbs: null,
  instanceToGlobalGlb: null,
  ready: false,
  backend: 'uninitialized',
  initTimings: null,
  maxImmediate: 384,
  maxPrefetch: 2048,
  prefetchThreshold: 0.04,
  downloadPlanMode: 'viewcell-priority',
  diagnostics: false,
};

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function finiteOr(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function versionedUrl(url) {
  if (!state.assetVersion) return url;
  return `${url}${url.includes('?') ? '&' : '?'}v=${encodeURIComponent(state.assetVersion)}`;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const text = await response.text();
  if (text.trimStart().startsWith('<')) throw new Error(`Expected JSON from ${url}, received HTML.`);
  return JSON.parse(text);
}

function buildCamera(snapshot, fov, position = null) {
  const aspect = Math.max(1e-4, finiteOr(snapshot.aspect, 16 / 9));
  const near = Math.max(1e-4, finiteOr(snapshot.near, 0.1));
  const far = Math.max(near + 1, finiteOr(snapshot.far, 20000));
  const camera = new PerspectiveCamera(fov, aspect, near, far);
  const sourcePosition = position || snapshot.position || [0, 0, 0];
  camera.position.set(
    finiteOr(sourcePosition[0], 0),
    finiteOr(sourcePosition[1], 0),
    finiteOr(sourcePosition[2], 0),
  );
  camera.quaternion.set(
    finiteOr(snapshot.quaternion?.[0], 0),
    finiteOr(snapshot.quaternion?.[1], 0),
    finiteOr(snapshot.quaternion?.[2], 0),
    finiteOr(snapshot.quaternion?.[3], 1),
  );
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return camera;
}

function buildCandidateCamera(snapshot) {
  const backOffset = Number(state.meta?.query?.candidateCameraBackOffsetM);
  if (!Number.isFinite(backOffset) || backOffset < 0) {
    throw new Error('V4 metadata has no valid candidate-camera back offset.');
  }
  _quaternion.set(
    finiteOr(snapshot.quaternion?.[0], 0),
    finiteOr(snapshot.quaternion?.[1], 0),
    finiteOr(snapshot.quaternion?.[2], 0),
    finiteOr(snapshot.quaternion?.[3], 1),
  );
  _forward.set(0, 0, -1).applyQuaternion(_quaternion).normalize();
  const position = [
    finiteOr(snapshot.position?.[0], 0) - _forward.x * backOffset,
    finiteOr(snapshot.position?.[1], 0) - _forward.y * backOffset,
    finiteOr(snapshot.position?.[2], 0) - _forward.z * backOffset,
  ];
  return buildCamera(snapshot, MODEL_INPUT_FOV_Y_DEG, position);
}

function frustumForCamera(camera) {
  camera.updateMatrixWorld(true);
  _matrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  return _frustum.setFromProjectionMatrix(_matrix);
}

function intersectsComponent(componentId, frustum) {
  const aabbs = state.componentAabbs;
  const offset = componentId * 6;
  if (!aabbs || offset + 5 >= aabbs.length) return false;
  _box.min.set(aabbs[offset], aabbs[offset + 1], aabbs[offset + 2]);
  _box.max.set(aabbs[offset + 3], aabbs[offset + 4], aabbs[offset + 5]);
  return frustum.intersectsBox(_box);
}

function filterComponentsByFrustum(componentIds, camera) {
  const frustum = frustumForCamera(camera);
  const result = [];
  for (const value of componentIds || []) {
    const id = Number(value);
    if (Number.isInteger(id) && id >= 0 && intersectsComponent(id, frustum)) result.push(id >>> 0);
  }
  return result;
}

function componentIdsToGlbIds(componentIds) {
  const glbs = new Set();
  for (const id of componentIds || []) {
    if (id < state.instanceToGlobalGlb.length) glbs.add(state.instanceToGlobalGlb[id] >>> 0);
  }
  return Array.from(glbs).sort((a, b) => a - b);
}

function priority(item) {
  const value = Number(item?.downloadPriority ?? item?.visibilityScore ?? item?.confidence ?? 0);
  return Number.isFinite(value) ? value : 0;
}

function sortByPriority(a, b) {
  return priority(b) - priority(a) || Number(a.globalGlbId) - Number(b.globalGlbId);
}

function candidatesFromComponents(componentIds, score = 1, source = 'aabb-fallback') {
  const byGlb = new Map();
  for (const id of componentIds || []) {
    const glbId = state.instanceToGlobalGlb[id];
    let item = byGlb.get(glbId);
    if (!item) {
      item = {
        globalGlbId: glbId >>> 0,
        confidence: score,
        importance: score,
        downloadPriority: score,
        visibilityScore: score,
        prioritySource: source,
        sourceComponentIds: [],
        sourceComponents: [],
      };
      byGlb.set(glbId, item);
    }
    item.sourceComponentIds.push(id >>> 0);
    item.sourceComponents.push({
      componentId: id >>> 0,
      visibilityScore: score,
      downloadPriority: score,
      visible: true,
    });
  }
  return Array.from(byGlb.values()).sort(sortByPriority);
}

function splitDownloadPlan(visibleCandidates, prefetchCandidates, activeGlbIds) {
  const immediate = [];
  const prefetch = [];
  const seen = new Set();
  const active = new Set(activeGlbIds || []);
  const sortedVisible = (visibleCandidates || []).slice().sort(sortByPriority);
  if (state.downloadPlanMode === 'raw-visible') {
    return {
      immediate: sortedVisible,
      prefetch: [],
      activeVisibleCount: sortedVisible.filter((item) => active.has(item.globalGlbId)).length,
      viewcellPrefetchVisibleCount: sortedVisible.filter((item) => !active.has(item.globalGlbId)).length,
    };
  }
  const activeVisible = sortedVisible.filter((item) => active.has(item.globalGlbId));
  const viewcellVisible = sortedVisible.filter((item) => !active.has(item.globalGlbId));
  for (const item of activeVisible) {
    if (seen.has(item.globalGlbId)) continue;
    seen.add(item.globalGlbId);
    if (immediate.length < state.maxImmediate) immediate.push(item);
    else if (prefetch.length < state.maxPrefetch) prefetch.push(item);
  }
  for (const item of [...viewcellVisible, ...(prefetchCandidates || []).slice().sort(sortByPriority)]) {
    if (seen.has(item.globalGlbId) || prefetch.length >= state.maxPrefetch) continue;
    seen.add(item.globalGlbId);
    prefetch.push(item);
  }
  return {
    immediate,
    prefetch,
    activeVisibleCount: activeVisible.length,
    viewcellPrefetchVisibleCount: viewcellVisible.length,
  };
}

function idsFromCandidates(items) {
  return Uint32Array.from((items || []).map((item) => Number(item.globalGlbId) >>> 0));
}

function weightsFromCandidates(items) {
  return Float32Array.from((items || []).map((item) => Math.max(0.000001, priority(item))));
}

function modelInfo() {
  const selected = state.meta?.calibration?.selected || {};
  return {
    runtimeModelName: state.meta?.experimentName || 'pvs_mainline_v4',
    runtimeModelDisplayName: 'PVS V4 分层生存场',
    runtimeSchema: state.meta?.schema || '-',
    assetFile: 'V4 six-file runtime bundle',
    thresholdSelection: state.meta?.calibration?.source || 'checkpoint calibration',
    visibilityThreshold: Number(state.meta?.threshold),
    prefetchThreshold: state.prefetchThreshold,
    outputsDownloadPriority: false,
    usesDynamicOcclusionPool: false,
    usesFixedInstanceFeatures: true,
    runtimeFeatureDim: Number(state.meta?.modelConfig?.runtimeFeatureDim),
    runtimeFeatureBytes: Number(state.meta?.files?.runtimeFeatures?.byteLength),
    viewcellRadiusM: Number(state.meta?.query?.viewcellRadiusM),
    candidateCameraBackOffsetM: Number(state.meta?.query?.candidateCameraBackOffsetM),
    calibrationWorkpoint: {
      aggregateWeightedRecall: Number(selected.aggregateWeightedRecall),
      aggregateWeightedRecallLowerConfidenceBound: Number(
        selected.aggregateWeightedRecallLowerConfidenceBound,
      ),
      poseWeightedRecall: Number(selected.poseMacroWeightedRecall),
      poseWeightedRecallLowerConfidenceBound: Number(
        selected.poseMacroWeightedRecallLowerConfidenceBound,
      ),
      testEvaluationCount: Number(state.meta?.calibration?.testEvaluationCount || 0),
    },
  };
}

function postResult(payload) {
  const transfers = [];
  for (const key of [
    'componentModelList', 'modelList', 'weightList', 'immediateGlbIds', 'immediateWeights',
    'prefetchGlbIds', 'prefetchWeights', 'renderComponentModelList', 'renderModelList',
    'candidateInstanceIds', 'candidateScores', 'candidateDiagnostics',
  ]) {
    if (payload[key]?.buffer) transfers.push(payload[key].buffer);
  }
  self.postMessage(payload, transfers);
}

async function initWorker(message) {
  const startedAt = nowMs();
  state = {
    ...state,
    assetBaseUrl: String(message.assetBaseUrl || '').replace(/\/$/, ''),
    assetVersion: message.assetVersion || null,
    maxImmediate: Math.max(1, Number(message.maxImmediate || 384)),
    maxPrefetch: Math.max(0, Number(message.maxPrefetch || 2048)),
    prefetchThreshold: message.prefetchThreshold != null
      && Number.isFinite(Number(message.prefetchThreshold))
      ? Math.max(0, Math.min(1, Number(message.prefetchThreshold)))
      : 0.04,
    downloadPlanMode: message.downloadPlanMode === 'raw-visible' ? 'raw-visible' : 'viewcell-priority',
    diagnostics: Boolean(message.debugLogging),
  };
  state.meta = await fetchJson(versionedUrl(`${state.assetBaseUrl}/model_meta.json`));
  if (state.meta?.schema !== RUNTIME_SCHEMA) {
    throw new Error(`The worker only accepts the current V4 runtime, got ${state.meta?.schema || 'missing'}.`);
  }
  const pvs = new InstancePVS(state.assetBaseUrl, {
    assetVersion: state.assetVersion,
    preloadedMeta: state.meta,
    debugLogging: Boolean(message.debugLogging),
  });
  state.pvs = pvs;
  try {
    await pvs.init();
    state.backend = 'worker-webgpu-v4';
  } catch (error) {
    if (!pvs.instanceAabbs || !pvs.instanceToGlobalGlbArray) throw error;
    state.backend = 'worker-aabb-fallback';
    state.fallbackReason = error?.message || String(error);
  }
  state.componentAabbs = pvs.instanceAabbs;
  state.instanceToGlobalGlb = pvs.instanceToGlobalGlbArray;
  state.ready = true;
  state.initTimings = {
    totalMs: nowMs() - startedAt,
    pvs: pvs.lastInitTimings,
    fallbackReason: state.fallbackReason || null,
  };
  self.postMessage({
    type: 'ready',
    backend: state.backend,
    timings: state.initTimings,
    fallbackReason: state.fallbackReason || null,
    modelInfo: modelInfo(),
  });
}

async function predictWorker(message) {
  if (!state.ready) throw new Error('PVS worker is not ready.');
  const startedAt = nowMs();
  const snapshot = message.snapshot || {};
  const activeCamera = buildCamera(snapshot, FRONTEND_RENDER_FOV_Y_DEG);
  const candidateCamera = buildCandidateCamera(snapshot);
  let componentIds;
  let visibleCandidates;
  let prefetchCandidates;
  let candidateCount;
  let candidateSelection;
  let candidateInstanceIds = null;
  let candidateScores = null;
  let candidateDiagnostics = null;
  let diagnosticOutputFloats = 0;
  let inferenceMs = 0;

  if (state.backend === 'worker-webgpu-v4') {
    const inferenceStartedAt = nowMs();
    const prediction = await state.pvs.predict(activeCamera, {
      candidateCamera,
      returnAllScores: true,
      returnScores: state.diagnostics,
      returnCandidateIds: state.diagnostics,
      prefetchThreshold: state.prefetchThreshold,
    });
    inferenceMs = nowMs() - inferenceStartedAt;
    componentIds = prediction.componentModelList || [];
    visibleCandidates = prediction.candidates || [];
    prefetchCandidates = prediction.prefetchCandidates || [];
    candidateCount = Number(prediction.candidateCount || 0);
    candidateSelection = prediction.timings?.candidateSelection || null;
    if (state.diagnostics) {
      candidateInstanceIds = Uint32Array.from(prediction.rawCandidateIds || []);
      candidateScores = Float32Array.from(prediction.scores || []);
      candidateDiagnostics = prediction.diagnosticRows || null;
      diagnosticOutputFloats = Number(prediction.diagnosticOutputFloats || 0);
    }
  } else {
    const frustum = frustumForCamera(candidateCamera);
    componentIds = [];
    for (let id = 0; id < state.meta.numInstances; id += 1) {
      if (intersectsComponent(id, frustum)) componentIds.push(id >>> 0);
    }
    visibleCandidates = candidatesFromComponents(componentIds);
    prefetchCandidates = [];
    candidateCount = componentIds.length;
    candidateSelection = { source: 'worker_v4_aabb_fallback', candidateCount };
  }

  const renderComponentIds = filterComponentsByFrustum(componentIds, activeCamera);
  const renderGlbIds = componentIdsToGlbIds(renderComponentIds);
  const plan = splitDownloadPlan(visibleCandidates, prefetchCandidates, renderGlbIds);
  const modelList = idsFromCandidates(visibleCandidates);
  const weightList = weightsFromCandidates(visibleCandidates);
  const finishedAt = nowMs();
  postResult({
    type: 'result',
    serial: message.serial,
    idMode: 'global-glb-priority',
    backend: state.backend,
    fallbackReason: state.fallbackReason || null,
    componentModelList: Uint32Array.from(componentIds),
    modelList,
    weightList,
    immediateGlbIds: idsFromCandidates(plan.immediate),
    immediateWeights: weightsFromCandidates(plan.immediate),
    prefetchGlbIds: idsFromCandidates(plan.prefetch),
    prefetchWeights: weightsFromCandidates(plan.prefetch),
    renderComponentModelList: Uint32Array.from(renderComponentIds),
    renderModelList: Uint32Array.from(renderGlbIds),
    candidateInstanceIds,
    candidateScores,
    candidateDiagnostics,
    diagnosticOutputFloats,
    candidateCount,
    candidateSelection,
    visibleInstanceCount: componentIds.length,
    hasModelDownloadPriority: false,
    executionTime: finishedAt - startedAt,
    timings: {
      totalMs: finishedAt - startedAt,
      inferenceMs,
      candidateMs: Number(state.pvs?.lastPredictTimings?.candidateMs || 0),
      postMs: finishedAt - startedAt - inferenceMs,
      candidateSelection,
      rawInstanceCount: componentIds.length,
      rawGlbCount: modelList.length,
      immediateGlbCount: plan.immediate.length,
      prefetchGlbCount: plan.prefetch.length,
      activeVisibleGlbCount: plan.activeVisibleCount,
      viewcellPrefetchGlbCount: plan.viewcellPrefetchVisibleCount,
      renderInstanceCount: renderComponentIds.length,
      renderGlbCount: renderGlbIds.length,
      downloadPlanMode: state.downloadPlanMode,
      hasModelDownloadPriority: false,
      prioritySource: 'visibility-probability',
      modelInfo: modelInfo(),
    },
  });
}

self.onmessage = (event) => {
  const message = event.data || {};
  Promise.resolve().then(async () => {
    if (message.type === 'init') {
      await initWorker(message);
    } else if (message.type === 'setDownloadPlanMode') {
      state.downloadPlanMode = message.downloadPlanMode === 'raw-visible' ? 'raw-visible' : 'viewcell-priority';
    } else if (message.type === 'predict') {
      await predictWorker(message);
    }
  }).catch((error) => {
    self.postMessage({
      type: 'error',
      serial: message.serial,
      message: error?.message || String(error),
      stack: error?.stack || null,
    });
  });
};
