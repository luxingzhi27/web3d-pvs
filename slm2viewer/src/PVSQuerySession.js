import { PerspectiveCamera } from 'three/src/cameras/PerspectiveCamera.js';
import { Quaternion } from 'three/src/math/Quaternion.js';
import { Vector3 } from 'three/src/math/Vector3.js';
import { InstancePVSRuntime } from './InstancePVSRuntime.js';
import { FRONTEND_RENDER_FOV_Y_DEG, MODEL_INPUT_FOV_Y_DEG } from './neuralPvsFovProtocol.js';
import { bitsetFromIds, diffIdBitsets } from './sortedIdDelta.js';

const RUNTIME_SCHEMA = 'pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4';

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function finiteOr(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
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

function idsFromCandidates(items) {
  return Uint32Array.from((items || []).map((item) => Number(item.globalGlbId) >>> 0));
}

function weightsFromCandidates(items) {
  return Float32Array.from((items || []).map((item) => {
    const value = Number(item?.downloadPriority ?? item?.visibilityScore ?? item?.confidence ?? 0);
    return Math.max(0.000001, Number.isFinite(value) ? value : 0);
  }));
}

export class PVSQuerySession {
  constructor(assetBaseUrl, options = {}) {
    this.assetBaseUrl = String(assetBaseUrl || '').replace(/\/$/, '');
    this.assetVersion = options.assetVersion || null;
    this.maxPrefetch = Math.max(0, Number(options.maxPrefetch || 2048));
    this.prefetchThreshold = options.prefetchThreshold != null
      && Number.isFinite(Number(options.prefetchThreshold))
      ? Math.max(0, Math.min(1, Number(options.prefetchThreshold)))
      : 0.04;
    this.downloadPlanMode = options.downloadPlanMode === 'raw-visible'
      ? 'raw-visible'
      : 'viewcell-priority';
    this.debugLogging = Boolean(options.debugLogging);
    this.backendPreference = options.backendPreference || 'auto';
    this.externalWebGPUContext = options.externalWebGPUContext || null;
    this.executionLocation = options.executionLocation === 'renderer' ? 'renderer' : 'worker';
    this.meta = null;
    this.pvs = null;
    this.ready = false;
    this.backend = 'uninitialized';
    this.initTimings = null;
    this.renderComponentBitset = new Uint32Array();
    this.renderGlbBitset = new Uint32Array();
    this.renderRevision = 0;
    this._quaternion = new Quaternion();
    this._forward = new Vector3();
  }

  _versionedUrl(url) {
    if (!this.assetVersion) return url;
    return `${url}${url.includes('?') ? '&' : '?'}v=${encodeURIComponent(this.assetVersion)}`;
  }

  async _fetchJson(url) {
    const response = await fetch(url, { cache: 'force-cache' });
    if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
    const text = await response.text();
    if (text.trimStart().startsWith('<')) throw new Error(`Expected JSON from ${url}, received HTML.`);
    return JSON.parse(text);
  }

  _buildCandidateCamera(snapshot) {
    const backOffset = Number(this.meta?.query?.candidateCameraBackOffsetM);
    if (!Number.isFinite(backOffset) || backOffset < 0) {
      throw new Error('V4 metadata has no valid candidate-camera back offset.');
    }
    this._quaternion.set(
      finiteOr(snapshot.quaternion?.[0], 0),
      finiteOr(snapshot.quaternion?.[1], 0),
      finiteOr(snapshot.quaternion?.[2], 0),
      finiteOr(snapshot.quaternion?.[3], 1),
    );
    this._forward.set(0, 0, -1).applyQuaternion(this._quaternion).normalize();
    const position = [
      finiteOr(snapshot.position?.[0], 0) - this._forward.x * backOffset,
      finiteOr(snapshot.position?.[1], 0) - this._forward.y * backOffset,
      finiteOr(snapshot.position?.[2], 0) - this._forward.z * backOffset,
    ];
    return buildCamera(snapshot, MODEL_INPUT_FOV_Y_DEG, position);
  }

  _runtimeBackend() {
    const backend = this.pvs?.backend || 'uninitialized';
    return this.executionLocation === 'renderer' && backend.includes('webgpu')
      ? `renderer-shared-${backend}`
      : `worker-${backend}`;
  }

  getModelInfo() {
    const selected = this.meta?.calibration?.selected || {};
    return {
      runtimeModelName: this.meta?.experimentName || 'pvs_mainline_v4',
      runtimeModelDisplayName: 'PVS V4 分层生存场',
      runtimeSchema: this.meta?.schema || '-',
      assetFile: 'V4 six-file runtime bundle',
      thresholdSelection: this.meta?.calibration?.source || 'checkpoint calibration',
      visibilityThreshold: Number(this.meta?.threshold),
      prefetchThreshold: this.prefetchThreshold,
      outputsDownloadPriority: false,
      usesDynamicOcclusionPool: false,
      usesFixedInstanceFeatures: true,
      runtimeFeatureDim: Number(this.meta?.modelConfig?.runtimeFeatureDim),
      runtimeFeatureBytes: Number(this.meta?.files?.runtimeFeatures?.byteLength),
      numInstances: Number(this.meta?.numInstances),
      numGlbs: Number(this.meta?.numGlbs),
      viewcellRadiusM: Number(this.meta?.query?.viewcellRadiusM),
      candidateCameraBackOffsetM: Number(this.meta?.query?.candidateCameraBackOffsetM),
      calibrationWorkpoint: {
        aggregateWeightedRecall: Number(selected.aggregateWeightedRecall),
        aggregateWeightedRecallLowerConfidenceBound: Number(
          selected.aggregateWeightedRecallLowerConfidenceBound,
        ),
        poseWeightedRecall: Number(selected.poseMacroWeightedRecall),
        poseWeightedRecallLowerConfidenceBound: Number(
          selected.poseMacroWeightedRecallLowerConfidenceBound,
        ),
        testEvaluationCount: Number(this.meta?.calibration?.testEvaluationCount || 0),
      },
    };
  }

  async init() {
    if (this.ready) return this;
    const startedAt = nowMs();
    this.meta = await this._fetchJson(this._versionedUrl(`${this.assetBaseUrl}/model_meta.json`));
    if (this.meta?.schema !== RUNTIME_SCHEMA) {
      throw new Error(`PVS runtime only accepts the current V4 schema, got ${this.meta?.schema || 'missing'}.`);
    }
    this.pvs = new InstancePVSRuntime(this.assetBaseUrl, {
      assetVersion: this.assetVersion,
      preloadedMeta: this.meta,
      debugLogging: this.debugLogging,
      backendPreference: this.backendPreference,
      externalWebGPUContext: this.externalWebGPUContext,
    });
    await this.pvs.init();
    this.backend = this._runtimeBackend();
    this.ready = true;
    this.initTimings = {
      totalMs: nowMs() - startedAt,
      pvs: this.pvs.lastInitTimings,
    };
    return this;
  }

  setDownloadPlanMode(value) {
    this.downloadPlanMode = value === 'raw-visible' ? 'raw-visible' : 'viewcell-priority';
    return this.downloadPlanMode;
  }

  async predict(snapshot, serial = 0) {
    if (!this.ready) throw new Error('PVS query session is not ready.');
    const startedAt = nowMs();
    const activeCamera = buildCamera(snapshot, FRONTEND_RENDER_FOV_Y_DEG);
    const candidateCamera = this._buildCandidateCamera(snapshot);
    const inferenceStartedAt = nowMs();
    const prediction = await this.pvs.predict(activeCamera, {
      candidateCamera,
      renderCamera: activeCamera,
      prefetchThreshold: this.prefetchThreshold,
    });
    this.backend = this._runtimeBackend();
    const inferenceMs = nowMs() - inferenceStartedAt;
    const componentIds = prediction.componentModelList || new Uint32Array();
    const renderComponentIds = prediction.renderComponentModelList || new Uint32Array();
    const visibleCandidates = prediction.candidates || [];
    const prefetchCandidates = prediction.prefetchCandidates || [];
    const renderGlbIds = prediction.renderModelList || [];
    const renderGlbSet = new Set(Array.from(renderGlbIds, Number));
    const warmCandidates = visibleCandidates.filter(
      (item) => !renderGlbSet.has(Number(item.globalGlbId)),
    );
    const limitedPrefetchCandidates = this.downloadPlanMode === 'raw-visible'
      ? []
      : prefetchCandidates.slice(0, this.maxPrefetch);
    this.renderComponentBitset = bitsetFromIds(renderComponentIds, Number(this.meta.numInstances));
    this.renderGlbBitset = bitsetFromIds(renderGlbIds, Number(this.meta.numGlbs));
    this.renderRevision += 1;
    const finishedAt = nowMs();
    return {
      type: 'result',
      serial,
      idMode: 'global-glb-priority',
      backend: this.backend,
      fallbackReason: this.pvs.fallbackReason,
      componentModelList: componentIds,
      modelList: idsFromCandidates(visibleCandidates),
      weightList: weightsFromCandidates(visibleCandidates),
      prefetchGlbIds: idsFromCandidates(limitedPrefetchCandidates),
      prefetchWeights: weightsFromCandidates(limitedPrefetchCandidates),
      renderComponentModelList: renderComponentIds,
      renderModelList: Uint32Array.from(renderGlbIds),
      renderRevision: this.renderRevision,
      candidateInstanceIds: this.debugLogging ? prediction.rawCandidateIds || null : null,
      candidateScores: this.debugLogging ? prediction.scores || null : null,
      candidateDiagnostics: this.debugLogging ? prediction.diagnosticRows || null : null,
      diagnosticOutputFloats: this.debugLogging
        ? Number(prediction.diagnosticOutputFloats || 0)
        : 0,
      candidateCount: Number(prediction.candidateCount || 0),
      candidateSelection: prediction.timings?.candidateSelection || null,
      visibleInstanceCount: componentIds.length,
      hasModelDownloadPriority: false,
      executionTime: finishedAt - startedAt,
      timings: {
        totalMs: finishedAt - startedAt,
        inferenceMs,
        candidateMs: 0,
        postMs: finishedAt - startedAt - inferenceMs,
        candidateSelection: prediction.timings?.candidateSelection || null,
        rawInstanceCount: componentIds.length,
        rawGlbCount: visibleCandidates.length,
        urgentGlbCount: renderGlbIds.length,
        warmGlbCount: warmCandidates.length,
        prefetchGlbCount: limitedPrefetchCandidates.length,
        renderInstanceCount: renderComponentIds.length,
        renderGlbCount: renderGlbIds.length,
        gpuFusedCandidateAndFilter: Boolean(prediction.timings?.gpuFusedCandidateAndFilter),
        workerWasmNeuralInference: Boolean(prediction.timings?.workerWasmNeuralInference),
        readbackBytes: Number(prediction.timings?.readbackBytes || 0),
        downloadPlanMode: this.downloadPlanMode,
        hasModelDownloadPriority: false,
        prioritySource: 'visibility-probability',
        modelInfo: this.getModelInfo(),
      },
    };
  }

  async refilter(snapshot, serial = 0) {
    if (!this.ready) throw new Error('PVS query session is not ready.');
    const startedAt = nowMs();
    const activeCamera = buildCamera(snapshot, FRONTEND_RENDER_FOV_Y_DEG);
    const filtered = await this.pvs.refilter(activeCamera);
    if (!filtered) throw new Error('No cached model prediction is available for render refiltering.');
    this.backend = this._runtimeBackend();
    const renderComponentBitset = filtered.renderComponentBitset || new Uint32Array();
    const renderGlbBitset = filtered.renderGlbBitset || new Uint32Array();
    const componentDelta = diffIdBitsets(
      this.renderComponentBitset,
      renderComponentBitset,
      Number(this.meta.numInstances),
    );
    const glbDelta = diffIdBitsets(
      this.renderGlbBitset,
      renderGlbBitset,
      Number(this.meta.numGlbs),
    );
    this.renderComponentBitset = renderComponentBitset;
    this.renderGlbBitset = renderGlbBitset;
    this.renderRevision += 1;
    const finishedAt = nowMs();
    return {
      type: 'filter-result',
      serial,
      idMode: 'global-glb-priority',
      backend: `${this.backend}-cached-render-filter`,
      fallbackReason: this.pvs.fallbackReason,
      renderComponentAddedIds: componentDelta.added,
      renderComponentRemovedIds: componentDelta.removed,
      renderGlbAddedIds: glbDelta.added,
      renderGlbRemovedIds: glbDelta.removed,
      renderInstanceCount: Number(filtered.renderInstanceCount || 0),
      renderGlbCount: Number(filtered.renderGlbCount || 0),
      renderRevision: this.renderRevision,
      executionTime: finishedAt - startedAt,
      timings: {
        ...(filtered.timings || {}),
        totalMs: finishedAt - startedAt,
        transferredIdCount: componentDelta.added.length + componentDelta.removed.length
          + glbDelta.added.length + glbDelta.removed.length,
        modelInfo: this.getModelInfo(),
      },
    };
  }

  dispose() {
    this.pvs?.dispose();
    this.pvs = null;
    this.ready = false;
    this.backend = 'disposed';
  }
}
