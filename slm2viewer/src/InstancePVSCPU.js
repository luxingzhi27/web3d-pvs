import { InstancePVS } from './InstancePVS.js';
import {
  intersectsAabbFrustum,
  linearInto,
  lowRankInto,
  makeRayQuery,
  sigmoid,
  spectralInto,
  survivalInto,
  unpackFloat16,
} from './InstancePVSCPUMath.js';

const RUNTIME_FEATURE_DIM = 124;
const GEOMETRY_DIM = 96;
const SURVIVAL_COEFFICIENT_DIM = 28;
const DIAGNOSTIC_OUTPUT_FLOATS = 1 + 9 + 18 + 64;
const GLB_FLAG_MODEL_VISIBLE = 1;
const GLB_FLAG_RENDER_VISIBLE = 2;

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function makeScratch() {
  return {
    query: {
      center: new Float32Array(9),
      axes: new Float32Array(18),
      distance: 0,
      radius: 0,
    },
    spectral: new Float32Array(64),
    lowRank: new Float32Array(4),
    boundaryInput: new Float32Array(85),
    boundaryHidden: new Float32Array(48),
    boundary: new Float32Array(8),
    directionInput: new Float32Array(19),
    directionHidden: new Float32Array(24),
    basis: new Float32Array(4),
    semantic: new Float32Array(8),
    trunkInput: new Float32Array(130),
    trunkHidden0: new Float32Array(64),
    trunkHidden1: new Float32Array(64),
    visibility: new Float32Array(1),
  };
}

function setBit(bitset, id) {
  bitset[id >>> 5] |= 1 << (id & 31);
}

export class InstancePVSCPU extends InstancePVS {
  constructor(assetBaseUrl, options = {}) {
    super(assetBaseUrl, options);
    this.runtimeFeatures = null;
    this.queryWeights = null;
    this.frequencies = null;
    this.chi = null;
    this.relationFeatures = null;
    this.cachedModelVisibleIds = new Uint32Array();
    this.scratch = makeScratch();
  }

  async _initInternal() {
    const debugLogging = this.debugLogging;
    this.debugLogging = false;
    try {
      await super._initInternal();
    } finally {
      this.debugLogging = debugLogging;
    }
    this.backend = 'cpu-js-v4';
    this.lastInitTimings = {
      ...this.lastInitTimings,
      cpuInitMs: Number(this.lastInitTimings?.gpuInitMs || 0),
      gpuInitMs: 0,
      webgpu: null,
    };
    if (this.debugLogging) console.log('[InstancePVSCPU] V4 CPU runtime ready.', this.lastInitTimings);
    return this;
  }

  async _initializeBackend(runtimeWords, weightWords, frequencies, chi) {
    const startedAt = nowMs();
    const numInstances = Number(this.meta.numInstances);
    const weightElements = this.meta.networkWeights.layout.reduce(
      (maximum, item) => Math.max(maximum, Number(item.offsetElements) + Number(item.elementCount)),
      0,
    );
    this.runtimeFeatures = unpackFloat16(runtimeWords, numInstances * RUNTIME_FEATURE_DIM);
    this.queryWeights = unpackFloat16(weightWords, weightElements);
    this.frequencies = frequencies;
    this.chi = chi;
    this.relationFeatures = new Float32Array(numInstances * 8);
    const hidden = new Float32Array(16);
    const relation = new Float32Array(8);
    for (let instanceId = 0; instanceId < numInstances; instanceId += 1) {
      const coefficientOffset = instanceId * RUNTIME_FEATURE_DIM + GEOMETRY_DIM;
      const coefficients = this.runtimeFeatures.subarray(
        coefficientOffset,
        coefficientOffset + SURVIVAL_COEFFICIENT_DIM,
      );
      this._linear('relation_condition_head.0', coefficients, hidden, 'silu');
      this._linear('relation_condition_head.2', hidden, relation);
      this.relationFeatures.set(relation, instanceId * 8);
    }
    this.cpuInfo = {
      decodeAndPrecomputeMs: nowMs() - startedAt,
      execution: 'dedicated-worker-javascript',
      relationPrecomputed: true,
    };
  }

  _linear(layerName, input, output, activation = 'none') {
    return linearInto(
      input,
      output,
      this.queryWeights,
      this._weightOffset(`${layerName}.weight`),
      this._weightOffset(`${layerName}.bias`),
      activation,
    );
  }

  _scoreInstance(instanceId, cameraQuery, diagnosticRows, diagnosticOffset) {
    const scratch = this.scratch;
    const runtimeOffset = instanceId * RUNTIME_FEATURE_DIM;
    const coefficients = this.runtimeFeatures.subarray(
      runtimeOffset + GEOMETRY_DIM,
      runtimeOffset + RUNTIME_FEATURE_DIM,
    );
    const relation = this.relationFeatures.subarray(instanceId * 8, instanceId * 8 + 8);
    const query = makeRayQuery(
      this.instanceAabbs,
      instanceId,
      cameraQuery,
      Number(this.meta.query.viewcellRadiusM),
      scratch.query,
    );
    spectralInto(query, this.frequencies, this.chi, scratch.spectral);
    lowRankInto(query, scratch.lowRank);

    scratch.boundaryInput.set(scratch.spectral, 0);
    scratch.boundaryInput.set(query.center, 64);
    scratch.boundaryInput.set(scratch.lowRank, 73);
    scratch.boundaryInput.set(relation, 77);
    this._linear('boundary_summary_head.0', scratch.boundaryInput, scratch.boundaryHidden, 'silu');
    this._linear('boundary_summary_head.2', scratch.boundaryHidden, scratch.boundary, 'tanh');

    scratch.directionInput.set(scratch.boundary, 0);
    scratch.directionInput.set(relation, 8);
    scratch.directionInput.set(query.center.subarray(0, 3), 16);
    this._linear('direction_basis_head.0', scratch.directionInput, scratch.directionHidden, 'silu');
    this._linear('direction_basis_head.2', scratch.directionHidden, scratch.basis, 'tanh');

    const depth = this.meta.depth;
    const rawDepth = Math.log1p(query.distance / (query.radius + Number(depth.epsilon)));
    const normalizedDepth = clamp(
      (rawDepth - Number(depth.q01)) / (Number(depth.q99) - Number(depth.q01)),
      0,
      1,
    );
    survivalInto(scratch.basis, coefficients, normalizedDepth, scratch.semantic);

    scratch.trunkInput.set(this.runtimeFeatures.subarray(runtimeOffset, runtimeOffset + GEOMETRY_DIM), 0);
    scratch.trunkInput.set(scratch.basis, 96);
    scratch.trunkInput.set(scratch.semantic, 100);
    scratch.trunkInput.set(scratch.boundary, 108);
    scratch.trunkInput.set(query.center, 116);
    scratch.trunkInput.set(scratch.lowRank, 125);
    scratch.trunkInput[129] = normalizedDepth;
    this._linear('shared_trunk.0', scratch.trunkInput, scratch.trunkHidden0, 'relu');
    this._linear('shared_trunk.2', scratch.trunkHidden0, scratch.trunkHidden1, 'relu');
    this._linear('visibility_head', scratch.trunkHidden1, scratch.visibility);
    const probability = sigmoid(scratch.visibility[0]);

    if (diagnosticRows) {
      diagnosticRows[diagnosticOffset] = probability;
      diagnosticRows.set(query.center, diagnosticOffset + 1);
      diagnosticRows.set(query.axes, diagnosticOffset + 10);
      diagnosticRows.set(scratch.spectral, diagnosticOffset + 28);
    }
    return probability;
  }

  async _predictUnlocked(queryCamera, options = {}) {
    if (!this.isReady) return null;
    const startedAt = nowMs();
    const cameraQuery = this._cameraQuery(queryCamera);
    const candidatePlanes = this._frustumPlanes(options.candidateCamera, 'backed 66-degree candidate');
    const renderPlanes = this._frustumPlanes(
      options.renderCamera || queryCamera,
      'current 60-degree render',
    );
    const threshold = Number(this.meta.threshold);
    const prefetchThreshold = clamp(Number(options.prefetchThreshold ?? 0.04), 0, 1);
    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    const glbScores = new Float32Array(numGlbs);
    const glbFlags = new Uint8Array(numGlbs);
    const glbTouched = new Uint8Array(numGlbs);
    const visibleIds = [];
    const renderIds = [];
    const rawCandidateIds = this.debugLogging ? new Uint32Array(numInstances) : null;
    const scores = this.debugLogging ? new Float32Array(numInstances) : null;
    const diagnosticRows = this.debugLogging
      ? new Float32Array(numInstances * DIAGNOSTIC_OUTPUT_FLOATS)
      : null;
    let candidateCount = 0;

    const inferenceStartedAt = nowMs();
    for (let instanceId = 0; instanceId < numInstances; instanceId += 1) {
      if (!intersectsAabbFrustum(this.instanceAabbs, instanceId, candidatePlanes)) continue;
      const diagnosticOffset = candidateCount * DIAGNOSTIC_OUTPUT_FLOATS;
      const probability = this._scoreInstance(instanceId, cameraQuery, diagnosticRows, diagnosticOffset);
      if (rawCandidateIds) rawCandidateIds[candidateCount] = instanceId;
      if (scores) scores[candidateCount] = probability;
      candidateCount += 1;
      const glbId = this.instanceToGlobalGlbArray[instanceId];
      glbTouched[glbId] = 1;
      if (probability > glbScores[glbId]) glbScores[glbId] = probability;
      if (probability < threshold) continue;
      visibleIds.push(instanceId);
      glbFlags[glbId] |= GLB_FLAG_MODEL_VISIBLE;
      if (intersectsAabbFrustum(this.instanceAabbs, instanceId, renderPlanes)) {
        renderIds.push(instanceId);
        glbFlags[glbId] |= GLB_FLAG_RENDER_VISIBLE;
      }
    }
    const inferenceFinishedAt = nowMs();

    const glbQueue = [];
    for (let glbId = 0; glbId < numGlbs; glbId += 1) {
      if (!glbTouched[glbId] || glbScores[glbId] < prefetchThreshold) continue;
      const score = glbScores[glbId];
      const flags = glbFlags[glbId];
      glbQueue.push({
        globalGlbId: glbId,
        confidence: score,
        importance: score,
        downloadPriority: score,
        visibilityScore: score,
        modelVisible: (flags & GLB_FLAG_MODEL_VISIBLE) !== 0,
        renderVisible: (flags & GLB_FLAG_RENDER_VISIBLE) !== 0,
        prioritySource: 'cpu-max-visibility-probability',
      });
    }
    glbQueue.sort((a, b) => b.downloadPriority - a.downloadPriority || a.globalGlbId - b.globalGlbId);
    const candidates = glbQueue.filter((item) => item.modelVisible);
    const prefetchCandidates = glbQueue.filter((item) => !item.modelVisible);
    const renderModelList = glbQueue
      .filter((item) => item.renderVisible)
      .map((item) => item.globalGlbId)
      .sort((a, b) => a - b);
    const componentModelList = Uint32Array.from(visibleIds);
    const renderComponentModelList = Uint32Array.from(renderIds);
    this.cachedModelVisibleIds = componentModelList.slice();
    this.hasCachedPrediction = true;
    this.lastCandidateSelection = {
      source: 'cpu_v4_backed_frustum_candidates',
      candidateCount,
      indexedInstanceCount: numInstances,
      overflowInstanceCount: 0,
    };
    const finishedAt = nowMs();
    this.predictSerial += 1;
    this.lastPredictTimings = {
      serial: this.predictSerial,
      totalMs: finishedAt - startedAt,
      candidateMs: 0,
      inferenceMs: inferenceFinishedAt - inferenceStartedAt,
      postMs: finishedAt - inferenceFinishedAt,
      candidateCount,
      visibleInstanceCount: componentModelList.length,
      renderInstanceCount: renderComponentModelList.length,
      visibleGlbCount: candidates.length,
      renderGlbCount: renderModelList.length,
      glbQueueCount: glbQueue.length,
      candidateSelection: this.lastCandidateSelection,
      hasModelDownloadPriority: false,
      gpuFusedCandidateAndFilter: false,
      workerCpuNeuralInference: true,
      readbackBytes: 0,
      backend: this.backend,
    };
    return {
      idMode: 'global-glb-priority',
      componentModelList,
      renderComponentModelList,
      modelList: candidates.map((item) => item.globalGlbId),
      weightList: candidates.map((item) => item.downloadPriority),
      renderModelList,
      candidates,
      prefetchCandidates,
      glbQueue,
      candidateCount,
      visibleInstanceCount: componentModelList.length,
      renderVisibleInstanceCount: renderComponentModelList.length,
      hasModelDownloadPriority: false,
      ...(this.debugLogging ? {
        scores: scores.slice(0, candidateCount),
        rawCandidateIds: rawCandidateIds.slice(0, candidateCount),
        diagnosticOutputFloats: DIAGNOSTIC_OUTPUT_FLOATS,
        diagnosticRows: diagnosticRows.slice(0, candidateCount * DIAGNOSTIC_OUTPUT_FLOATS),
      } : {}),
      backend: this.backend,
      executionTime: finishedAt - startedAt,
      timings: { ...this.lastPredictTimings },
    };
  }

  async _refilterUnlocked(renderCamera) {
    if (!this.isReady || !this.hasCachedPrediction) return null;
    const startedAt = nowMs();
    const planes = this._frustumPlanes(renderCamera, 'current 60-degree render');
    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    const renderComponentBitset = new Uint32Array(Math.ceil(numInstances / 32));
    const renderGlbBitset = new Uint32Array(Math.ceil(numGlbs / 32));
    let renderInstanceCount = 0;
    let renderGlbCount = 0;
    for (const instanceId of this.cachedModelVisibleIds) {
      if (!intersectsAabbFrustum(this.instanceAabbs, instanceId, planes)) continue;
      setBit(renderComponentBitset, instanceId);
      renderInstanceCount += 1;
      const glbId = this.instanceToGlobalGlbArray[instanceId];
      const word = glbId >>> 5;
      const mask = 1 << (glbId & 31);
      if ((renderGlbBitset[word] & mask) === 0) {
        renderGlbBitset[word] |= mask;
        renderGlbCount += 1;
      }
    }
    const finishedAt = nowMs();
    return {
      idMode: 'global-glb-priority',
      renderComponentBitset,
      renderGlbBitset,
      renderInstanceCount,
      renderGlbCount,
      backend: `${this.backend}-cached-render-filter`,
      executionTime: finishedAt - startedAt,
      timings: {
        totalMs: finishedAt - startedAt,
        inferenceMs: 0,
        filterMs: finishedAt - startedAt,
        renderInstanceCount,
        renderGlbCount,
        cachedModelPrediction: true,
        workerCpuNeuralInference: true,
        readbackBytes: 0,
      },
    };
  }

  dispose() {
    this.runtimeFeatures = null;
    this.queryWeights = null;
    this.frequencies = null;
    this.chi = null;
    this.relationFeatures = null;
    this.cachedModelVisibleIds = new Uint32Array();
    super.dispose();
  }
}
