import { InstancePVSBase, nowMs } from './InstancePVSBase.js';

const RUNTIME_FEATURE_DIM = 124;
const DEFAULT_WASM_URL = './assets/wasm/instance_pvs_v4.wasm';
const GLB_FLAG_MODEL_VISIBLE = 1;
const GLB_FLAG_RENDER_VISIBLE = 2;

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function versionedUrl(url, version) {
  if (!version) return url;
  return `${url}${url.includes('?') ? '&' : '?'}v=${encodeURIComponent(version)}`;
}

function requiredExport(exports, name) {
  const value = exports?.[name];
  if (typeof value !== 'function') throw new Error(`V4 WASM module is missing export ${name}.`);
  return value;
}

export class InstancePVSWasm extends InstancePVSBase {
  constructor(assetBaseUrl, options = {}) {
    super(assetBaseUrl, options);
    this.wasmUrl = options.wasmUrl || DEFAULT_WASM_URL;
    this.wasmDiagnostics = Boolean(options.debugLogging);
    this.wasmInstance = null;
    this.wasm = null;
    this.wasmMemory = null;
    this.allocations = [];
    this.pointers = null;
    this.cachedVisibleCount = 0;
  }

  async _initInternal() {
    const debugLogging = this.debugLogging;
    this.debugLogging = false;
    try {
      await super._initInternal();
    } finally {
      this.debugLogging = debugLogging;
    }
    this.backend = 'wasm-simd-v4';
    this.lastInitTimings = {
      ...this.lastInitTimings,
      wasmInitMs: Number(this.lastInitTimings?.backendInitMs || 0),
      wasm: this.wasmInfo,
    };
    if (this.debugLogging) console.log('[InstancePVSWasm] V4 WASM SIMD runtime ready.', this.lastInitTimings);
    return this;
  }

  async _initializeBackend(runtimeWords, weightWords, frequencies, chi) {
    const startedAt = nowMs();
    const response = await fetch(versionedUrl(this.wasmUrl, this.assetVersion), { cache: 'force-cache' });
    if (!response.ok) throw new Error(`Failed to load ${this.wasmUrl}: ${response.status}`);
    const moduleBytes = await response.arrayBuffer();
    let instantiated;
    try {
      instantiated = await WebAssembly.instantiate(moduleBytes, {});
    } catch (error) {
      throw new Error(`The browser cannot initialize the required WASM SIMD backend: ${error.message}`);
    }
    this.wasmInstance = instantiated.instance;
    this.wasm = this.wasmInstance.exports;
    this.wasmMemory = this.wasm.memory;
    if (!(this.wasmMemory instanceof WebAssembly.Memory)) {
      throw new Error('V4 WASM module does not export linear memory.');
    }
    for (const name of [
      'allocate', 'deallocate', 'decode_fp16', 'configure_runtime',
      'precompute_relations', 'predict_v4', 'refilter_v4', 'wasm_simd_enabled',
    ]) requiredExport(this.wasm, name);
    if (this.wasm.wasm_simd_enabled() !== 1) {
      throw new Error('V4 compatibility backend was not compiled with WASM SIMD enabled.');
    }

    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    const weightElements = this.meta.networkWeights.layout.reduce(
      (maximum, item) => Math.max(maximum, Number(item.offsetElements) + Number(item.elementCount)),
      0,
    );
    const pointers = {};
    pointers.runtimeFeatures = this._allocate(numInstances * RUNTIME_FEATURE_DIM * 4);
    pointers.weights = this._allocate(weightElements * 4);
    const runtimeSource = this._copyBytes(
      new Uint8Array(runtimeWords.buffer, runtimeWords.byteOffset, numInstances * RUNTIME_FEATURE_DIM * 2),
    );
    const weightSource = this._copyBytes(
      new Uint8Array(weightWords.buffer, weightWords.byteOffset, weightElements * 2),
    );
    if (this.wasm.decode_fp16(
      runtimeSource.pointer,
      numInstances * RUNTIME_FEATURE_DIM,
      pointers.runtimeFeatures,
    ) !== 0 || this.wasm.decode_fp16(weightSource.pointer, weightElements, pointers.weights) !== 0) {
      throw new Error('V4 WASM failed to decode the FP16 model tables.');
    }
    this._deallocate(runtimeSource);
    this._deallocate(weightSource);

    pointers.aabbs = this._copyBytes(new Uint8Array(
      this.instanceAabbs.buffer,
      this.instanceAabbs.byteOffset,
      this.instanceAabbs.byteLength,
    )).pointer;
    pointers.instanceToGlb = this._copyBytes(new Uint8Array(
      this.instanceToGlobalGlbArray.buffer,
      this.instanceToGlobalGlbArray.byteOffset,
      this.instanceToGlobalGlbArray.byteLength,
    )).pointer;
    pointers.frequencies = this._copyBytes(new Uint8Array(
      frequencies.buffer,
      frequencies.byteOffset,
      frequencies.byteLength,
    )).pointer;
    pointers.chi = this._copyBytes(new Uint8Array(chi.buffer, chi.byteOffset, chi.byteLength)).pointer;
    pointers.relations = this._allocate(numInstances * 8 * 4);

    if (this.wasm.configure_runtime(
      numInstances,
      numGlbs,
      pointers.runtimeFeatures,
      pointers.aabbs,
      pointers.instanceToGlb,
      pointers.weights,
      pointers.frequencies,
      pointers.chi,
      pointers.relations,
    ) !== 0 || this.wasm.precompute_relations() !== 0) {
      throw new Error('V4 WASM failed to configure or precompute the relation features.');
    }

    pointers.query = this._allocate(14 * 4);
    pointers.candidatePlanes = this._allocate(24 * 4);
    pointers.renderPlanes = this._allocate(24 * 4);
    pointers.counters = this._allocate(4 * 4);
    pointers.visibleIds = this._allocate(numInstances * 4);
    pointers.renderIds = this._allocate(numInstances * 4);
    pointers.glbScores = this._allocate(numGlbs * 4);
    pointers.glbFlags = this._allocate(numGlbs * 4);
    pointers.queueIds = this._allocate(numGlbs * 4);
    pointers.queueScores = this._allocate(numGlbs * 4);
    pointers.queueFlags = this._allocate(numGlbs * 4);
    pointers.debugCandidateIds = this.wasmDiagnostics ? this._allocate(numInstances * 4) : 0;
    pointers.debugScores = this.wasmDiagnostics ? this._allocate(numInstances * 4) : 0;
    pointers.instanceBitWords = Math.ceil(numInstances / 32);
    pointers.glbBitWords = Math.ceil(numGlbs / 32);
    pointers.instanceBits = this._allocate(pointers.instanceBitWords * 4);
    pointers.glbBits = this._allocate(pointers.glbBitWords * 4);
    pointers.filterCounters = this._allocate(2 * 4);
    this.pointers = pointers;
    this.instanceAabbs = null;
    this.instanceToGlobalGlbArray = null;
    this.wasmInfo = {
      moduleBytes: moduleBytes.byteLength,
      decodeAndPrecomputeMs: nowMs() - startedAt,
      execution: 'dedicated-worker-wasm-simd',
      relationPrecomputed: true,
      sharedMemoryThreads: false,
    };
  }

  _allocate(byteLength) {
    const length = Math.max(1, Number(byteLength));
    const pointer = Number(this.wasm.allocate(length));
    if (!pointer) throw new Error(`V4 WASM allocation failed for ${length} bytes.`);
    this.allocations.push({ pointer, byteLength: length });
    return pointer;
  }

  _deallocate(allocation) {
    if (!allocation?.pointer) return;
    this.wasm.deallocate(allocation.pointer, allocation.byteLength);
    const index = this.allocations.findIndex((item) => item.pointer === allocation.pointer);
    if (index >= 0) this.allocations.splice(index, 1);
  }

  _copyBytes(source) {
    const allocation = { pointer: this._allocate(source.byteLength), byteLength: source.byteLength };
    new Uint8Array(this.wasmMemory.buffer, allocation.pointer, source.byteLength).set(source);
    return allocation;
  }

  _writeFloat32(pointer, values) {
    new Float32Array(this.wasmMemory.buffer, pointer, values.length).set(values);
  }

  _copyUint32(pointer, count) {
    return new Uint32Array(this.wasmMemory.buffer, pointer, count).slice();
  }

  _copyFloat32(pointer, count) {
    return new Float32Array(this.wasmMemory.buffer, pointer, count).slice();
  }

  async _predictUnlocked(queryCamera, options = {}) {
    if (!this.isReady) return null;
    const startedAt = nowMs();
    const query = this._cameraQuery(queryCamera);
    const candidatePlanes = this._frustumPlanes(options.candidateCamera, 'backed 66-degree candidate');
    const renderPlanes = this._frustumPlanes(options.renderCamera || queryCamera, 'current 60-degree render');
    const depth = this.meta.depth;
    const prefetchThreshold = clamp(Number(options.prefetchThreshold ?? 0.04), 0, 1);
    this._writeFloat32(this.pointers.query, Float32Array.of(
      ...query.center,
      ...query.forward,
      query.tanX,
      query.tanY,
      Number(this.meta.query.viewcellRadiusM),
      Number(depth.q01),
      Number(depth.q99),
      Number(depth.epsilon),
      Number(this.meta.threshold),
      prefetchThreshold,
    ));
    this._writeFloat32(this.pointers.candidatePlanes, candidatePlanes);
    this._writeFloat32(this.pointers.renderPlanes, renderPlanes);

    const inferenceStartedAt = nowMs();
    const status = this.wasm.predict_v4(
      this.pointers.query,
      this.pointers.candidatePlanes,
      this.pointers.renderPlanes,
      this.pointers.counters,
      this.pointers.visibleIds,
      this.pointers.renderIds,
      this.pointers.glbScores,
      this.pointers.glbFlags,
      this.pointers.queueIds,
      this.pointers.queueScores,
      this.pointers.queueFlags,
      this.pointers.debugCandidateIds,
      this.pointers.debugScores,
    );
    const inferenceFinishedAt = nowMs();
    if (status !== 0) throw new Error(`V4 WASM prediction failed with status ${status}.`);

    const counters = this._copyUint32(this.pointers.counters, 4);
    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    const candidateCount = Math.min(counters[0], numInstances);
    const visibleCount = Math.min(counters[1], numInstances);
    const renderCount = Math.min(counters[2], numInstances);
    const glbCount = Math.min(counters[3], numGlbs);
    const componentModelList = this._copyUint32(this.pointers.visibleIds, visibleCount);
    const renderComponentModelList = this._copyUint32(this.pointers.renderIds, renderCount);
    const queueIds = this._copyUint32(this.pointers.queueIds, glbCount);
    const queueScores = this._copyFloat32(this.pointers.queueScores, glbCount);
    const queueFlags = this._copyUint32(this.pointers.queueFlags, glbCount);
    const glbQueue = [];
    for (let index = 0; index < glbCount; index += 1) {
      const score = clamp(queueScores[index], 0, 1);
      const flags = queueFlags[index];
      glbQueue.push({
        globalGlbId: queueIds[index],
        confidence: score,
        importance: score,
        downloadPriority: score,
        visibilityScore: score,
        modelVisible: (flags & GLB_FLAG_MODEL_VISIBLE) !== 0,
        renderVisible: (flags & GLB_FLAG_RENDER_VISIBLE) !== 0,
        prioritySource: 'wasm-max-visibility-probability',
      });
    }
    glbQueue.sort((a, b) => b.downloadPriority - a.downloadPriority || a.globalGlbId - b.globalGlbId);
    const candidates = glbQueue.filter((item) => item.modelVisible);
    const prefetchCandidates = glbQueue.filter((item) => !item.modelVisible);
    const renderModelList = glbQueue
      .filter((item) => item.renderVisible)
      .map((item) => item.globalGlbId)
      .sort((a, b) => a - b);
    let rawCandidateIds;
    let scores;
    if (this.debugLogging) {
      rawCandidateIds = this._copyUint32(this.pointers.debugCandidateIds, candidateCount);
      scores = this._copyFloat32(this.pointers.debugScores, candidateCount);
    }
    this.cachedVisibleCount = visibleCount;
    this.hasCachedPrediction = true;
    this.lastCandidateSelection = {
      source: 'wasm_v4_back_frustum_aabb',
      candidateCount,
      indexedInstanceCount: numInstances,
      overflowInstanceCount: 0,
    };
    const finishedAt = nowMs();
    const readbackBytes = (visibleCount + renderCount + glbCount * 3
      + (this.debugLogging ? candidateCount * 2 : 0)) * 4;
    this.predictSerial += 1;
    this.lastPredictTimings = {
      serial: this.predictSerial,
      totalMs: finishedAt - startedAt,
      candidateMs: 0,
      inferenceMs: inferenceFinishedAt - inferenceStartedAt,
      postMs: finishedAt - inferenceFinishedAt,
      candidateCount,
      visibleInstanceCount: visibleCount,
      renderInstanceCount: renderCount,
      visibleGlbCount: candidates.length,
      renderGlbCount: renderModelList.length,
      glbQueueCount: glbQueue.length,
      candidateSelection: this.lastCandidateSelection,
      hasModelDownloadPriority: false,
      wasmFusedCandidateAndFilter: true,
      workerWasmNeuralInference: true,
      readbackBytes,
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
      visibleInstanceCount: visibleCount,
      renderVisibleInstanceCount: renderCount,
      hasModelDownloadPriority: false,
      ...(this.debugLogging ? { scores, rawCandidateIds } : {}),
      backend: this.backend,
      executionTime: finishedAt - startedAt,
      timings: { ...this.lastPredictTimings },
    };
  }

  async _refilterUnlocked(renderCamera) {
    if (!this.isReady || !this.hasCachedPrediction) return null;
    const startedAt = nowMs();
    this._writeFloat32(
      this.pointers.renderPlanes,
      this._frustumPlanes(renderCamera, 'current 60-degree render'),
    );
    const status = this.wasm.refilter_v4(
      this.pointers.renderPlanes,
      this.pointers.visibleIds,
      this.cachedVisibleCount,
      this.pointers.instanceBits,
      this.pointers.instanceBitWords,
      this.pointers.glbBits,
      this.pointers.glbBitWords,
      this.pointers.filterCounters,
    );
    if (status !== 0) throw new Error(`V4 WASM refilter failed with status ${status}.`);
    const counters = this._copyUint32(this.pointers.filterCounters, 2);
    const renderComponentBitset = this._copyUint32(
      this.pointers.instanceBits,
      this.pointers.instanceBitWords,
    );
    const renderGlbBitset = this._copyUint32(this.pointers.glbBits, this.pointers.glbBitWords);
    const finishedAt = nowMs();
    return {
      idMode: 'global-glb-priority',
      renderComponentBitset,
      renderGlbBitset,
      renderInstanceCount: counters[0],
      renderGlbCount: counters[1],
      backend: `${this.backend}-cached-render-filter`,
      executionTime: finishedAt - startedAt,
      timings: {
        totalMs: finishedAt - startedAt,
        inferenceMs: 0,
        filterMs: finishedAt - startedAt,
        renderInstanceCount: counters[0],
        renderGlbCount: counters[1],
        cachedModelPrediction: true,
        workerWasmNeuralInference: true,
        readbackBytes: (this.pointers.instanceBitWords + this.pointers.glbBitWords + 2) * 4,
      },
    };
  }

  dispose() {
    if (this.wasm) {
      for (let index = this.allocations.length - 1; index >= 0; index -= 1) {
        const allocation = this.allocations[index];
        this.wasm.deallocate(allocation.pointer, allocation.byteLength);
      }
    }
    this.allocations = [];
    this.pointers = null;
    this.cachedVisibleCount = 0;
    this.wasmInstance = null;
    this.wasm = null;
    this.wasmMemory = null;
    super.dispose();
  }
}
