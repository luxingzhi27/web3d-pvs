// V4 WebGPU compute backend: GPU buffers, dispatch, compact readback and WGSL.
import {
  DIAGNOSTIC_OUTPUT_FLOATS,
  InstancePVSBase,
  nowMs,
} from './InstancePVSBase.js';
import {
  buildGlbCompactionShader,
  buildPredictionShader,
  buildRenderFilterShader,
} from './InstancePVSWebGPUShaders.js';
import { countBitsetIds } from './sortedIdDelta.js';

const WORKGROUP_SIZE = 64;
const RESULT_COUNTER_WORDS = 4;
const GLB_FLAG_MODEL_VISIBLE = 1;
const GLB_FLAG_RENDER_VISIBLE = 2;

const WEBGPU_SINGLETON_KEY = '__SLM_INSTANCE_PVS_WEBGPU_SINGLETON__';

function getSharedWebGPUState() {
  const root = typeof globalThis !== 'undefined' ? globalThis : {};
  if (!root[WEBGPU_SINGLETON_KEY]) {
    root[WEBGPU_SINGLETON_KEY] = {
      adapter: null,
      adapterPromise: null,
      device: null,
      devicePromise: null,
    };
  }
  return root[WEBGPU_SINGLETON_KEY];
}

async function getSharedWebGPUAdapter() {
  const state = getSharedWebGPUState();
  if (state.adapter) return state.adapter;
  if (!state.adapterPromise) {
    state.adapterPromise = navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      .then((adapter) => {
        state.adapter = adapter || null;
        if (!adapter) state.adapterPromise = null;
        return adapter;
      })
      .catch((error) => {
        state.adapterPromise = null;
        throw error;
      });
  }
  return state.adapterPromise;
}

async function getSharedWebGPUDevice(adapter) {
  const state = getSharedWebGPUState();
  if (state.device) return state.device;
  if (!state.devicePromise) {
    state.devicePromise = adapter.requestDevice().then((device) => {
      state.device = device;
      device.lost.then(() => {
        if (state.device === device) {
          state.device = null;
          state.devicePromise = null;
        }
      });
      return device;
    }).catch((error) => {
      state.devicePromise = null;
      throw error;
    });
  }
  return state.devicePromise;
}

export class InstancePVSWebGPU extends InstancePVSBase {
  constructor(assetBaseUrl, options = {}) {
    super(assetBaseUrl, options);
    this.adapter = null;
    this.device = null;
    this.pipeline = null;
    this.uniformBuffer = null;
    this.runtimeFeatureBuffer = null;
    this.aabbBuffer = null;
    this.weightBuffer = null;
    this.frequencyBuffer = null;
    this.chiBuffer = null;
    this.instanceToGlbBuffer = null;
    this.resultBuffer = null;
    this.readbackBuffer = null;
    this.bindGroup = null;
    this.glbPipeline = null;
    this.glbBindGroup = null;
    this.filterPipeline = null;
    this.filterResultBuffer = null;
    this.filterReadbackBuffer = null;
    this.filterBindGroup = null;
    this.filterResultLayout = null;
    this.hasCachedPrediction = false;
    this.resultLayout = null;
    this.readbackLayout = null;
  }

  async _initInternal() {
    await super._initInternal();
    this.backend = 'webgpu-v4';
    this.lastInitTimings = {
      ...this.lastInitTimings,
      gpuInitMs: this.lastInitTimings.backendInitMs,
      webgpu: this.webgpuInfo,
    };
    if (this.debugLogging) console.log('[InstancePVSWebGPU] V4 runtime ready.', this.lastInitTimings);
    return this;
  }

  _createStorageBuffer(data) {
    const buffer = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  async _initializeBackend(runtimeWords, weightWords, frequencies, chi) {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('V4 instance visibility requires WebGPU.');
    }
    const shared = getSharedWebGPUState();
    const adapterCached = Boolean(shared.adapter);
    const adapterStartedAt = nowMs();
    this.adapter = await getSharedWebGPUAdapter();
    if (!this.adapter) throw new Error('WebGPU requestAdapter returned null.');
    const deviceStartedAt = nowMs();
    const deviceCached = Boolean(shared.device);
    this.device = await getSharedWebGPUDevice(this.adapter);

    this.uniformBuffer = this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.runtimeFeatureBuffer = this._createStorageBuffer(runtimeWords);
    this.aabbBuffer = this._createStorageBuffer(this.instanceAabbs);
    this.weightBuffer = this._createStorageBuffer(weightWords);
    this.frequencyBuffer = this._createStorageBuffer(frequencies);
    this.chiBuffer = this._createStorageBuffer(chi);
    this.instanceToGlbBuffer = this._createStorageBuffer(this.instanceToGlobalGlbArray);
    this.resultLayout = this._makeResultLayout();
    this.readbackLayout = this._makeReadbackLayout();

    const pipelineStartedAt = nowMs();
    const shaderModule = this.device.createShaderModule({ code: buildPredictionShader.call(this) });
    if (typeof shaderModule.getCompilationInfo === 'function') {
      const info = await shaderModule.getCompilationInfo();
      const errors = (info.messages || [])
        .filter((message) => message.type === 'error')
        .map((message) => `${message.lineNum || 0}:${message.linePos || 0} ${message.message}`);
      if (errors.length) throw new Error(`V4 WGSL compilation failed: ${errors.join('; ')}`);
    }
    const descriptor = { layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' } };
    this.pipeline = this.device.createComputePipelineAsync
      ? await this.device.createComputePipelineAsync(descriptor)
      : this.device.createComputePipeline(descriptor);
    const glbShaderModule = this.device.createShaderModule({ code: buildGlbCompactionShader.call(this) });
    if (typeof glbShaderModule.getCompilationInfo === 'function') {
      const info = await glbShaderModule.getCompilationInfo();
      const errors = (info.messages || [])
        .filter((message) => message.type === 'error')
        .map((message) => `${message.lineNum || 0}:${message.linePos || 0} ${message.message}`);
      if (errors.length) throw new Error(`V4 GLB compaction WGSL compilation failed: ${errors.join('; ')}`);
    }
    const glbDescriptor = { layout: 'auto', compute: { module: glbShaderModule, entryPoint: 'main' } };
    this.glbPipeline = this.device.createComputePipelineAsync
      ? await this.device.createComputePipelineAsync(glbDescriptor)
      : this.device.createComputePipeline(glbDescriptor);
    const filterShaderModule = this.device.createShaderModule({ code: buildRenderFilterShader.call(this) });
    if (typeof filterShaderModule.getCompilationInfo === 'function') {
      const info = await filterShaderModule.getCompilationInfo();
      const errors = (info.messages || [])
        .filter((message) => message.type === 'error')
        .map((message) => `${message.lineNum || 0}:${message.linePos || 0} ${message.message}`);
      if (errors.length) throw new Error(`V4 render refilter WGSL compilation failed: ${errors.join('; ')}`);
    }
    const filterDescriptor = { layout: 'auto', compute: { module: filterShaderModule, entryPoint: 'main' } };
    this.filterPipeline = this.device.createComputePipelineAsync
      ? await this.device.createComputePipelineAsync(filterDescriptor)
      : this.device.createComputePipeline(filterDescriptor);
    this._createRuntimeBuffers();
    this.instanceAabbs = null;
    this.instanceToGlobalGlbArray = null;
    const adapterInfo = this.adapter.info || {};
    this.webgpuInfo = {
      requestAdapterMs: deviceStartedAt - adapterStartedAt,
      pipelineMs: nowMs() - pipelineStartedAt,
      adapterCached,
      deviceCached,
      adapter: {
        vendor: adapterInfo.vendor || '',
        architecture: adapterInfo.architecture || '',
        device: adapterInfo.device || '',
        description: adapterInfo.description || '',
      },
      packedFp16Storage: true,
    };
  }

  _makeResultLayout() {
    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    let offset = RESULT_COUNTER_WORDS;
    const layout = { counters: 0 };
    layout.modelVisibleIds = offset;
    offset += numInstances;
    layout.renderVisibleIds = offset;
    offset += numInstances;
    layout.glbScores = offset;
    offset += numGlbs;
    layout.glbFlags = offset;
    offset += numGlbs;
    layout.glbQueueIds = offset;
    offset += numGlbs;
    layout.glbQueueScores = offset;
    offset += numGlbs;
    layout.glbQueueFlags = offset;
    offset += numGlbs;
    if (this.debugLogging) {
      layout.debugCandidateIds = offset;
      offset += numInstances;
      layout.debugRows = offset;
      offset += numInstances * DIAGNOSTIC_OUTPUT_FLOATS;
    }
    layout.totalWords = offset;
    return layout;
  }

  _makeReadbackLayout() {
    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    let offset = RESULT_COUNTER_WORDS;
    const layout = { counters: 0 };
    layout.modelVisibleIds = offset;
    offset += numInstances;
    layout.renderVisibleIds = offset;
    offset += numInstances;
    layout.glbQueueIds = offset;
    offset += numGlbs;
    layout.glbQueueScores = offset;
    offset += numGlbs;
    layout.glbQueueFlags = offset;
    offset += numGlbs;
    if (this.debugLogging) {
      layout.debugCandidateIds = offset;
      offset += numInstances;
      layout.debugRows = offset;
      offset += numInstances * DIAGNOSTIC_OUTPUT_FLOATS;
    }
    layout.totalWords = offset;
    return layout;
  }

  _makeFilterResultLayout() {
    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    let offset = 2;
    const layout = { counters: 0 };
    layout.instanceBitWords = Math.ceil(numInstances / 32);
    layout.glbBitWords = Math.ceil(numGlbs / 32);
    layout.renderVisibleBits = offset;
    offset += layout.instanceBitWords;
    layout.glbVisibleBits = offset;
    offset += layout.glbBitWords;
    layout.totalWords = offset;
    return layout;
  }

  _createRuntimeBuffers() {
    const resultByteLength = this.resultLayout.totalWords * 4;
    const readbackByteLength = this.readbackLayout.totalWords * 4;
    this.resultBuffer = this.device.createBuffer({
      size: resultByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.readbackBuffer = this.device.createBuffer({
      size: readbackByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.filterResultLayout = this._makeFilterResultLayout();
    this.filterResultBuffer = this.device.createBuffer({
      size: this.filterResultLayout.totalWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.filterReadbackBuffer = this.device.createBuffer({
      size: this.filterResultLayout.totalWords * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.runtimeFeatureBuffer } },
        { binding: 2, resource: { buffer: this.aabbBuffer } },
        { binding: 3, resource: { buffer: this.weightBuffer } },
        { binding: 4, resource: { buffer: this.frequencyBuffer } },
        { binding: 5, resource: { buffer: this.chiBuffer } },
        { binding: 6, resource: { buffer: this.instanceToGlbBuffer } },
        { binding: 7, resource: { buffer: this.resultBuffer } },
      ],
    });
    this.glbBindGroup = this.device.createBindGroup({
      layout: this.glbPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.resultBuffer } },
      ],
    });
    this.filterBindGroup = this.device.createBindGroup({
      layout: this.filterPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.aabbBuffer } },
        { binding: 2, resource: { buffer: this.instanceToGlbBuffer } },
        { binding: 3, resource: { buffer: this.resultBuffer } },
        { binding: 4, resource: { buffer: this.filterResultBuffer } },
      ],
    });
  }

  async _predictWebGPU(query, candidateCamera, renderCamera, prefetchThreshold) {
    const depth = this.meta.depth;
    const uniform = new Float32Array(64);
    uniform.set([
      query.center[0], query.center[1], query.center[2], Number(this.meta.threshold),
      query.forward[0], query.forward[1], query.forward[2], query.tanX,
      query.tanY, Number(this.meta.query.viewcellRadiusM), Number(depth.q01), Number(depth.q99),
      Number(depth.epsilon), Number(this.meta.numInstances), prefetchThreshold, Number(this.meta.numGlbs),
    ], 0);
    uniform.set(this._frustumPlanes(candidateCamera, 'backed 66-degree candidate'), 16);
    uniform.set(this._frustumPlanes(renderCamera, 'current 60-degree render'), 40);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniform);

    const outputBytes = this.readbackLayout.totalWords * 4;
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(this.resultBuffer);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(Number(this.meta.numInstances) / WORKGROUP_SIZE));
    pass.end();
    const glbPass = encoder.beginComputePass();
    glbPass.setPipeline(this.glbPipeline);
    glbPass.setBindGroup(0, this.glbBindGroup);
    glbPass.dispatchWorkgroups(Math.ceil(Number(this.meta.numGlbs) / WORKGROUP_SIZE));
    glbPass.end();
    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    encoder.copyBufferToBuffer(
      this.resultBuffer,
      0,
      this.readbackBuffer,
      0,
      (RESULT_COUNTER_WORDS + 2 * numInstances) * 4,
    );
    encoder.copyBufferToBuffer(
      this.resultBuffer,
      this.resultLayout.glbQueueIds * 4,
      this.readbackBuffer,
      this.readbackLayout.glbQueueIds * 4,
      3 * numGlbs * 4,
    );
    if (this.debugLogging) {
      encoder.copyBufferToBuffer(
        this.resultBuffer,
        this.resultLayout.debugCandidateIds * 4,
        this.readbackBuffer,
        this.readbackLayout.debugCandidateIds * 4,
        (numInstances + numInstances * DIAGNOSTIC_OUTPUT_FLOATS) * 4,
      );
    }
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuffer.mapAsync(GPUMapMode.READ, 0, outputBytes);
    const output = this.readbackBuffer.getMappedRange(0, outputBytes).slice(0);
    this.readbackBuffer.unmap();
    this.hasCachedPrediction = true;
    return output;
  }

  async _refilterWebGPU(renderCamera) {
    const uniform = new Float32Array(64);
    uniform.set(this._frustumPlanes(renderCamera, 'current 60-degree render'), 40);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniform);

    const numInstances = Number(this.meta.numInstances);
    const outputBytes = this.filterResultLayout.totalWords * 4;
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(this.filterResultBuffer);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.filterPipeline);
    pass.setBindGroup(0, this.filterBindGroup);
    pass.dispatchWorkgroups(Math.ceil(numInstances / WORKGROUP_SIZE));
    pass.end();
    encoder.copyBufferToBuffer(
      this.filterResultBuffer,
      0,
      this.filterReadbackBuffer,
      0,
      outputBytes,
    );
    this.device.queue.submit([encoder.finish()]);
    await this.filterReadbackBuffer.mapAsync(GPUMapMode.READ, 0, outputBytes);
    const output = this.filterReadbackBuffer.getMappedRange(0, outputBytes).slice(0);
    this.filterReadbackBuffer.unmap();
    return output;
  }

  async _refilterUnlocked(renderCamera) {
    if (!this.isReady || !this.hasCachedPrediction) return null;
    const startedAt = nowMs();
    const rawBuffer = await this._refilterWebGPU(renderCamera);
    const words = new Uint32Array(rawBuffer);
    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    const renderComponentBitset = words.slice(
      this.filterResultLayout.renderVisibleBits,
      this.filterResultLayout.renderVisibleBits + this.filterResultLayout.instanceBitWords,
    );
    const renderGlbBitset = words.slice(
      this.filterResultLayout.glbVisibleBits,
      this.filterResultLayout.glbVisibleBits + this.filterResultLayout.glbBitWords,
    );
    const counterInstanceCount = Math.min(words[0], numInstances);
    const counterGlbCount = Math.min(words[1], numGlbs);
    if (counterInstanceCount !== countBitsetIds(renderComponentBitset, numInstances)
        || counterGlbCount !== countBitsetIds(renderGlbBitset, numGlbs)) {
      throw new Error('V4 cached refilter bitset counters do not match decoded IDs.');
    }
    const finishedAt = nowMs();
    return {
      idMode: 'global-glb-priority',
      renderComponentBitset,
      renderGlbBitset,
      renderInstanceCount: counterInstanceCount,
      renderGlbCount: counterGlbCount,
      backend: `${this.backend}-cached-render-filter`,
      executionTime: finishedAt - startedAt,
      timings: {
        totalMs: finishedAt - startedAt,
        inferenceMs: 0,
        filterMs: finishedAt - startedAt,
        renderInstanceCount: counterInstanceCount,
        renderGlbCount: counterGlbCount,
        cachedModelPrediction: true,
        readbackBytes: rawBuffer.byteLength,
      },
    };
  }

  async _predictUnlocked(queryCamera, options) {
    if (!this.isReady) return null;
    const startedAt = nowMs();
    const query = this._cameraQuery(queryCamera);
    const prefetchThreshold = Math.max(0, Math.min(1, Number(options.prefetchThreshold ?? 0.04)));
    const inferenceStartedAt = nowMs();
    const rawBuffer = await this._predictWebGPU(
      query,
      options.candidateCamera,
      options.renderCamera || queryCamera,
      prefetchThreshold,
    );
    const inferenceFinishedAt = nowMs();
    const words = new Uint32Array(rawBuffer);
    const floats = new Float32Array(rawBuffer);
    const numInstances = Number(this.meta.numInstances);
    const numGlbs = Number(this.meta.numGlbs);
    const candidateCount = Math.min(words[0], numInstances);
    const visibleCount = Math.min(words[1], numInstances);
    const renderCount = Math.min(words[2], numInstances);
    const glbCount = Math.min(words[3], numGlbs);
    const visibleInstances = words.slice(
      this.readbackLayout.modelVisibleIds,
      this.readbackLayout.modelVisibleIds + visibleCount,
    );
    const renderVisibleInstances = words.slice(
      this.readbackLayout.renderVisibleIds,
      this.readbackLayout.renderVisibleIds + renderCount,
    );
    visibleInstances.sort();
    renderVisibleInstances.sort();
    const glbQueue = [];
    for (let index = 0; index < glbCount; index += 1) {
      const glbId = words[this.readbackLayout.glbQueueIds + index];
      const score = Math.max(0, Math.min(1, floats[this.readbackLayout.glbQueueScores + index]));
      const flags = words[this.readbackLayout.glbQueueFlags + index];
      glbQueue.push({
        globalGlbId: glbId,
        confidence: score,
        importance: score,
        downloadPriority: score,
        visibilityScore: score,
        modelVisible: (flags & GLB_FLAG_MODEL_VISIBLE) !== 0,
        renderVisible: (flags & GLB_FLAG_RENDER_VISIBLE) !== 0,
        prioritySource: 'gpu-max-visibility-probability',
      });
    }
    const sortPriority = (a, b) => b.downloadPriority - a.downloadPriority || a.globalGlbId - b.globalGlbId;
    glbQueue.sort(sortPriority);
    const candidates = glbQueue.filter((item) => item.modelVisible);
    const prefetchCandidates = glbQueue.filter((item) => !item.modelVisible);
    const renderModelList = glbQueue
      .filter((item) => item.renderVisible)
      .map((item) => item.globalGlbId)
      .sort((a, b) => a - b);
    let rawCandidateIds;
    let scores;
    let diagnosticRows;
    if (this.debugLogging) {
      rawCandidateIds = words.slice(
        this.readbackLayout.debugCandidateIds,
        this.readbackLayout.debugCandidateIds + candidateCount,
      );
      diagnosticRows = floats.slice(
        this.readbackLayout.debugRows,
        this.readbackLayout.debugRows + candidateCount * DIAGNOSTIC_OUTPUT_FLOATS,
      );
      scores = new Float32Array(candidateCount);
      for (let index = 0; index < candidateCount; index += 1) {
        scores[index] = diagnosticRows[index * DIAGNOSTIC_OUTPUT_FLOATS];
      }
    }
    this.lastCandidateSelection = {
      source: 'gpu_v4_back_frustum_aabb',
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
      visibleInstanceCount: visibleInstances.length,
      renderInstanceCount: renderVisibleInstances.length,
      visibleGlbCount: candidates.length,
      renderGlbCount: renderModelList.length,
      glbQueueCount: glbQueue.length,
      candidateSelection: this.lastCandidateSelection,
      hasModelDownloadPriority: false,
      gpuFusedCandidateAndFilter: true,
      readbackBytes: rawBuffer.byteLength,
      backend: this.backend,
    };
    return {
      idMode: 'global-glb-priority',
      componentModelList: visibleInstances,
      renderComponentModelList: renderVisibleInstances,
      modelList: candidates.map((item) => item.globalGlbId),
      weightList: candidates.map((item) => item.downloadPriority),
      renderModelList,
      candidates,
      prefetchCandidates,
      glbQueue,
      candidateCount,
      visibleInstanceCount: visibleInstances.length,
      renderVisibleInstanceCount: renderVisibleInstances.length,
      hasModelDownloadPriority: false,
      ...(this.debugLogging ? {
        scores,
        rawCandidateIds,
        diagnosticOutputFloats: this.outputFloats,
        diagnosticRows,
      } : {}),
      backend: this.backend,
      executionTime: finishedAt - startedAt,
      timings: { ...this.lastPredictTimings },
    };
  }

  dispose() {
    for (const buffer of [
      this.uniformBuffer,
      this.runtimeFeatureBuffer,
      this.aabbBuffer,
      this.weightBuffer,
      this.frequencyBuffer,
      this.chiBuffer,
      this.instanceToGlbBuffer,
      this.resultBuffer,
      this.readbackBuffer,
      this.filterResultBuffer,
      this.filterReadbackBuffer,
    ]) buffer?.destroy();
    super.dispose();
  }

}
