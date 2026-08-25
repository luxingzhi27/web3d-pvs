// 单一 V4 实例可见性运行时。
// 浏览器只读取离线实例表，以当前 view-cell 中心和方向做一次 WebGPU 批量查询。
// 离线关系图、子视点、点云编码和邻居查询均不进入本文件。

import { Frustum, Matrix4, Vector3 } from 'three';
import { MODEL_INPUT_FOV_Y_DEG } from './neuralPvsFovProtocol.js';

const RUNTIME_SCHEMA = 'pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4';
const MODEL_SCHEMA = 'pvs-bounded-relation-prior-instance-calibrated-moment-envelope-v4';
const WEIGHTS_SCHEMA = 'pvs-bounded-relation-prior-instance-calibrated-query-weights-v4';
const RAY_SCHEMA = 'viewcell-ray-space-horizontal-disk-v3';
const DEG2RAD = Math.PI / 180;
const WORKGROUP_SIZE = 64;
const DIAGNOSTIC_OUTPUT_FLOATS = 1 + 9 + 18 + 64;
const RESULT_COUNTER_WORDS = 4;
const GLB_FLAG_MODEL_VISIBLE = 1;
const GLB_FLAG_RENDER_VISIBLE = 2;

const EXPECTED_FILES = Object.freeze([
  'instance_runtime_features_fp16.bin',
  'instance_aabb_fp32.bin',
  'instance_to_glb_uint32.bin',
  'query_weights_fp16.bin',
  'frequency_cycles_fp32.bin',
  'chi_table_fp32.bin',
]);

const EXPECTED_WEIGHT_LAYOUT = Object.freeze([
  ['relation_condition_head.0.weight', [16, 28]],
  ['relation_condition_head.0.bias', [16]],
  ['relation_condition_head.2.weight', [8, 16]],
  ['relation_condition_head.2.bias', [8]],
  ['direction_basis_head.0.weight', [24, 19]],
  ['direction_basis_head.0.bias', [24]],
  ['direction_basis_head.2.weight', [4, 24]],
  ['direction_basis_head.2.bias', [4]],
  ['boundary_summary_head.0.weight', [48, 85]],
  ['boundary_summary_head.0.bias', [48]],
  ['boundary_summary_head.2.weight', [8, 48]],
  ['boundary_summary_head.2.bias', [8]],
  ['shared_trunk.0.weight', [64, 130]],
  ['shared_trunk.0.bias', [64]],
  ['shared_trunk.2.weight', [64, 64]],
  ['shared_trunk.2.bias', [64]],
  ['visibility_head.weight', [1, 64]],
  ['visibility_head.bias', [1]],
  ['utility_head.0.weight', [32, 65]],
  ['utility_head.0.bias', [32]],
  ['utility_head.2.weight', [1, 32]],
  ['utility_head.2.bias', [1]],
  ['download_head.0.weight', [32, 65]],
  ['download_head.0.bias', [32]],
  ['download_head.2.weight', [1, 32]],
  ['download_head.2.bias', [1]],
]);

const WEBGPU_SINGLETON_KEY = '__SLM_INSTANCE_PVS_WEBGPU_SINGLETON__';

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function finiteNumber(value, name) {
  const result = Number(value);
  if (!Number.isFinite(result)) throw new Error(`${name} must be finite.`);
  return result;
}

function positiveInteger(value, name) {
  const result = Number(value);
  if (!Number.isInteger(result) || result <= 0) throw new Error(`${name} must be a positive integer.`);
  return result;
}

function sameShape(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => Number(value) === expected[index]);
}

function versionedUrl(url, version) {
  if (!version) return url;
  return `${url}${url.includes('?') ? '&' : '?'}v=${encodeURIComponent(version)}`;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  const text = await response.text();
  if (text.trimStart().startsWith('<')) throw new Error(`Expected JSON from ${url}, received HTML.`);
  return JSON.parse(text);
}

async function fetchBinary(url, expectedBytes) {
  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(`${url} has ${buffer.byteLength} bytes; expected ${expectedBytes}.`);
  }
  return buffer;
}

function packedFp16Words(buffer) {
  const halves = new Uint16Array(buffer);
  if ((halves.length & 1) === 0) return new Uint32Array(buffer);
  const padded = new Uint16Array(halves.length + 1);
  padded.set(halves);
  return new Uint32Array(padded.buffer);
}

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

function activationExpression(kind, value) {
  if (kind === 'silu') return `silu(${value})`;
  if (kind === 'relu') return `max(${value}, 0.0)`;
  if (kind === 'tanh') return `tanh(${value})`;
  return value;
}

function linearFunction(name, inputDim, outputDim, weightOffset, biasOffset, activation = 'none') {
  return `
fn ${name}(input: array<f32, ${inputDim}>) -> array<f32, ${outputDim}> {
  var output: array<f32, ${outputDim}>;
  for (var row = 0u; row < ${outputDim}u; row += 1u) {
    var value = weight_value(${biasOffset}u + row);
    let row_offset = ${weightOffset}u + row * ${inputDim}u;
    for (var col = 0u; col < ${inputDim}u; col += 1u) {
      value += weight_value(row_offset + col) * input[col];
    }
    output[row] = ${activationExpression(activation, 'value')};
  }
  return output;
}`;
}

export class InstancePVS {
  constructor(assetBaseUrl, options = {}) {
    this.assetBaseUrl = String(assetBaseUrl || '').replace(/\/$/, '');
    this.assetVersion = options.assetVersion || null;
    this.preloadedMeta = options.preloadedMeta || null;
    this.debugLogging = Boolean(options.debugLogging);
    this.outputFloats = this.debugLogging ? DIAGNOSTIC_OUTPUT_FLOATS : 1;
    this.meta = null;
    this.weightLayout = new Map();
    this.instanceAabbs = null;
    this.instanceToGlobalGlbArray = null;
    this.isReady = false;
    this.backend = 'uninitialized';
    this.initPromise = null;
    this.initError = null;
    this.lastInitTimings = null;
    this.lastPredictTimings = null;
    this.lastCandidateSelection = null;
    this.predictSerial = 0;
    this.predictQueue = Promise.resolve();

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
    this.resultLayout = null;
    this.readbackLayout = null;

    this._frustum = new Frustum();
    this._frustumMatrix = new Matrix4();
  }

  async init() {
    if (this.isReady) return this;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._initInternal().catch((error) => {
      this.initPromise = null;
      this.initError = error;
      throw error;
    });
    return this.initPromise;
  }

  async _initInternal() {
    const startedAt = nowMs();
    const metaUrl = versionedUrl(`${this.assetBaseUrl}/model_meta.json`, this.assetVersion);
    this.meta = this.preloadedMeta || await fetchJson(metaUrl);
    this._validateMeta();

    const fetchStartedAt = nowMs();
    const buffers = await Promise.all(EXPECTED_FILES.map((file) => {
      const descriptor = this.meta.files[this._fileKey(file)];
      return fetchBinary(
        versionedUrl(`${this.assetBaseUrl}/${file}`, this.assetVersion),
        positiveInteger(descriptor.byteLength, `${file}.byteLength`),
      );
    }));
    const byFile = new Map(EXPECTED_FILES.map((file, index) => [file, buffers[index]]));
    const fetchFinishedAt = nowMs();

    this.instanceAabbs = new Float32Array(byFile.get('instance_aabb_fp32.bin'));
    this.instanceToGlobalGlbArray = new Uint32Array(byFile.get('instance_to_glb_uint32.bin'));
    const runtimeWords = packedFp16Words(byFile.get('instance_runtime_features_fp16.bin'));
    const weightWords = packedFp16Words(byFile.get('query_weights_fp16.bin'));
    const frequencies = new Float32Array(byFile.get('frequency_cycles_fp32.bin'));
    const chi = new Float32Array(byFile.get('chi_table_fp32.bin'));
    this._validateBinaryShapes(runtimeWords, weightWords, frequencies, chi);

    const gpuStartedAt = nowMs();
    await this._initWebGPU(runtimeWords, weightWords, frequencies, chi);
    const finishedAt = nowMs();
    this.isReady = true;
    this.backend = 'webgpu-v4';
    this.initError = null;
    this.lastInitTimings = {
      totalMs: finishedAt - startedAt,
      fetchMs: fetchFinishedAt - fetchStartedAt,
      gpuInitMs: finishedAt - gpuStartedAt,
      assetBytes: EXPECTED_FILES.reduce(
        (sum, file) => sum + Number(this.meta.files[this._fileKey(file)].byteLength),
        0,
      ),
      webgpu: this.webgpuInfo,
    };
    if (this.debugLogging) console.log('[InstancePVS] V4 runtime ready.', this.lastInitTimings);
    return this;
  }

  _fileKey(file) {
    return {
      'instance_runtime_features_fp16.bin': 'runtimeFeatures',
      'instance_aabb_fp32.bin': 'instanceAabb',
      'instance_to_glb_uint32.bin': 'instanceToGlb',
      'query_weights_fp16.bin': 'queryWeights',
      'frequency_cycles_fp32.bin': 'frequency',
      'chi_table_fp32.bin': 'chiTable',
    }[file];
  }

  _validateMeta() {
    const meta = this.meta;
    if (meta?.schema !== RUNTIME_SCHEMA || meta?.modelSchema !== MODEL_SCHEMA) {
      throw new Error(`Unsupported PVS runtime schema: ${meta?.schema || 'missing'}.`);
    }
    if (meta.testRead !== false || meta.safety?.safe !== true) {
      throw new Error('The frontend requires a safe, calibration-frozen V4 export.');
    }
    const config = meta.modelConfig || {};
    const expected = {
      geometryDim: 96,
      runtimeFeatureDim: 124,
      runtimeHeadInputDim: 130,
      boundarySummaryDim: 8,
      lowRankSummaryDim: 4,
      hiddenDim: 64,
    };
    for (const [key, value] of Object.entries(expected)) {
      if (Number(config[key]) !== value) throw new Error(`V4 modelConfig.${key} must equal ${value}.`);
    }
    if (config.runtimeSchema !== MODEL_SCHEMA
      || config.occlusionRepresentation?.mode !== 'survival'
      || config.relationSource !== 'bounded_hierarchical'
      || config.spectralMode !== 'moment_envelope'
      || config.instanceCalibration?.mode !== 'residual'
      || !sameShape(config.survivalCoefficientShape, [4, 7])) {
      throw new Error('The exported model is not the current calibrated survival-field V4 model.');
    }
    if (meta.networkWeights?.schema !== WEIGHTS_SCHEMA
      || meta.query?.raySpace?.schema !== RAY_SCHEMA
      || meta.viewcell?.shape !== 'horizontal_disk') {
      throw new Error('The V4 query contract is incomplete.');
    }
    if (finiteNumber(meta.query.modelInputFovYDeg, 'query.modelInputFovYDeg') !== MODEL_INPUT_FOV_Y_DEG
      || finiteNumber(meta.query.frontendRenderFovYDeg, 'query.frontendRenderFovYDeg') !== 60) {
      throw new Error('The frontend requires the registered 66-degree model / 60-degree render FOV contract.');
    }
    if (finiteNumber(meta.query.candidateCameraBackOffsetM, 'query.candidateCameraBackOffsetM') < 0
      || finiteNumber(meta.query.viewcellRadiusM, 'query.viewcellRadiusM') <= 0) {
      throw new Error('The V4 view-cell camera contract is invalid.');
    }
    const threshold = finiteNumber(meta.threshold, 'threshold');
    if (threshold < 0 || threshold > 1 || Math.abs(threshold - Number(meta.calibrationFrozenThreshold)) > 1e-7) {
      throw new Error('The V4 calibration threshold is invalid.');
    }
    const numInstances = positiveInteger(meta.numInstances, 'numInstances');
    positiveInteger(meta.numGlbs, 'numGlbs');
    if (!sameShape(meta.fixedTable?.shape, [numInstances, 124])) {
      throw new Error('The V4 fixed instance table must have shape [N, 124].');
    }
    for (const file of EXPECTED_FILES) {
      const key = this._fileKey(file);
      const descriptor = meta.files?.[key];
      if (!descriptor || descriptor.file !== file) throw new Error(`Missing V4 file descriptor for ${file}.`);
    }
    const layout = meta.networkWeights.layout;
    if (!Array.isArray(layout) || layout.length !== EXPECTED_WEIGHT_LAYOUT.length) {
      throw new Error('Unexpected V4 query-weight layout length.');
    }
    for (let index = 0; index < EXPECTED_WEIGHT_LAYOUT.length; index += 1) {
      const [name, shape] = EXPECTED_WEIGHT_LAYOUT[index];
      const item = layout[index];
      if (item.name !== name || !sameShape(item.shape, shape) || item.dtype !== 'float16') {
        throw new Error(`Unexpected V4 query-weight layout at ${name}.`);
      }
      this.weightLayout.set(name, item);
    }
  }

  _validateBinaryShapes(runtimeWords, weightWords, frequencies, chi) {
    const count = Number(this.meta.numInstances);
    if (this.instanceAabbs.length !== count * 6) throw new Error('V4 AABB table length mismatch.');
    if (this.instanceToGlobalGlbArray.length !== count) throw new Error('V4 instance-to-GLB table length mismatch.');
    if (runtimeWords.length * 2 !== count * 124) throw new Error('V4 fixed feature table length mismatch.');
    const weightElements = this.meta.networkWeights.layout.reduce(
      (maximum, item) => Math.max(maximum, Number(item.offsetElements) + Number(item.elementCount)),
      0,
    );
    if (weightWords.length * 2 < weightElements || weightWords.length * 2 > weightElements + 1) {
      throw new Error('V4 packed query-weight table length mismatch.');
    }
    if (frequencies.length !== 16 * 9) throw new Error('V4 frequency table must have 16 x 9 values.');
    if (chi.length !== 8192) throw new Error('V4 chi table must have 8192 values.');
    for (let index = 0; index < this.instanceAabbs.length; index += 1) {
      if (!Number.isFinite(this.instanceAabbs[index])) throw new Error('V4 AABB table contains non-finite values.');
    }
    const numGlbs = Number(this.meta.numGlbs);
    for (let index = 0; index < this.instanceToGlobalGlbArray.length; index += 1) {
      if (this.instanceToGlobalGlbArray[index] >= numGlbs) {
        throw new Error('V4 instance-to-GLB table contains an out-of-range GLB ID.');
      }
    }
  }

  _weightOffset(name) {
    const item = this.weightLayout.get(name);
    if (!item) throw new Error(`Missing V4 query weight ${name}.`);
    return Number(item.offsetElements);
  }

  _createStorageBuffer(data) {
    const buffer = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  async _initWebGPU(runtimeWords, weightWords, frequencies, chi) {
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
    const shaderModule = this.device.createShaderModule({ code: this._buildShader() });
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
    const glbShaderModule = this.device.createShaderModule({ code: this._buildGlbCompactionShader() });
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
  }

  _frustumPlanes(camera, name) {
    if (!camera) throw new Error(`V4 prediction requires the ${name} camera.`);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    this._frustumMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._frustumMatrix);
    const output = new Float32Array(24);
    for (let index = 0; index < 6; index += 1) {
      const plane = this._frustum.planes[index];
      const offset = index * 4;
      output[offset] = plane.normal.x;
      output[offset + 1] = plane.normal.y;
      output[offset + 2] = plane.normal.z;
      output[offset + 3] = plane.constant;
    }
    return output;
  }

  _cameraQuery(queryCamera) {
    if (!queryCamera || typeof queryCamera.getWorldDirection !== 'function') {
      throw new Error('V4 prediction requires the current render camera as its query center.');
    }
    queryCamera.updateMatrixWorld(true);
    const forward = new Vector3(0, 0, -1);
    queryCamera.getWorldDirection(forward).normalize();
    const tanY = Math.tan(MODEL_INPUT_FOV_Y_DEG * DEG2RAD * 0.5);
    const aspect = Number.isFinite(queryCamera.aspect) ? Number(queryCamera.aspect) : 16 / 9;
    return {
      center: [queryCamera.position.x, queryCamera.position.y, queryCamera.position.z],
      forward: [forward.x, forward.y, forward.z],
      tanX: tanY * Math.max(1e-4, aspect),
      tanY,
    };
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
    return output;
  }

  async predict(queryCamera, options = {}) {
    const run = this.predictQueue.catch(() => {}).then(() => this._predictUnlocked(queryCamera, options));
    this.predictQueue = run.catch(() => {});
    return run;
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
    ]) buffer?.destroy();
    this.isReady = false;
  }

  _buildShader() {
    const offset = (name) => this._weightOffset(name);
    const layers = [
      linearFunction('relation_hidden', 28, 16, offset('relation_condition_head.0.weight'), offset('relation_condition_head.0.bias'), 'silu'),
      linearFunction('relation_output', 16, 8, offset('relation_condition_head.2.weight'), offset('relation_condition_head.2.bias')),
      linearFunction('boundary_hidden', 85, 48, offset('boundary_summary_head.0.weight'), offset('boundary_summary_head.0.bias'), 'silu'),
      linearFunction('boundary_output', 48, 8, offset('boundary_summary_head.2.weight'), offset('boundary_summary_head.2.bias'), 'tanh'),
      linearFunction('direction_hidden', 19, 24, offset('direction_basis_head.0.weight'), offset('direction_basis_head.0.bias'), 'silu'),
      linearFunction('direction_output', 24, 4, offset('direction_basis_head.2.weight'), offset('direction_basis_head.2.bias'), 'tanh'),
      linearFunction('trunk_hidden0', 130, 64, offset('shared_trunk.0.weight'), offset('shared_trunk.0.bias'), 'relu'),
      linearFunction('trunk_hidden1', 64, 64, offset('shared_trunk.2.weight'), offset('shared_trunk.2.bias'), 'relu'),
      linearFunction('visibility_output', 64, 1, offset('visibility_head.weight'), offset('visibility_head.bias')),
    ].join('\n');
    const layout = this.resultLayout;
    const diagnosticWrites = this.debugLogging ? `
  atomicStore(&results[${layout.debugCandidateIds}u + candidate_index], instance_id);
  let diagnostic_offset = ${layout.debugRows}u + candidate_index * ${DIAGNOSTIC_OUTPUT_FLOATS}u;
  atomicStore(&results[diagnostic_offset], bitcast<u32>(probability));
  for (var value_index = 0u; value_index < 9u; value_index += 1u) {
    atomicStore(&results[diagnostic_offset + 1u + value_index], bitcast<u32>(query.center[value_index]));
  }
  for (var value_index = 0u; value_index < 18u; value_index += 1u) {
    atomicStore(&results[diagnostic_offset + 10u + value_index], bitcast<u32>(query.axes[value_index]));
  }
  for (var value_index = 0u; value_index < 64u; value_index += 1u) {
    atomicStore(&results[diagnostic_offset + 28u + value_index], bitcast<u32>(spectral[value_index]));
  }` : '';

    return `
struct Uniforms {
  query_center_threshold: vec4<f32>,
  forward_tan_x: vec4<f32>,
  query_parameters: vec4<f32>,
  depth_count: vec4<f32>,
  candidate_planes: array<vec4<f32>, 6>,
  render_planes: array<vec4<f32>, 6>,
};

struct RayQuery {
  center: array<f32, 9>,
  axes: array<f32, 18>,
  distance: f32,
  radius: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> runtime_words: array<u32>;
@group(0) @binding(2) var<storage, read> instance_aabbs: array<f32>;
@group(0) @binding(3) var<storage, read> weight_words: array<u32>;
@group(0) @binding(4) var<storage, read> frequency_cycles: array<f32>;
@group(0) @binding(5) var<storage, read> chi_table: array<f32>;
@group(0) @binding(6) var<storage, read> instance_to_glb: array<u32>;
@group(0) @binding(7) var<storage, read_write> results: array<atomic<u32>>;

fn runtime_value(index: u32) -> f32 {
  let pair = unpack2x16float(runtime_words[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn weight_value(index: u32) -> f32 {
  let pair = unpack2x16float(weight_words[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn sigmoid(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-value));
}

fn silu(value: f32) -> f32 {
  return value * sigmoid(value);
}

fn softplus(value: f32) -> f32 {
  return log(1.0 + exp(-abs(value))) + max(value, 0.0);
}

fn safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let length_value = length(value);
  return select(fallback, value / length_value, length_value > 1e-6);
}

fn intersects_frustum(instance_id: u32, render_frustum: bool) -> bool {
  let aabb_offset = instance_id * 6u;
  let minimum = vec3<f32>(
    instance_aabbs[aabb_offset],
    instance_aabbs[aabb_offset + 1u],
    instance_aabbs[aabb_offset + 2u]
  );
  let maximum = vec3<f32>(
    instance_aabbs[aabb_offset + 3u],
    instance_aabbs[aabb_offset + 4u],
    instance_aabbs[aabb_offset + 5u]
  );
  for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
    let plane = select(
      uniforms.candidate_planes[plane_index],
      uniforms.render_planes[plane_index],
      render_frustum
    );
    let positive = select(minimum, maximum, plane.xyz >= vec3<f32>(0.0));
    if (dot(plane.xyz, positive) + plane.w < 0.0) { return false; }
  }
  return true;
}

fn ray_differential(
  axis: vec3<f32>,
  ray: vec3<f32>,
  distance: f32,
  radius: f32,
  forward: vec3<f32>,
  right: vec3<f32>,
  up: vec3<f32>,
  tan_x: f32,
  tan_y: f32,
  dot_forward: f32,
  dot_right: f32,
  dot_up: f32,
  denominator_x_raw: f32,
  denominator_y_raw: f32,
  denominator_x: f32,
  denominator_y: f32,
  screen_u_raw: f32,
  screen_v_raw: f32,
  horizontal_raw: f32,
  vertical_raw: f32,
) -> array<f32, 9> {
  var result: array<f32, 9>;
  let ray_axis = dot(ray, axis);
  let d_distance = -ray_axis;
  let d_ray = -(axis - ray * ray_axis) / distance;
  let d_log = d_distance / (4.0 * (100.0 + distance));
  let d_forward = dot(d_ray, forward);
  let d_right = dot(d_ray, right);
  let d_up = dot(d_ray, up);
  let d_denominator_x_raw = sign(dot_forward) * d_forward * tan_x;
  let d_denominator_y_raw = sign(dot_forward) * d_forward * tan_y;
  let d_denominator_x = select(0.0, d_denominator_x_raw, denominator_x_raw > 1e-4);
  let d_denominator_y = select(0.0, d_denominator_y_raw, denominator_y_raw > 1e-4);
  let d_u_raw = (d_right * denominator_x - dot_right * d_denominator_x) / (denominator_x * denominator_x);
  let d_v_raw = (d_up * denominator_y - dot_up * d_denominator_y) / (denominator_y * denominator_y);
  let d_angular = select(0.0, -radius * d_distance / (distance * distance), distance > 1.0);
  let d_horizontal_raw = d_angular / tan_x;
  let d_vertical_raw = d_angular / tan_y;
  result[0] = d_ray.x;
  result[1] = d_ray.y;
  result[2] = d_ray.z;
  result[3] = d_log;
  result[4] = d_forward;
  result[5] = select(0.0, d_u_raw / 4.0, abs(screen_u_raw) < 4.0);
  result[6] = select(0.0, d_v_raw / 4.0, abs(screen_v_raw) < 4.0);
  result[7] = select(0.0, d_horizontal_raw / 2.0, horizontal_raw > 0.0 && horizontal_raw < 4.0);
  result[8] = select(0.0, d_vertical_raw / 2.0, vertical_raw > 0.0 && vertical_raw < 4.0);
  return result;
}

fn make_ray_query(instance_id: u32) -> RayQuery {
  let aabb_offset = instance_id * 6u;
  let minimum = vec3<f32>(
    instance_aabbs[aabb_offset],
    instance_aabbs[aabb_offset + 1u],
    instance_aabbs[aabb_offset + 2u]
  );
  let maximum = vec3<f32>(
    instance_aabbs[aabb_offset + 3u],
    instance_aabbs[aabb_offset + 4u],
    instance_aabbs[aabb_offset + 5u]
  );
  let center_world = 0.5 * (minimum + maximum);
  let size = max(maximum - minimum, vec3<f32>(1e-4));
  let delta = center_world - uniforms.query_center_threshold.xyz;
  let distance = max(length(delta), 1e-4);
  let ray = delta / distance;
  let radius = 0.5 * length(size);
  let forward = safe_normalize(uniforms.forward_tan_x.xyz, vec3<f32>(0.0, 0.0, -1.0));
  let world_y = vec3<f32>(0.0, 1.0, 0.0);
  let up_seed = select(world_y, vec3<f32>(0.0, 0.0, 1.0), abs(dot(forward, world_y)) > 0.98);
  let right = safe_normalize(cross(forward, up_seed), vec3<f32>(1.0, 0.0, 0.0));
  let up = safe_normalize(cross(right, forward), vec3<f32>(0.0, 1.0, 0.0));
  let tan_x = max(uniforms.forward_tan_x.w, 1e-4);
  let tan_y = max(uniforms.query_parameters.x, 1e-4);
  let dot_forward = dot(ray, forward);
  let dot_right = dot(ray, right);
  let dot_up = dot(ray, up);
  let denominator_x_raw = abs(dot_forward) * tan_x;
  let denominator_y_raw = abs(dot_forward) * tan_y;
  let denominator_x = max(denominator_x_raw, 1e-4);
  let denominator_y = max(denominator_y_raw, 1e-4);
  let screen_u_raw = dot_right / denominator_x;
  let screen_v_raw = dot_up / denominator_y;
  let safe_distance = max(distance, 1.0);
  let angular = radius / safe_distance;
  let horizontal_raw = angular / tan_x;
  let vertical_raw = angular / tan_y;
  var result: RayQuery;
  result.center[0] = ray.x;
  result.center[1] = ray.y;
  result.center[2] = ray.z;
  result.center[3] = clamp(log(1.0 + distance / 100.0) / 4.0, 0.0, 2.0) - 1.0;
  result.center[4] = clamp(dot_forward, -1.0, 1.0);
  result.center[5] = clamp(screen_u_raw, -4.0, 4.0) / 4.0;
  result.center[6] = clamp(screen_v_raw, -4.0, 4.0) / 4.0;
  result.center[7] = clamp(horizontal_raw, 0.0, 4.0) / 2.0 - 1.0;
  result.center[8] = clamp(vertical_raw, 0.0, 4.0) / 2.0 - 1.0;
  let differential_x = ray_differential(
    vec3<f32>(1.0, 0.0, 0.0), ray, distance, radius, forward, right, up,
    tan_x, tan_y, dot_forward, dot_right, dot_up, denominator_x_raw,
    denominator_y_raw, denominator_x, denominator_y, screen_u_raw,
    screen_v_raw, horizontal_raw, vertical_raw
  );
  let differential_z = ray_differential(
    vec3<f32>(0.0, 0.0, 1.0), ray, distance, radius, forward, right, up,
    tan_x, tan_y, dot_forward, dot_right, dot_up, denominator_x_raw,
    denominator_y_raw, denominator_x, denominator_y, screen_u_raw,
    screen_v_raw, horizontal_raw, vertical_raw
  );
  let viewcell_radius = uniforms.query_parameters.y;
  for (var feature = 0u; feature < 9u; feature += 1u) {
    let raw_axis = vec2<f32>(
      differential_x[feature] * viewcell_radius,
      differential_z[feature] * viewcell_radius
    );
    let row_norm = length(raw_axis);
    let budget = max(1.0 - abs(result.center[feature]), 0.0);
    let scale = select(1.0, budget / max(row_norm, 1e-12), row_norm > budget);
    result.axes[feature * 2u] = raw_axis.x * scale;
    result.axes[feature * 2u + 1u] = raw_axis.y * scale;
  }
  result.distance = distance;
  result.radius = radius;
  return result;
}

fn lookup_chi(argument: f32) -> f32 {
  let position = clamp(argument, 0.0, 320.0) * (8191.0 / 320.0);
  let lower = min(u32(floor(position)), 8190u);
  let fraction = position - f32(lower);
  return chi_table[lower] + fraction * (chi_table[lower + 1u] - chi_table[lower]);
}

fn spectral_features(query: RayQuery) -> array<f32, 64> {
  var output: array<f32, 64>;
  for (var frequency = 0u; frequency < 16u; frequency += 1u) {
    var phase_cycles = 0.0;
    var projected_x = 0.0;
    var projected_z = 0.0;
    for (var feature = 0u; feature < 9u; feature += 1u) {
      let cycle = frequency_cycles[frequency * 9u + feature];
      phase_cycles += query.center[feature] * cycle;
      projected_x += query.axes[feature * 2u] * cycle;
      projected_z += query.axes[feature * 2u + 1u] * cycle;
    }
    let phase = 6.283185307179586 * phase_cycles;
    let radial = 6.283185307179586 * length(vec2<f32>(projected_x, projected_z));
    let chi_s = lookup_chi(radial);
    let chi_2s = lookup_chi(2.0 * radial);
    let mean_sin = sin(phase) * chi_s;
    let mean_cos = cos(phase) * chi_s;
    let chi_squared = chi_s * chi_s;
    let phase_double_cos = cos(2.0 * phase);
    let shared_variance = 0.5 * (1.0 - chi_squared);
    let directed_variance = 0.5 * phase_double_cos * (chi_squared - chi_2s);
    let variance_sin = max(shared_variance + directed_variance, 0.0);
    let variance_cos = max(shared_variance - directed_variance, 0.0);
    var std_sin = select(0.0, sqrt(max(variance_sin, 1.1920929e-7)), variance_sin > 1.1920929e-7);
    var std_cos = select(0.0, sqrt(max(variance_cos, 1.1920929e-7)), variance_cos > 1.1920929e-7);
    if (radial == 0.0) {
      std_sin = 0.0;
      std_cos = 0.0;
    }
    let base = frequency * 4u;
    output[base] = mean_sin;
    output[base + 1u] = mean_cos;
    output[base + 2u] = std_sin;
    output[base + 3u] = std_cos;
  }
  return output;
}

fn low_rank_summary(query: RayQuery) -> array<f32, 4> {
  var output: array<f32, 4>;
  var axis_x_squared = 0.0;
  var axis_z_squared = 0.0;
  var maximum = 0.0;
  for (var feature = 0u; feature < 9u; feature += 1u) {
    let axis_x = query.axes[feature * 2u];
    let axis_z = query.axes[feature * 2u + 1u];
    axis_x_squared += axis_x * axis_x;
    axis_z_squared += axis_z * axis_z;
    maximum = max(maximum, max(abs(axis_x), abs(axis_z)));
  }
  output[0] = sqrt(axis_x_squared);
  output[1] = sqrt(axis_z_squared);
  output[2] = sqrt(axis_x_squared + axis_z_squared);
  output[3] = maximum;
  return output;
}

${layers}

fn survival_semantic(
  basis: array<f32, 4>,
  coefficients: array<f32, 28>,
  normalized_depth: f32,
) -> array<f32, 8> {
  var parameters: array<f32, 7>;
  for (var parameter = 0u; parameter < 7u; parameter += 1u) {
    var value = 0.0;
    for (var rank = 0u; rank < 4u; rank += 1u) {
      value += basis[rank] * coefficients[rank * 7u + parameter];
    }
    parameters[parameter] = value;
  }
  let no_block = sigmoid(parameters[0]);
  let weight_max = max(parameters[1], parameters[2]);
  let weight_exp0 = exp(parameters[1] - weight_max);
  let weight_exp1 = exp(parameters[2] - weight_max);
  let weight_sum = weight_exp0 + weight_exp1;
  var weight0 = weight_exp0 / weight_sum;
  var weight1 = weight_exp1 / weight_sum;
  var depth0 = 0.05 + 0.90 * sigmoid(parameters[3]);
  var depth1 = 0.05 + 0.90 * sigmoid(parameters[4]);
  var scale0 = 0.03 + softplus(parameters[5]);
  var scale1 = 0.03 + softplus(parameters[6]);
  if (depth0 > depth1) {
    let old_depth = depth0;
    let old_weight = weight0;
    let old_scale = scale0;
    depth0 = depth1;
    weight0 = weight1;
    scale0 = scale1;
    depth1 = old_depth;
    weight1 = old_weight;
    scale1 = old_scale;
  }
  let cdf0 = sigmoid((normalized_depth - depth0) / max(scale0, 1e-3));
  let cdf1 = sigmoid((normalized_depth - depth1) / max(scale1, 1e-3));
  let cdf = (1.0 - no_block) * (weight0 * cdf0 + weight1 * cdf1);
  let survival = clamp(1.0 - cdf, 1e-5, 1.0);
  let expected_depth = weight0 * depth0 + weight1 * depth1;
  let variance = weight0 * (depth0 - expected_depth) * (depth0 - expected_depth)
    + weight1 * (depth1 - expected_depth) * (depth1 - expected_depth);
  let uncertainty = sqrt(max(variance, 1e-8));
  let local_slope = (1.0 - no_block) * (
    weight0 / max(scale0, 1e-3) * cdf0 * (1.0 - cdf0)
    + weight1 / max(scale1, 1e-3) * cdf1 * (1.0 - cdf1)
  );
  var output: array<f32, 8>;
  output[0] = survival;
  output[1] = 1.0 - survival;
  output[2] = no_block;
  output[3] = expected_depth;
  output[4] = uncertainty;
  output[5] = weight0;
  output[6] = weight1;
  output[7] = local_slope;
  return output;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let instance_id = global_id.x;
  let count = u32(uniforms.depth_count.y);
  if (instance_id >= count || !intersects_frustum(instance_id, false)) { return; }
  let candidate_index = atomicAdd(&results[0], 1u);
  let query = make_ray_query(instance_id);
  let runtime_offset = instance_id * 124u;
  var coefficients: array<f32, 28>;
  for (var value_index = 0u; value_index < 28u; value_index += 1u) {
    coefficients[value_index] = runtime_value(runtime_offset + 96u + value_index);
  }
  let relation = relation_output(relation_hidden(coefficients));
  let spectral = spectral_features(query);
  let low_rank = low_rank_summary(query);
  var boundary_input: array<f32, 85>;
  for (var value_index = 0u; value_index < 64u; value_index += 1u) {
    boundary_input[value_index] = spectral[value_index];
  }
  for (var value_index = 0u; value_index < 9u; value_index += 1u) {
    boundary_input[64u + value_index] = query.center[value_index];
  }
  for (var value_index = 0u; value_index < 4u; value_index += 1u) {
    boundary_input[73u + value_index] = low_rank[value_index];
  }
  for (var value_index = 0u; value_index < 8u; value_index += 1u) {
    boundary_input[77u + value_index] = relation[value_index];
  }
  let boundary = boundary_output(boundary_hidden(boundary_input));
  var direction_input: array<f32, 19>;
  for (var value_index = 0u; value_index < 8u; value_index += 1u) {
    direction_input[value_index] = boundary[value_index];
    direction_input[8u + value_index] = relation[value_index];
  }
  direction_input[16] = query.center[0];
  direction_input[17] = query.center[1];
  direction_input[18] = query.center[2];
  let basis = direction_output(direction_hidden(direction_input));
  let depth_raw = log(1.0 + query.distance / (query.radius + uniforms.depth_count.x));
  let normalized_depth = clamp(
    (depth_raw - uniforms.query_parameters.z)
      / (uniforms.query_parameters.w - uniforms.query_parameters.z),
    0.0,
    1.0
  );
  let semantic = survival_semantic(basis, coefficients, normalized_depth);
  var trunk_input: array<f32, 130>;
  for (var value_index = 0u; value_index < 96u; value_index += 1u) {
    trunk_input[value_index] = runtime_value(runtime_offset + value_index);
  }
  for (var value_index = 0u; value_index < 4u; value_index += 1u) {
    trunk_input[96u + value_index] = basis[value_index];
  }
  for (var value_index = 0u; value_index < 8u; value_index += 1u) {
    trunk_input[100u + value_index] = semantic[value_index];
    trunk_input[108u + value_index] = boundary[value_index];
  }
  for (var value_index = 0u; value_index < 9u; value_index += 1u) {
    trunk_input[116u + value_index] = query.center[value_index];
  }
  for (var value_index = 0u; value_index < 4u; value_index += 1u) {
    trunk_input[125u + value_index] = low_rank[value_index];
  }
  trunk_input[129] = normalized_depth;
  let hidden = trunk_hidden1(trunk_hidden0(trunk_input));
  let logit = visibility_output(hidden)[0];
  let probability = sigmoid(logit);
  let global_glb_id = instance_to_glb[instance_id];
  atomicMax(&results[${layout.glbScores}u + global_glb_id], bitcast<u32>(probability));
  atomicOr(&results[${layout.glbFlags}u + global_glb_id], 4u);
  if (probability >= uniforms.query_center_threshold.w) {
    let visible_index = atomicAdd(&results[1], 1u);
    atomicStore(&results[${layout.modelVisibleIds}u + visible_index], instance_id);
    atomicOr(&results[${layout.glbFlags}u + global_glb_id], ${GLB_FLAG_MODEL_VISIBLE}u);
    if (intersects_frustum(instance_id, true)) {
      let render_index = atomicAdd(&results[2], 1u);
      atomicStore(&results[${layout.renderVisibleIds}u + render_index], instance_id);
      atomicOr(&results[${layout.glbFlags}u + global_glb_id], ${GLB_FLAG_RENDER_VISIBLE}u);
    }
  }
${diagnosticWrites}
}`;
  }

  _buildGlbCompactionShader() {
    const layout = this.resultLayout;
    return `
struct Uniforms {
  query_center_threshold: vec4<f32>,
  forward_tan_x: vec4<f32>,
  query_parameters: vec4<f32>,
  depth_count: vec4<f32>,
  candidate_planes: array<vec4<f32>, 6>,
  render_planes: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> results: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let glb_id = global_id.x;
  if (glb_id >= u32(uniforms.depth_count.w)) { return; }
  let flags = atomicLoad(&results[${layout.glbFlags}u + glb_id]);
  if ((flags & 4u) == 0u) { return; }
  let score_bits = atomicLoad(&results[${layout.glbScores}u + glb_id]);
  let score = bitcast<f32>(score_bits);
  if (score < uniforms.depth_count.z) { return; }
  let queue_index = atomicAdd(&results[3], 1u);
  atomicStore(&results[${layout.glbQueueIds}u + queue_index], glb_id);
  atomicStore(&results[${layout.glbQueueScores}u + queue_index], score_bits);
  atomicStore(&results[${layout.glbQueueFlags}u + queue_index], flags);
}`;
  }
}
