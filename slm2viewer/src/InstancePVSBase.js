import { Frustum, Matrix4, Vector3 } from 'three';
import { MODEL_INPUT_FOV_Y_DEG } from './neuralPvsFovProtocol.js';

export const RUNTIME_SCHEMA = 'pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4';
export const MODEL_SCHEMA = 'pvs-bounded-relation-prior-instance-calibrated-moment-envelope-v4';
export const WEIGHTS_SCHEMA = 'pvs-bounded-relation-prior-instance-calibrated-query-weights-v4';
export const RAY_SCHEMA = 'viewcell-ray-space-horizontal-disk-v3';
export const DIAGNOSTIC_OUTPUT_FLOATS = 1 + 9 + 18 + 64;

export const EXPECTED_FILES = Object.freeze([
  'instance_runtime_features_fp16.bin',
  'instance_aabb_fp32.bin',
  'instance_to_glb_uint32.bin',
  'query_weights_fp16.bin',
  'frequency_cycles_fp32.bin',
  'chi_table_fp32.bin',
]);

export const EXPECTED_WEIGHT_LAYOUT = Object.freeze([
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

const DEG2RAD = Math.PI / 180;

export function nowMs() {
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

export class InstancePVSBase {
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
    this.hasCachedPrediction = false;
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

    const backendStartedAt = nowMs();
    await this._initializeBackend(runtimeWords, weightWords, frequencies, chi);
    const finishedAt = nowMs();
    this.isReady = true;
    this.initError = null;
    this.lastInitTimings = {
      totalMs: finishedAt - startedAt,
      fetchMs: fetchFinishedAt - fetchStartedAt,
      backendInitMs: finishedAt - backendStartedAt,
      assetBytes: EXPECTED_FILES.reduce(
        (sum, file) => sum + Number(this.meta.files[this._fileKey(file)].byteLength),
        0,
      ),
    };
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
    for (const value of this.instanceAabbs) {
      if (!Number.isFinite(value)) throw new Error('V4 AABB table contains non-finite values.');
    }
    const numGlbs = Number(this.meta.numGlbs);
    for (const glbId of this.instanceToGlobalGlbArray) {
      if (glbId >= numGlbs) throw new Error('V4 instance-to-GLB table contains an out-of-range GLB ID.');
    }
  }

  _weightOffset(name) {
    const item = this.weightLayout.get(name);
    if (!item) throw new Error(`Missing V4 query weight ${name}.`);
    return Number(item.offsetElements);
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

  async predict(queryCamera, options = {}) {
    const run = this.predictQueue.catch(() => {}).then(() => this._predictUnlocked(queryCamera, options));
    this.predictQueue = run.catch(() => {});
    return run;
  }

  async refilter(renderCamera) {
    const run = this.predictQueue.catch(() => {}).then(() => this._refilterUnlocked(renderCamera));
    this.predictQueue = run.catch(() => {});
    return run;
  }

  dispose() {
    this.hasCachedPrediction = false;
    this.isReady = false;
  }
}
