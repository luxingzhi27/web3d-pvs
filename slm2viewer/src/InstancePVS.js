// InstancePVS 前端运行时。
//
// 用途：
//   1. 从 assets/neural_instance_culling/v* 加载 instance_model_meta*.json 和 instance_pvs_assets*.bin。
//   2. 用 CPU 对实例 AABB 做视锥过滤，得到候选 componentGlobalId。
//   3. 用 WebGPU compute 对候选实例并行推理遮挡可见性。
//   4. 输出可见 componentModelList、实例可见性分数和 GLB 下载优先级。
//      新 utility scheduler 资产应由模型直接输出 downloadPriority；旧模型缺失该输出时
//      只把 visibilityScore 作为兼容 fallback，不再把它当作新模型的下载头。
//
// 重要约束：
//   - 当前主线是实例级剔除，不要在这里恢复旧 chunk/cascaded 逻辑。
//   - WebGPU adapter/device 必须走全局单例，避免热更新或重复进入页面时反复 requestAdapter。
//   - 当前部署主线是 directional occlusion proxy：浏览器只使用固定实例特征表、
//     ray-space 查询、方向遮挡代理选择、可见性头和模型输出的 GLB 下载优先级。
//   - v3 screen-grid / fixed-geo / dynamic-pool 兼容分支暂时保留用于离线对照。

import { Box3, Frustum, Matrix4, Vector3 } from 'three';
import { MODEL_INPUT_FOV_Y_DEG } from './neuralPvsFovProtocol.js';

const DEG2RAD = Math.PI / 180;

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function sceneMinMax(sceneBounds) {
  const size = sceneBounds.size;
  if (sceneBounds.min && sceneBounds.max) return { min: sceneBounds.min, max: sceneBounds.max, size };
  const center = sceneBounds.center;
  return {
    min: center.map((v, i) => v - size[i] * 0.5),
    max: center.map((v, i) => v + size[i] * 0.5),
    size,
  };
}

function normalizePoint(point, bounds) {
  return [
    (point.x - bounds.min[0]) / bounds.size[0],
    (point.y - bounds.min[1]) / bounds.size[1],
    (point.z - bounds.min[2]) / bounds.size[2],
  ].map((v) => Math.max(0, Math.min(1, v)));
}

function layoutMap(meta) {
  const out = new Map();
  for (const item of meta.layout || []) out.set(item.name, item);
  return out;
}

function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const mantissa = value & 0x03ff;
  if (exponent === 0) {
    return sign * (mantissa / 1024) * 2 ** -14;
  }
  if (exponent === 31) {
    return mantissa ? NaN : sign * Infinity;
  }
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function normalizeArray3(value, fallback = [0, 0, -1]) {
  const x = Number(value?.[0] ?? fallback[0]);
  const y = Number(value?.[1] ?? fallback[1]);
  const z = Number(value?.[2] ?? fallback[2]);
  const len = Math.hypot(x, y, z);
  if (!(len > 1e-6)) return fallback.slice();
  return [x / len, y / len, z / len];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

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
  // Adapter 是浏览器/GPU 驱动层的重量级资源。
  // 开发环境热更新或页面频繁刷新时，重复 requestAdapter 可能触发驱动卡顿。
  // 所以这里必须走 globalThis 单例。
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
  // Device 也复用；如果 device lost，再清理单例，让下次预测重新创建。
  const state = getSharedWebGPUState();
  if (state.device) return state.device;
  if (!state.devicePromise) {
    state.devicePromise = adapter.requestDevice()
      .then((device) => {
        state.device = device;
        device.lost.then(() => {
          if (state.device === device) {
            state.device = null;
            state.devicePromise = null;
          }
        });
        return device;
      })
      .catch((error) => {
        state.devicePromise = null;
        throw error;
      });
  }
  return state.devicePromise;
}

export class InstancePVS {
  constructor(assetBaseUrl, options = {}) {
    this.assetBaseUrl = (assetBaseUrl || '').replace(/\/$/, '');
    this.runtimeMetaUrl = options.runtimeMetaUrl || null;
    this.meta = null;
    this.runtimeMeta = null;
    this.assetBuffer = null;
    this.gpuWeightData = null;
    this.layout = null;
    this.sceneBounds = null;
    this.cameraBounds = null;
    this.instanceBoxes = [];
    this.instanceWorldAabbs = null;
    this.spatialAabbIndex = null;
    this.lastCandidateSelection = null;
    this._spatialScratchBox = new Box3();
    this.instanceToGlobalGlb = [];
    this.instanceToGlobalGlbArray = null;
    // Some exported feature tables use compact local rows while the renderer
    // still addresses scene-global component IDs.
    this.localToGlobalInstance = [];
    this.globalToLocalInstance = new Map();
    this.isReady = false;
    this.backend = 'uninitialized';
    this.debugLogging = Boolean(options.debugLogging);
    this.assetVersion = options.assetVersion || null;
    this.configuredInferenceFovYDeg = MODEL_INPUT_FOV_Y_DEG;
    this.inferenceFovYDeg = MODEL_INPUT_FOV_Y_DEG;
    this.deferRuntimeMeta = Boolean(options.deferRuntimeMeta);
    this.runtimeMetaProvider = typeof options.runtimeMetaProvider === 'function' ? options.runtimeMetaProvider : null;
    this.lastInitTimings = null;
    this.initPromise = null;
    this.initError = null;

    this.adapter = null;
    this.device = null;
    this.pipeline = null;
    this.bindGroup = null;
    this.uniformBuffer = null;
    this.weightBuffer = null;
    this.candidateBuffer = null;
    this.neighborBuffer = null;
    this.instanceToGlbBuffer = null;
    this.outputBuffer = null;
    this.readbackBuffer = null;
    this.capacity = 0;
    this.workgroupSize = 64;
    this.predictSerial = 0;
    this.predictQueue = Promise.resolve();
    this.lastPredictTimings = null;
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
    // 初始化步骤：
    // 1. 拉取 meta 与二进制权重。
    // 2. 构建实例 AABB、instanceToGlobalGlb 等运行时索引。
    // 3. 创建 WebGPU buffer / pipeline。
    const initStart = nowMs();
    const fetchStart = nowMs();
    this.meta = await this._fetchJson(`${this.assetBaseUrl}/instance_model_meta.json`);
    const assetFile = this.meta.assetFile || (this.meta.runtimeSchema === 'oa-ge-ivn-v3'
      ? 'instance_pvs_assets_v3.bin'
      : 'instance_pvs_assets.bin');
    const assetUrl = this._versionedUrl(`${this.assetBaseUrl}/${assetFile}`);
    const assetResponse = await fetch(assetUrl, { cache: 'force-cache' });
    const fetchEnd = nowMs();
    if (!assetResponse.ok) throw new Error(`Failed to load ${assetUrl}: ${assetResponse.status}`);

    const parseStart = nowMs();
    const assetBuffer = await assetResponse.arrayBuffer();
    const parseEnd = nowMs();
    this.layout = layoutMap(this.meta);
    this.assetBuffer = assetBuffer;
    this.sceneBounds = sceneMinMax(this.meta.sceneBounds);
    this.cameraBounds = sceneMinMax(this.meta.cameraBounds || this.meta.sceneBounds);
    this.inferenceFovYDeg = MODEL_INPUT_FOV_Y_DEG;
    this._buildInstanceIdMaps();
    this.instanceToGlobalGlb = this.meta.instanceToGlobalGlb || [];
    this.instanceToGlobalGlbArray = this._buildInstanceToGlbArray();
    this._validateRuntimeLayout();
    if (this._isV3Runtime() || this._isDirectionalOcclusionProxyRuntime() || this._isUtilityDynamicPoolRuntime()) {
      this._buildV3AabbCacheFromAsset(assetBuffer);
      this._rebuildSpatialAabbIndex();
    }

    const runtimeStart = nowMs();
    const providedRuntimeMeta = this.runtimeMetaProvider ? this.runtimeMetaProvider() : null;
    if (providedRuntimeMeta) {
      this.setRuntimeMeta(providedRuntimeMeta);
    } else if (this.runtimeMetaUrl && !this.deferRuntimeMeta) {
      this.setRuntimeMeta(await this._fetchJson(this.runtimeMetaUrl, { optional: true }));
    }
    const runtimeEnd = nowMs();

    const gpuStart = nowMs();
    const gpuReady = await this._tryInitWebGPU();
    const gpuEnd = nowMs();
    if (!gpuReady) {
      throw new Error('InstancePVS requires WebGPU support.');
    }
    const readyEnd = nowMs();
    this.backend = 'webgpu';
    this.isReady = true;
    this.initError = null;
    this.lastInitTimings = {
      totalMs: readyEnd - initStart,
      fetchMs: fetchEnd - fetchStart,
      parseMs: parseEnd - parseStart + runtimeEnd - runtimeStart,
      decodeMs: 0,
      gpuInitMs: gpuEnd - gpuStart,
      webgpu: this.lastWebGPUTimings || null,
      deferredWebGPU: false,
    };
    if (this.debugLogging) {
      console.log(`[InstancePVS] ready backend=${this.backend} instances=${this.meta.numInstances}`, this.lastInitTimings);
    }
    return this;
  }

  requestWebGPUUpgrade() {
    return this.backend === 'webgpu';
  }

  _versionedUrl(url) {
    if (!url || !this.assetVersion) return url;
    return `${url}${url.indexOf('?') >= 0 ? '&' : '?'}v=${encodeURIComponent(this.assetVersion)}`;
  }

  async _fetchJson(rawUrl, options = {}) {
    const url = this._versionedUrl(rawUrl);
    const response = await fetch(url, { cache: 'force-cache' });
    if (!response.ok) {
      if (options.optional) return null;
      throw new Error(`Failed to load JSON ${url}: ${response.status}`);
    }

    const text = await response.text();
    const firstChar = text.trimStart().charAt(0);
    if (firstChar === '<') {
      throw new Error(`Expected JSON from ${url}, but received HTML. Check the asset path or server fallback.`);
    }

    try {
      return JSON.parse(text);
    } catch (error) {
      throw new Error(`Failed to parse JSON ${url}: ${error && error.message ? error.message : String(error)}`);
    }
  }

  _isV3Runtime() {
    return this.meta && this.meta.runtimeSchema === 'oa-ge-ivn-v3';
  }

  _isUtilityDynamicPoolRuntime() {
    return this.meta && this.meta.runtimeSchema === 'utility-preserving-asset-scheduler-dynamic-occlusion-pool-v1';
  }

  _isDirectionalOcclusionProxyRuntime() {
    return this.meta && this.meta.runtimeSchema === 'directional-occlusion-proxy-scheduler-v1';
  }

  _usesV3Attention() {
    return this._isV3Runtime()
      && Number(this.meta.attentionDim ?? 0) > 0
      && this.meta.usesOccluderAttention !== false
      && Number(this.meta.occluderK ?? 0) > 0;
  }

  _outputValueWords() {
    const hasPriorityOutput = this.meta?.outputsDownloadPriority === true || this.meta?.outputsGlbPriority === true;
    return Math.max(1, Number(this.meta?.outputValueWords || (hasPriorityOutput ? 2 : 1)) | 0);
  }

  _hasModelDownloadPriority() {
    return this.meta?.outputsDownloadPriority === true || this.meta?.outputsGlbPriority === true;
  }

  _decodePackedVisibility(word) {
    const packed = Number(word || 0) >>> 0;
    return {
      visible: (packed & 1) !== 0,
      score: ((packed >>> 1) & 0xffff) / 65535,
    };
  }

  _decodeScoreWord(word) {
    const packed = Number(word || 0) >>> 0;
    // Preferred compact format is q16 in the low 16 bits. If a shader reuses the
    // packed visibility format for the second word, this still decodes the high
    // q16 probability correctly.
    const q16 = (packed > 0xffff) ? ((packed >>> 1) & 0xffff) : (packed & 0xffff);
    return q16 / 65535;
  }

  setRuntimeMeta(runtimeMeta) {
    if (!runtimeMeta || !this.meta) {
      this.runtimeMeta = runtimeMeta || null;
      return 0;
    }
    this.runtimeMeta = runtimeMeta;
    const cache = this._buildInstanceBoxCache(runtimeMeta);
    this.instanceBoxes = cache.boxes;
    if (cache.aabbs) this.instanceWorldAabbs = cache.aabbs;
    this._rebuildSpatialAabbIndex();
    return this.instanceBoxes.length;
  }

  _validateRuntimeLayout() {
    if (this.meta.usesRuntimeTriplane) throw new Error('InstancePVS runtime must not require Triplane assets.');
    if (this._isDirectionalOcclusionProxyRuntime()) {
      const required = [
        'runtime_features', 'instance_world_aabbs',
        'ray_proxy_gate_w0', 'ray_proxy_gate_b0', 'ray_proxy_gate_w1', 'ray_proxy_gate_b1',
        'ray_runtime_gate_w', 'ray_runtime_gate_b',
        'mlp_vis_cam_inter_w', 'mlp_vis_cam_inter_b', 'mlp_vis_inst_inter_w', 'mlp_vis_inst_inter_b',
        'mlp_vis_w0', 'mlp_vis_b0', 'mlp_vis_w1', 'mlp_vis_b1', 'mlp_vis_w2', 'mlp_vis_b2',
        'inhibition_w0', 'inhibition_b0', 'inhibition_w1', 'inhibition_b1',
        'utility_w0', 'utility_b0', 'utility_w1', 'utility_b1', 'utility_w2', 'utility_b2',
        'download_w0', 'download_b0', 'download_w1', 'download_b1',
      ];
      for (const name of required) this._offset(name);
      return;
    }
    if (this._isUtilityDynamicPoolRuntime()) {
      const required = [
        'runtime_features', 'instance_world_aabbs', 'occlusion_pool_ids',
        'ray_gate_w', 'ray_gate_b',
        'mlp_vis_cam_inter_w', 'mlp_vis_cam_inter_b', 'mlp_vis_inst_inter_w', 'mlp_vis_inst_inter_b',
        'mlp_vis_w0', 'mlp_vis_b0', 'mlp_vis_w1', 'mlp_vis_b1', 'mlp_vis_w2', 'mlp_vis_b2',
        'edge_shared_w0', 'edge_shared_b0', 'edge_shared_w1', 'edge_shared_b1',
        'edge_occ_w', 'edge_occ_b', 'edge_supp_w', 'edge_supp_b',
        'utility_w0', 'utility_b0', 'utility_w1', 'utility_b1', 'utility_w2', 'utility_b2',
        'download_w0', 'download_b0', 'download_w1', 'download_b1',
      ];
      for (const name of required) this._offset(name);
      return;
    }
    if (this._isV3Runtime()) {
      const levels = this.meta.cameraHashCellSizesM || [];
      for (let level = 0; level < levels.length; level += 1) this._offset(`camera_hash_l${level}`);
      const required = [
        'mlp_vis_w0', 'mlp_vis_b0', 'mlp_vis_w1', 'mlp_vis_b1', 'mlp_vis_w2', 'mlp_vis_b2',
        'glb_geo_features', 'instance_world_aabbs',
      ];
      if (this._usesV3Attention()) {
        required.push(
        'attn_q_w', 'attn_q_b', 'attn_k_w', 'attn_k_b', 'attn_v_w', 'attn_v_b', 'attn_out_w', 'attn_out_b',
        );
      }
      for (const name of required) {
        this._offset(name);
      }
      return;
    }
    if (this.meta.runtimeSchema !== 'camera-hash-latent-geo') {
      throw new Error(`Unsupported InstancePVS runtime schema: ${this.meta.runtimeSchema}`);
    }
    const levels = this.meta.cameraHashCellSizesM || [];
    for (let level = 0; level < levels.length; level += 1) this._offset(`camera_hash_l${level}`);
    this._offset('latent_codes');
    this._offset('geo_summaries');
    if (Number(this.meta.relativeFeatureDim || 0) > 0) this._offset('instance_world_aabbs');
    if (Number(this.meta.visibilityInteractionDim || 0) > 0) {
      this._offset('mlp_vis_cam_inter_w');
      this._offset('mlp_vis_cam_inter_b');
      this._offset('mlp_vis_inst_inter_w');
      this._offset('mlp_vis_inst_inter_b');
    }
  }

  _buildInstanceBoxCache(runtimeMeta) {
    const boxes = new Array(this.meta.numInstances);
    const aabbs = new Float32Array(this.meta.numInstances * 6);
    for (const record of runtimeMeta.componentRecords || []) {
      if (!record || !record.bounds) continue;
      const globalId = Number(record.componentGlobalId);
      if (!Number.isFinite(globalId)) continue;
      const id = this.globalToLocalInstance.has(globalId)
        ? this.globalToLocalInstance.get(globalId)
        : (this.meta.globalInstanceIds ? null : globalId);
      if (id == null || !Number.isFinite(id) || id < 0 || id >= this.meta.numInstances) continue;
      const center = record.bounds.center;
      const size = record.bounds.size;
      let min;
      let max;
      if (record.bounds.min && record.bounds.max) {
        min = record.bounds.min.map(Number);
        max = record.bounds.max.map(Number);
      } else if (center && size) {
        min = [center[0] - size[0] * 0.5, center[1] - size[1] * 0.5, center[2] - size[2] * 0.5];
        max = [center[0] + size[0] * 0.5, center[1] + size[1] * 0.5, center[2] + size[2] * 0.5];
      } else {
        continue;
      }
      boxes[id] = new Box3(
        new Vector3(min[0], min[1], min[2]),
        new Vector3(max[0], max[1], max[2]),
      );
      const base = id * 6;
      aabbs[base + 0] = min[0];
      aabbs[base + 1] = min[1];
      aabbs[base + 2] = min[2];
      aabbs[base + 3] = max[0];
      aabbs[base + 4] = max[1];
      aabbs[base + 5] = max[2];
    }
    return { boxes, aabbs };
  }

  _buildInstanceToGlbArray() {
    const count = Number(this.meta?.numInstances || 0);
    const out = new Uint32Array(count);
    for (let i = 0; i < count; i += 1) {
      const value = Number(this.instanceToGlobalGlb[i]);
      out[i] = Number.isFinite(value) && value >= 0 ? value >>> 0 : 0;
    }
    return out;
  }

  _rebuildSpatialAabbIndex() {
    const count = Number(this.meta?.numInstances || 0);
    const boxes = this.instanceBoxes;
    if (!count || !boxes || boxes.length === 0 || !this.sceneBounds) {
      this.spatialAabbIndex = null;
      return;
    }
    const configuredCellSize = Number(this.meta?.spatialAabbCellSizeM ?? 64);
    const cellSize = Number.isFinite(configuredCellSize) && configuredCellSize > 0 ? configuredCellSize : 64;
    const configuredMaxCells = Number(this.meta?.spatialAabbMaxCellsPerInstance ?? 128);
    const maxCellsPerInstance = Number.isFinite(configuredMaxCells) && configuredMaxCells >= 1
      ? Math.floor(configuredMaxCells)
      : 128;
    const origin = this.sceneBounds.min.map(Number);
    const buckets = new Map();
    const overflowIds = [];
    let minCell = [Infinity, Infinity, Infinity];
    let maxCell = [-Infinity, -Infinity, -Infinity];
    let indexedInstanceCount = 0;
    for (let id = 0; id < Math.min(count, boxes.length); id += 1) {
      const box = boxes[id];
      if (!box || !box.min || !box.max) continue;
      const lo = [
        Math.floor((box.min.x - origin[0]) / cellSize),
        Math.floor((box.min.y - origin[1]) / cellSize),
        Math.floor((box.min.z - origin[2]) / cellSize),
      ];
      const hi = [
        Math.floor((box.max.x - origin[0]) / cellSize),
        Math.floor((box.max.y - origin[1]) / cellSize),
        Math.floor((box.max.z - origin[2]) / cellSize),
      ];
      const span = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1);
      if (!Number.isFinite(span) || span > maxCellsPerInstance) {
        overflowIds.push(id);
        continue;
      }
      indexedInstanceCount += 1;
      for (let ix = lo[0]; ix <= hi[0]; ix += 1) {
        for (let iy = lo[1]; iy <= hi[1]; iy += 1) {
          for (let iz = lo[2]; iz <= hi[2]; iz += 1) {
            const key = `${ix},${iy},${iz}`;
            let bucket = buckets.get(key);
            if (!bucket) {
              bucket = [];
              buckets.set(key, bucket);
            }
            bucket.push(id);
          }
        }
      }
      for (let axis = 0; axis < 3; axis += 1) {
        minCell[axis] = Math.min(minCell[axis], lo[axis]);
        maxCell[axis] = Math.max(maxCell[axis], hi[axis]);
      }
    }
    if (!buckets.size && !overflowIds.length) {
      this.spatialAabbIndex = null;
      return;
    }
    this.spatialAabbIndex = {
      cellSize,
      origin,
      buckets,
      overflowIds,
      minCell: minCell.map((value) => Number.isFinite(value) ? value : 0),
      maxCell: maxCell.map((value) => Number.isFinite(value) ? value : 0),
      indexedInstanceCount,
      overflowInstanceCount: overflowIds.length,
      maxQueryCells: Number(this.meta?.spatialAabbMaxQueryCells ?? 100000),
    };
  }

  _buildInstanceIdMaps() {
    const count = Number(this.meta?.numInstances || 0);
    const configured = Array.isArray(this.meta?.globalInstanceIds)
      ? this.meta.globalInstanceIds
      : null;
    this.localToGlobalInstance = new Array(count);
    this.globalToLocalInstance = new Map();
    for (let localId = 0; localId < count; localId += 1) {
      const configuredId = configured && configured[localId] != null
        ? Number(configured[localId])
        : localId;
      const globalId = Number.isFinite(configuredId) && configuredId >= 0
        ? configuredId
        : localId;
      this.localToGlobalInstance[localId] = globalId;
      this.globalToLocalInstance.set(globalId, localId);
    }
  }

  _globalIdsToLocal(globalIds) {
    const result = [];
    for (const value of globalIds || []) {
      const globalId = Number(value);
      if (!Number.isFinite(globalId)) continue;
      const localId = this.globalToLocalInstance.has(globalId)
        ? this.globalToLocalInstance.get(globalId)
        : null;
      if (localId != null) result.push(localId);
    }
    return Array.from(new Set(result));
  }

  _localInstanceIdToGlobal(localId) {
    const index = Number(localId);
    if (Number.isFinite(index) && this.localToGlobalInstance[index] != null) {
      return Number(this.localToGlobalInstance[index]);
    }
    return index;
  }

  _buildV3AabbCacheFromAsset(assetBuffer) {
    const item = this.layout.get('instance_world_aabbs');
    if (!item || !assetBuffer) return;
    const count = Number(this.meta.numInstances || 0);
    const half = new Uint16Array(assetBuffer, item.byteOffset, count * 6);
    const aabbs = new Float32Array(count * 6);
    const boxes = new Array(count);
    for (let id = 0; id < count; id += 1) {
      const base = id * 6;
      for (let k = 0; k < 6; k += 1) aabbs[base + k] = halfToFloat(half[base + k]);
      boxes[id] = new Box3(
        new Vector3(aabbs[base + 0], aabbs[base + 1], aabbs[base + 2]),
        new Vector3(aabbs[base + 3], aabbs[base + 4], aabbs[base + 5]),
      );
    }
    this.instanceWorldAabbs = aabbs;
    this.instanceBoxes = boxes;
  }

  _frustumCandidateIds(camera) {
    // CPU 候选过滤：先用实例 AABB 去掉明显不在视野附近的实例。
    // 神经网络只负责遮挡可见性，不负责基础视锥裁剪。
    if (!camera || this.instanceBoxes.length === 0) {
      const ids = Array.from({ length: this.meta.numInstances }, (_v, i) => i);
      this.lastCandidateSelection = { source: 'all_instances_no_aabb_cache', candidateCount: ids.length };
      return ids;
    }
    // 候选视锥使用固定的 66° 模型查询口径；真实 60° 视锥只负责最终显示。
    const candidateCamera = camera.clone();
    candidateCamera.fov = MODEL_INPUT_FOV_Y_DEG;
    candidateCamera.updateProjectionMatrix();
    candidateCamera.updateMatrixWorld(true);
    const matrix = new Matrix4().multiplyMatrices(candidateCamera.projectionMatrix, candidateCamera.matrixWorldInverse);
    const frustum = new Frustum().setFromProjectionMatrix(matrix);
    const index = this.spatialAabbIndex;
    if (index) {
      const corners = [];
      let finiteCorners = true;
      for (const z of [-1, 1]) {
        for (const y of [-1, 1]) {
          for (const x of [-1, 1]) {
            const point = new Vector3(x, y, z).unproject(candidateCamera);
            if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) {
              finiteCorners = false;
            }
            corners.push(point);
          }
        }
      }
      if (finiteCorners) {
        const boundsMin = [Infinity, Infinity, Infinity];
        const boundsMax = [-Infinity, -Infinity, -Infinity];
        for (const point of corners) {
          boundsMin[0] = Math.min(boundsMin[0], point.x);
          boundsMin[1] = Math.min(boundsMin[1], point.y);
          boundsMin[2] = Math.min(boundsMin[2], point.z);
          boundsMax[0] = Math.max(boundsMax[0], point.x);
          boundsMax[1] = Math.max(boundsMax[1], point.y);
          boundsMax[2] = Math.max(boundsMax[2], point.z);
        }
        const lo = boundsMin.map((value, axis) => Math.floor((value - index.origin[axis]) / index.cellSize) - 1);
        const hi = boundsMax.map((value, axis) => Math.floor((value - index.origin[axis]) / index.cellSize) + 1);
        const rawCellCount = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1);
        const maxQueryCells = Number.isFinite(index.maxQueryCells) && index.maxQueryCells > 0
          ? index.maxQueryCells
          : 100000;
        if (Number.isFinite(rawCellCount) && rawCellCount <= maxQueryCells) {
          const seen = new Set(index.overflowIds);
          for (let ix = lo[0]; ix <= hi[0]; ix += 1) {
            for (let iy = lo[1]; iy <= hi[1]; iy += 1) {
              for (let iz = lo[2]; iz <= hi[2]; iz += 1) {
                const cellMin = new Vector3(
                  index.origin[0] + ix * index.cellSize,
                  index.origin[1] + iy * index.cellSize,
                  index.origin[2] + iz * index.cellSize,
                );
                this._spatialScratchBox.min.copy(cellMin);
                this._spatialScratchBox.max.set(
                  cellMin.x + index.cellSize,
                  cellMin.y + index.cellSize,
                  cellMin.z + index.cellSize,
                );
                if (!frustum.intersectsBox(this._spatialScratchBox)) continue;
                const bucket = index.buckets.get(`${ix},${iy},${iz}`);
                if (bucket) for (const id of bucket) seen.add(id);
              }
            }
          }
          const ids = [];
          for (const id of seen) {
            const box = this.instanceBoxes[id];
            if (box && frustum.intersectsBox(box)) ids.push(id);
          }
          this.lastCandidateSelection = {
            source: 'spatial_aabb_index',
            queryCellCount: rawCellCount,
            indexedInstanceCount: index.indexedInstanceCount,
            overflowInstanceCount: index.overflowInstanceCount,
            candidateCount: ids.length,
          };
          return ids;
        }
      }
    }
    const ids = [];
    for (let i = 0; i < this.instanceBoxes.length; i += 1) {
      const box = this.instanceBoxes[i];
      if (box && frustum.intersectsBox(box)) ids.push(i);
    }
    this.lastCandidateSelection = {
      source: 'full_aabb_scan',
      candidateCount: ids.length,
      indexedInstanceCount: index?.indexedInstanceCount ?? 0,
      overflowInstanceCount: index?.overflowInstanceCount ?? 0,
    };
    return ids;
  }

  _cameraBasisFromView(cameraView) {
    const forward = normalizeArray3(cameraView, [0, 0, -1]);
    const upSeed = Math.abs(forward[1]) > 0.98 ? [0, 0, 1] : [0, 1, 0];
    const right = normalizeArray3(cross3(forward, upSeed), [1, 0, 0]);
    const up = normalizeArray3(cross3(right, forward), [0, 1, 0]);
    return { forward, right, up };
  }

  _buildV3CandidateContext(candidateIds, cameraWorld, cameraView) {
    // v3 的遮挡上下文必须和训练集一致：候选按相机深度从近到远排序，
    // 每个实例只关注更近且屏幕投影邻近/重叠的 K 个潜在遮挡者。
    const ids = Array.from(candidateIds || [], (id) => Number(id) >>> 0);
    const count = ids.length;
    const k = Math.max(0, Number(this.meta.occluderK ?? 0) | 0);
    if (!this._usesV3Attention() || count === 0 || k === 0 || !this.instanceWorldAabbs) {
      return { candidateIds: ids, neighborIndices: new Uint32Array(count * Math.max(1, k)).fill(0xffffffff) };
    }
    const basis = this._cameraBasisFromView(cameraView);
    const tanX = Math.max(1e-4, Number(cameraView[3] || 1));
    const tanY = Math.max(1e-4, Number(cameraView[4] || 1));
    const metrics = [];
    for (const id of ids) {
      const base = id * 6;
      const minX = this.instanceWorldAabbs[base + 0];
      const minY = this.instanceWorldAabbs[base + 1];
      const minZ = this.instanceWorldAabbs[base + 2];
      const maxX = this.instanceWorldAabbs[base + 3];
      const maxY = this.instanceWorldAabbs[base + 4];
      const maxZ = this.instanceWorldAabbs[base + 5];
      if (![minX, minY, minZ, maxX, maxY, maxZ].every(Number.isFinite)) continue;
      const cx = (minX + maxX) * 0.5;
      const cy = (minY + maxY) * 0.5;
      const cz = (minZ + maxZ) * 0.5;
      const sx = Math.max(maxX - minX, 1e-4);
      const sy = Math.max(maxY - minY, 1e-4);
      const sz = Math.max(maxZ - minZ, 1e-4);
      const dx = cx - cameraWorld[0];
      const dy = cy - cameraWorld[1];
      const dz = cz - cameraWorld[2];
      const delta = [dx, dy, dz];
      const depth = dot3(delta, basis.forward);
      const dist = Math.max(1e-4, Math.hypot(dx, dy, dz));
      const dotRight = dot3(delta, basis.right) / dist;
      const dotUp = dot3(delta, basis.up) / dist;
      const dotForward = Math.max(depth / dist, 1e-4);
      const radius = Math.hypot(sx, sy, sz) * 0.5;
      const angular = radius / Math.max(dist, 1.0);
      metrics.push({
        id,
        depth,
        u: clamp(dotRight / Math.max(dotForward * tanX, 1e-4), -8, 8),
        v: clamp(dotUp / Math.max(dotForward * tanY, 1e-4), -8, 8),
        ex: clamp(angular / tanX, 0, 4),
        ey: clamp(angular / tanY, 0, 4),
      });
    }
    metrics.sort((a, b) => a.depth - b.depth || a.id - b.id);
    const sortedIds = metrics.map((item) => item.id);
    const neighbors = new Uint32Array(metrics.length * k);
    neighbors.fill(0xffffffff);
    const search = Math.max(k, Number(this.meta.neighborSearch ?? 256) | 0);
    for (let i = 0; i < metrics.length; i += 1) {
      const start = Math.max(0, i - search);
      if (start >= i) continue;
      const scored = [];
      const current = metrics[i];
      for (let j = start; j < i; j += 1) {
        const other = metrics[j];
        const du = Math.max(0, Math.abs(other.u - current.u) - (other.ex + current.ex + 0.08));
        const dv = Math.max(0, Math.abs(other.v - current.v) - (other.ey + current.ey + 0.08));
        const depthGap = Math.max(0, current.depth - other.depth);
        scored.push({ index: j, score: du * du + dv * dv + 0.00005 * depthGap });
      }
      scored.sort((a, b) => a.score - b.score || a.index - b.index);
      const limit = Math.min(k, scored.length);
      for (let n = 0; n < limit; n += 1) neighbors[i * k + n] = scored[n].index >>> 0;
    }
    return { candidateIds: sortedIds, neighborIndices: neighbors };
  }

  async _tryInitWebGPU() {
    return this._tryInitWebGPUMainThread();
  }

  async _tryInitWebGPUMainThread() {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      this.lastWebGPUTimings = {
        unavailableReason: typeof window !== 'undefined' && window.isSecureContext === false
          ? 'navigator.gpu unavailable: insecure context'
          : 'navigator.gpu unavailable',
      };
      if (this.debugLogging) {
        console.warn('[InstancePVS] navigator.gpu unavailable.', this.lastWebGPUTimings);
      }
      return false;
    }

    try {
      const sharedWebGPU = getSharedWebGPUState();
      const adapterCached = Boolean(sharedWebGPU.adapter);
      const adapterStart = nowMs();
      // Adapter/Device 是浏览器与驱动层的重量级资源。放到 globalThis 单例里，
      // 避免开发环境 HMR 或频繁刷新时反复 requestAdapter/requestDevice 把驱动句柄打爆。
      this.adapter = await getSharedWebGPUAdapter();
      if (!this.adapter) {
        this.lastWebGPUTimings = {
          requestAdapterMs: nowMs() - adapterStart,
          unavailableReason: 'requestAdapter returned null',
          powerPreference: 'high-performance',
        };
        if (this.debugLogging) console.warn('[InstancePVS] requestAdapter() returned null.', this.lastWebGPUTimings);
        return false;
      }

      const deviceStart = nowMs();
      const deviceCached = Boolean(sharedWebGPU.device);
      this.device = await getSharedWebGPUDevice(this.adapter);

      const rawWeights = new Uint16Array(this.assetBuffer);
      if ((rawWeights.length & 1) === 0) {
        this.gpuWeightData = new Uint32Array(this.assetBuffer);
      } else {
        const padded = new Uint16Array(rawWeights.length + 1);
        padded.set(rawWeights);
        this.gpuWeightData = new Uint32Array(padded.buffer);
      }
      this.uniformBuffer = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.weightBuffer = this.device.createBuffer({ size: this.gpuWeightData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.weightBuffer, 0, this.gpuWeightData);
      if (this._isV3Runtime()) {
        this.instanceToGlbBuffer = this.device.createBuffer({
          size: Math.max(4, this.instanceToGlobalGlbArray.byteLength),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.instanceToGlbBuffer, 0, this.instanceToGlobalGlbArray);
      }
      this.assetBuffer = null;
      this.gpuWeightData = null;
      const pipelineStart = nowMs();
      const pipelineDesc = {
        layout: 'auto',
        compute: {
          module: this.device.createShaderModule({ code: this._buildShader() }),
          entryPoint: 'main',
        },
      };
      this.pipeline = this.device.createComputePipelineAsync
        ? await this.device.createComputePipelineAsync(pipelineDesc)
        : this.device.createComputePipeline(pipelineDesc);
      this.lastWebGPUTimings = {
        requestAdapterMs: deviceStart - adapterStart,
        requestDeviceMs: pipelineStart - deviceStart,
        pipelineMs: nowMs() - pipelineStart,
        adapterCached,
        deviceCached,
        powerPreference: 'high-performance',
        packedFp16: true,
      };
      if (this.debugLogging) console.log('[InstancePVS] Using WebGPU compute backend.', this.lastWebGPUTimings);
      return true;
    } catch (error) {
      if (this.debugLogging) console.warn('[InstancePVS] WebGPU init failed.', error);
      this.adapter = null;
      this.device = null;
      this.pipeline = null;
      return false;
    }
  }

  _ensureCapacity(count) {
    if (count <= this.capacity) return;
    this.capacity = Math.max(64, 2 ** Math.ceil(Math.log2(count)));
    const candidateBytes = this.capacity * 4;
    const neighborBytes = this.capacity * Math.max(1, Number(this.meta.occluderK ?? 1)) * 4;
    const outputBytes = this.capacity * this._outputValueWords() * 4;
    this.candidateBuffer = this.device.createBuffer({ size: candidateBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.neighborBuffer = this._isV3Runtime()
      ? this.device.createBuffer({ size: neighborBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
      : null;
    this.outputBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.readbackBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let entries;
    if (this._isV3Runtime()) {
      entries = [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.candidateBuffer } },
        { binding: 3, resource: { buffer: this.neighborBuffer } },
        { binding: 4, resource: { buffer: this.instanceToGlbBuffer } },
        { binding: 5, resource: { buffer: this.outputBuffer } },
      ];
    } else if (this._isDirectionalOcclusionProxyRuntime() || this._isUtilityDynamicPoolRuntime()) {
      entries = [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.candidateBuffer } },
        { binding: 4, resource: { buffer: this.outputBuffer } },
      ];
    } else {
      entries = [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.candidateBuffer } },
        { binding: 3, resource: { buffer: this.outputBuffer } },
      ];
    }
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries,
    });
  }

  async _predictWebGPU(cameraNorm, cameraWorld, cameraView, candidateIds, neighborIndices = null) {
    // WebGPU 推理主路径：每个 candidate instance 对应一个 compute thread。
    // v3 shader 内部读取 camera hash、视角、相对几何、GLB 几何特征和遮挡邻居，输出可见 bit 与置信度。
    this._ensureCapacity(candidateIds.length);
    // Uniform: 相机位置、方向/FOV、候选数量。每个 GPU 线程处理一个候选实例。
    const uniform = new Float32Array([
      cameraNorm[0],
      cameraNorm[1],
      cameraNorm[2],
      this.meta.visibilityThreshold,
      cameraView[0],
      cameraView[1],
      cameraView[2],
      cameraView[3],
      cameraView[4],
      candidateIds.length,
      0,
      0,
      cameraWorld[0],
      cameraWorld[1],
      cameraWorld[2],
      0,
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniform);
    this.device.queue.writeBuffer(this.candidateBuffer, 0, new Uint32Array(candidateIds));
    if (this._isV3Runtime()) {
      const empty = neighborIndices || new Uint32Array(candidateIds.length * Math.max(1, Number(this.meta.occluderK ?? 1))).fill(0xffffffff);
      this.device.queue.writeBuffer(this.neighborBuffer, 0, empty);
    }

    // Compute 输出第 1 个 u32 为 packed visibility：bit0 表示是否可见，bit1 起保存 q16 可见性分数。
    // 支持未来新模型用第 2 个 u32 输出归一化下载优先级；旧模型没有第 2 个 word 时，
    // postprocess 会把 visibilityScore 作为兼容 fallback。
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(candidateIds.length / this.workgroupSize));
    pass.end();
    const outputBytes = candidateIds.length * this._outputValueWords() * 4;
    encoder.copyBufferToBuffer(this.outputBuffer, 0, this.readbackBuffer, 0, outputBytes);
    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ, 0, outputBytes);
    const out = new Uint32Array(this.readbackBuffer.getMappedRange(0, outputBytes)).slice();
    this.readbackBuffer.unmap();
    return out;
  }

  _offset(name) {
    const item = this.layout.get(name);
    if (!item) throw new Error(`Missing layout item ${name}`);
    return item.floatOffset;
  }

  _wordOffset(name) {
    const item = this.layout.get(name);
    if (!item) throw new Error(`Missing layout item ${name}`);
    return Number(item.wordOffset ?? Math.floor(Number(item.byteOffset || 0) / 4));
  }

  async predict(cameraOrPosition, rotationOrOptions = {}, maybeCamera = null) {
    const run = this.predictQueue
      .catch(() => { })
      .then(() => this._predictUnlocked(cameraOrPosition, rotationOrOptions, maybeCamera));
    this.predictQueue = run.catch(() => { });
    return run;
  }

  async _predictUnlocked(cameraOrPosition, rotationOrOptions = {}, maybeCamera = null) {
    // 对外预测入口的实际实现。
    // 这里把 Three.js camera 转成模型需要的 camera_norm、camera_world、camera_forward 和固定神经 FOV。
    if (!this.isReady) return null;
    const start = performance.now();
    const legacyOptions = rotationOrOptions && !rotationOrOptions.isEuler ? rotationOrOptions : {};
    const renderCamera = maybeCamera && maybeCamera.projectionMatrix
      ? maybeCamera
      : (cameraOrPosition && cameraOrPosition.projectionMatrix ? cameraOrPosition : legacyOptions.camera);
    const cameraPos = renderCamera?.position || legacyOptions.position || cameraOrPosition;
    const candidateStart = performance.now();
    let candidateIds = (legacyOptions.candidateIds || this._frustumCandidateIds(renderCamera)).slice().sort((a, b) => a - b);
    let candidateEnd = performance.now();
    const cameraNorm = normalizePoint(cameraPos, this.cameraBounds);
    const viewDir = new Vector3(0, 0, -1);
    if (renderCamera && typeof renderCamera.getWorldDirection === 'function') {
      renderCamera.getWorldDirection(viewDir);
    }
    // 模型输入的 FOV 必须固定为训练口径。当前屏幕/相机 FOV 只用于候选视锥和最终渲染，不喂给网络。
    const tanHalfFovY = Math.tan(Number(this.inferenceFovYDeg || 66) * DEG2RAD * 0.5);
    const aspect = renderCamera && Number.isFinite(renderCamera.aspect) ? Number(renderCamera.aspect) : 16 / 9;
    const cameraView = [viewDir.x, viewDir.y, viewDir.z, tanHalfFovY * aspect, tanHalfFovY];
    const cameraWorld = [cameraPos.x, cameraPos.y, cameraPos.z];
    let neighborIndices = null;
    if (this._usesV3Attention()) {
      const context = this._buildV3CandidateContext(candidateIds, cameraWorld, cameraView);
      candidateIds = context.candidateIds;
      neighborIndices = context.neighborIndices;
      candidateEnd = performance.now();
    }
    if (candidateIds.length === 0) {
      const emptyEnd = performance.now();
      this.predictSerial += 1;
      this.lastPredictTimings = {
        serial: this.predictSerial,
        totalMs: emptyEnd - start,
        candidateMs: candidateEnd - candidateStart,
        candidateSelection: this.lastCandidateSelection,
        inferenceMs: 0,
        postMs: 0,
        candidateCount: 0,
        visibleInstanceCount: 0,
        visibleGlbCount: 0,
        hasModelDownloadPriority: this._hasModelDownloadPriority(),
        backend: this.backend,
      };
      return {
        idMode: 'global-glb-priority',
        componentModelList: [],
        modelList: [],
        weightList: [],
        candidates: [],
        prefetchCandidates: [],
        candidateCount: 0,
        visibleInstanceCount: 0,
        hasModelDownloadPriority: this._hasModelDownloadPriority(),
        backend: this.backend,
        executionTime: emptyEnd - start,
        timings: { ...this.lastPredictTimings },
      };
    }
    const inferenceStart = performance.now();
    if (this.backend !== 'webgpu') {
      throw new Error(`InstancePVS expected WebGPU backend, got ${this.backend}`);
    }
    const raw = await this._predictWebGPU(cameraNorm, cameraWorld, cameraView, candidateIds, neighborIndices);
    const inferenceEnd = performance.now();

    const postStart = performance.now();
    const returnAllScores = Boolean(legacyOptions.returnAllScores);
    const prefetchThreshold = Math.max(0, Math.min(1, Number(
      legacyOptions.prefetchThreshold !== undefined
        ? legacyOptions.prefetchThreshold
        : (this.meta.prefetchThreshold !== undefined ? this.meta.prefetchThreshold : 0)
    )));
    const outputValueWords = this._outputValueWords();
    const hasModelDownloadPriority = this._hasModelDownloadPriority();
    const visibleInstances = [];
    const visibleGlbs = new Set();
    const candidateByGlb = new Map();
    const prefetchCandidateByGlb = new Map();
    const allCandidateByGlb = new Map();
    for (let i = 0; i < candidateIds.length; i += 1) {
      const baseWord = i * outputValueWords;
      const decoded = this._decodePackedVisibility(raw[baseWord]);
      const instanceId = candidateIds[i];
      const visibilityScore = decoded.score;
      const downloadPriority = hasModelDownloadPriority
        ? this._decodeScoreWord(raw[baseWord + 1])
        : visibilityScore;
      const globalGlbId = this.instanceToGlobalGlb[instanceId];
      const globalInstanceId = this._localInstanceIdToGlobal(instanceId);
      if (globalGlbId >= 0) {
        let allCandidate = allCandidateByGlb.get(globalGlbId);
        if (!allCandidate) {
          allCandidate = {
            globalGlbId,
            confidence: downloadPriority,
            importance: downloadPriority,
            downloadPriority,
            visibilityScore,
            sourceComponentIds: [],
            sourceComponents: [],
          };
          allCandidateByGlb.set(globalGlbId, allCandidate);
        }
        allCandidate.confidence = Math.max(allCandidate.confidence, downloadPriority);
        allCandidate.importance = Math.max(allCandidate.importance, downloadPriority);
        allCandidate.downloadPriority = Math.max(allCandidate.downloadPriority, downloadPriority);
        allCandidate.visibilityScore = Math.max(allCandidate.visibilityScore, visibilityScore);
        allCandidate.sourceComponentIds.push(globalInstanceId);
        allCandidate.sourceComponents.push({ componentId: globalInstanceId, visibilityScore, downloadPriority, visible: decoded.visible });
      }
      if (returnAllScores && downloadPriority >= prefetchThreshold && !decoded.visible && globalGlbId >= 0) {
        let prefetchCandidate = prefetchCandidateByGlb.get(globalGlbId);
        if (!prefetchCandidate) {
          prefetchCandidate = {
            globalGlbId,
            confidence: downloadPriority,
            importance: downloadPriority,
            downloadPriority,
            visibilityScore,
            sourceComponentIds: [],
            sourceComponents: [],
            prefetchOnly: true,
          };
          prefetchCandidateByGlb.set(globalGlbId, prefetchCandidate);
        }
        prefetchCandidate.confidence = Math.max(prefetchCandidate.confidence, downloadPriority);
        prefetchCandidate.importance = Math.max(prefetchCandidate.importance, downloadPriority);
        prefetchCandidate.downloadPriority = Math.max(prefetchCandidate.downloadPriority, downloadPriority);
        prefetchCandidate.visibilityScore = Math.max(prefetchCandidate.visibilityScore, visibilityScore);
        prefetchCandidate.sourceComponentIds.push(globalInstanceId);
        prefetchCandidate.sourceComponents.push({ componentId: globalInstanceId, visibilityScore, downloadPriority, visible: false });
      }
      if (!decoded.visible) continue;
      visibleInstances.push(globalInstanceId);
      if (globalGlbId >= 0) {
        visibleGlbs.add(globalGlbId);
        let candidate = candidateByGlb.get(globalGlbId);
        if (!candidate) {
          candidate = {
            globalGlbId,
            confidence: downloadPriority,
            importance: downloadPriority,
            downloadPriority,
            visibilityScore,
            sourceComponentIds: [],
            sourceComponents: [],
          };
          candidateByGlb.set(globalGlbId, candidate);
        }
        candidate.confidence = Math.max(candidate.confidence, downloadPriority);
        candidate.importance = Math.max(candidate.importance, downloadPriority);
        candidate.downloadPriority = Math.max(candidate.downloadPriority, downloadPriority);
        candidate.visibilityScore = Math.max(candidate.visibilityScore, visibilityScore);
        candidate.sourceComponentIds.push(globalInstanceId);
        candidate.sourceComponents.push({ componentId: globalInstanceId, visibilityScore, downloadPriority, visible: true });
      }
    }
    const modelList = Array.from(visibleGlbs).sort((a, b) => a - b);
    const candidates = Array.from(candidateByGlb.values()).sort((a, b) => b.downloadPriority - a.downloadPriority || a.globalGlbId - b.globalGlbId);
    const prefetchCandidates = returnAllScores
      ? Array.from(prefetchCandidateByGlb.values()).sort((a, b) => b.downloadPriority - a.downloadPriority || a.globalGlbId - b.globalGlbId)
      : [];
    const allCandidates = returnAllScores
      ? Array.from(allCandidateByGlb.values()).sort((a, b) => b.downloadPriority - a.downloadPriority || a.globalGlbId - b.globalGlbId)
      : [];
    const end = performance.now();
    this.predictSerial += 1;
    this.lastPredictTimings = {
      serial: this.predictSerial,
      totalMs: end - start,
      candidateMs: candidateEnd - candidateStart,
      candidateSelection: this.lastCandidateSelection,
      inferenceMs: inferenceEnd - inferenceStart,
      postMs: end - postStart,
      candidateCount: candidateIds.length,
      visibleInstanceCount: visibleInstances.length,
      visibleGlbCount: modelList.length,
      hasModelDownloadPriority,
      backend: this.backend,
    };
    if (this.debugLogging && this.predictSerial <= 5) {
      console.log('[InstancePVS] predict timings', this.lastPredictTimings);
    }
    return {
      idMode: 'global-glb-priority',
      componentModelList: visibleInstances,
      modelList,
      weightList: modelList.map((globalGlbId) => candidateByGlb.get(globalGlbId)?.downloadPriority ?? 1),
      candidates,
      prefetchCandidates,
      allCandidates,
      candidateCount: candidateIds.length,
      visibleInstanceCount: visibleInstances.length,
      hasModelDownloadPriority,
      backend: this.backend,
      executionTime: end - start,
      timings: { ...this.lastPredictTimings },
    };
  }

  _buildShaderV3() {
    const meta = this.meta;
    const offsets = Object.fromEntries([...this.layout.entries()].map(([name, item]) => [name, item.floatOffset]));
    const featureDim = Number(meta.cameraHashFeatureDimPerLevel || 8);
    const hashDim = Number(meta.cameraHashOutputDim || (meta.cameraHashCellSizesM || []).length * featureDim);
    const viewDim = Number(meta.cameraViewFeatureDim || 45);
    const cameraFeatureDim = Number(meta.cameraFeatureDim || (hashDim + viewDim));
    const relativeDim = Number(meta.relativeFeatureDim || 15);
    const glbGeoDim = Number(meta.glbGeoDim || 64);
    const attentionDim = Number(meta.attentionDim ?? 64);
    const usesAttention = attentionDim > 0 && meta.usesOccluderAttention !== false && Number(meta.occluderK ?? 0) > 0;
    const attentionHeads = usesAttention ? Math.max(1, Number(meta.attentionHeads ?? 4)) : 1;
    const headDim = usesAttention ? Math.max(1, Math.floor(attentionDim / attentionHeads)) : 1;
    const hidden = Number(meta.mlpHiddenDim || 128);
    const inputDim = cameraFeatureDim + relativeDim + glbGeoDim + (usesAttention ? attentionDim : 0);
    const k = usesAttention ? Math.max(1, Number(meta.occluderK ?? 16)) : 1;
    const sceneSize = meta.sceneSizeM || this.sceneBounds.size;
    const levelLines = (meta.cameraHashCellSizesM || []).map((cell, level) => {
      const outOffset = level * featureDim;
      return `
  for (var c${level} = 0u; c${level} < ${featureDim}u; c${level} = c${level} + 1u) {
    fused[${outOffset}u + c${level}] = sample_camera_level(${level}u, ${offsets[`camera_hash_l${level}`]}u, ${meta.cameraHashTableSizes[level]}u, ${Number(cell).toFixed(8)}, cam, c${level});
  }`;
    }).join('\n');
    const viewCode = this._buildSphericalViewFeatureShader(hashDim, Number(meta.cameraViewFourierBands || 4));

    return `
struct Uniforms {
  camera_threshold: vec4<f32>,
  view_tanfov: vec4<f32>,
  count_pad: vec4<f32>,
  camera_world_pad: vec4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> candidate_ids: array<u32>;
@group(0) @binding(3) var<storage, read> neighbor_indices: array<u32>;
@group(0) @binding(4) var<storage, read> instance_to_glb: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_values: array<u32>;

const SCENE_X: f32 = ${Number(sceneSize[0]).toFixed(8)};
const SCENE_Y: f32 = ${Number(sceneSize[1]).toFixed(8)};
const SCENE_Z: f32 = ${Number(sceneSize[2]).toFixed(8)};
const FEATURE_DIM: u32 = ${featureDim}u;
const HASH_DIM: u32 = ${hashDim}u;
const CAMERA_FEATURE_DIM: u32 = ${cameraFeatureDim}u;
const RELATIVE_DIM: u32 = ${relativeDim}u;
const GLB_GEO_DIM: u32 = ${glbGeoDim}u;
const ATTN_DIM: u32 = ${attentionDim}u;
const ATTN_HEADS: u32 = ${attentionHeads}u;
const ATTN_HEAD_DIM: u32 = ${headDim}u;
const OCCLUDER_K: u32 = ${k}u;
const HIDDEN: u32 = ${hidden}u;
const INPUT_DIM: u32 = ${inputDim}u;
const NUM_INSTANCES: u32 = ${Number(meta.numInstances || 1)}u;
const NUM_GLBS: u32 = ${Number(meta.numGlbs || 1)}u;
const INVALID_INDEX: u32 = 4294967295u;
const OFFSET_AABBS: u32 = ${offsets.instance_world_aabbs}u;
const OFFSET_GLB_GEO: u32 = ${offsets.glb_geo_features}u;
${usesAttention ? `const OFFSET_ATTN_Q_W: u32 = ${offsets.attn_q_w}u;
const OFFSET_ATTN_Q_B: u32 = ${offsets.attn_q_b}u;
const OFFSET_ATTN_K_W: u32 = ${offsets.attn_k_w}u;
const OFFSET_ATTN_K_B: u32 = ${offsets.attn_k_b}u;
const OFFSET_ATTN_V_W: u32 = ${offsets.attn_v_w}u;
const OFFSET_ATTN_V_B: u32 = ${offsets.attn_v_b}u;
const OFFSET_ATTN_OUT_W: u32 = ${offsets.attn_out_w}u;
const OFFSET_ATTN_OUT_B: u32 = ${offsets.attn_out_b}u;` : ''}
const OFFSET_W0: u32 = ${offsets.mlp_vis_w0}u;
const OFFSET_B0: u32 = ${offsets.mlp_vis_b0}u;
const OFFSET_W1: u32 = ${offsets.mlp_vis_w1}u;
const OFFSET_B1: u32 = ${offsets.mlp_vis_b1}u;
const OFFSET_W2: u32 = ${offsets.mlp_vis_w2}u;
const OFFSET_B2: u32 = ${offsets.mlp_vis_b2}u;
const OUTPUT_VALUE_WORDS: u32 = ${this._outputValueWords()}u;

fn w(index: u32) -> f32 {
  let pair = unpack2x16float(weights[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn hash3(ix: u32, iy: u32, iz: u32, level: u32, table_size: u32) -> u32 {
  let h = (ix * 73856093u) ^ (iy * 19349663u) ^ (iz * 83492791u) ^ (level * 2654435761u);
  return h % table_size;
}

fn table_fetch(offset: u32, table_size: u32, feature: u32, ix: u32, iy: u32, iz: u32, level: u32) -> f32 {
  return w(offset + hash3(ix, iy, iz, level, table_size) * FEATURE_DIM + feature);
}

fn sample_camera_level(level: u32, offset: u32, table_size: u32, cell: f32, cam_in: vec3<f32>, feature: u32) -> f32 {
  let res = max(ceil(vec3<f32>(SCENE_X, SCENE_Y, SCENE_Z) / cell), vec3<f32>(1.0));
  let cam = clamp(cam_in, vec3<f32>(0.0), vec3<f32>(1.0)) * max(res - vec3<f32>(1.0), vec3<f32>(1.0));
  let x0 = u32(floor(cam.x));
  let y0 = u32(floor(cam.y));
  let z0 = u32(floor(cam.z));
  let x1 = min(x0 + 1u, u32(res.x) - 1u);
  let y1 = min(y0 + 1u, u32(res.y) - 1u);
  let z1 = min(z0 + 1u, u32(res.z) - 1u);
  let wx = cam.x - f32(x0);
  let wy = cam.y - f32(y0);
  let wz = cam.z - f32(z0);
  let c000 = table_fetch(offset, table_size, feature, x0, y0, z0, level);
  let c100 = table_fetch(offset, table_size, feature, x1, y0, z0, level);
  let c010 = table_fetch(offset, table_size, feature, x0, y1, z0, level);
  let c110 = table_fetch(offset, table_size, feature, x1, y1, z0, level);
  let c001 = table_fetch(offset, table_size, feature, x0, y0, z1, level);
  let c101 = table_fetch(offset, table_size, feature, x1, y0, z1, level);
  let c011 = table_fetch(offset, table_size, feature, x0, y1, z1, level);
  let c111 = table_fetch(offset, table_size, feature, x1, y1, z1, level);
  let c00 = c000 * (1.0 - wx) + c100 * wx;
  let c10 = c010 * (1.0 - wx) + c110 * wx;
  let c01 = c001 * (1.0 - wx) + c101 * wx;
  let c11 = c011 * (1.0 - wx) + c111 * wx;
  let c0 = c00 * (1.0 - wy) + c10 * wy;
  let c1 = c01 * (1.0 - wy) + c11 * wy;
  return c0 * (1.0 - wz) + c1 * wz;
}

fn safe_glb_id(inst_id: u32) -> u32 {
  let safe_inst = min(inst_id, NUM_INSTANCES - 1u);
  return min(instance_to_glb[safe_inst], NUM_GLBS - 1u);
}

fn compute_relative(inst_id: u32) -> array<f32, ${relativeDim}> {
  var rel: array<f32, ${relativeDim}>;
  let bounds_base = OFFSET_AABBS + inst_id * 6u;
  let bmin = vec3<f32>(w(bounds_base + 0u), w(bounds_base + 1u), w(bounds_base + 2u));
  let bmax = vec3<f32>(w(bounds_base + 3u), w(bounds_base + 4u), w(bounds_base + 5u));
  let center = (bmin + bmax) * 0.5;
  let size = max(bmax - bmin, vec3<f32>(0.0001));
  let camera_world = uniforms.camera_world_pad.xyz;
  let delta = center - camera_world;
  let scene_scale = max(vec3<f32>(SCENE_X, SCENE_Y, SCENE_Z), vec3<f32>(0.0001));
  let delta_norm = clamp(delta / scene_scale, vec3<f32>(-4.0), vec3<f32>(4.0));
  let dist = max(length(delta), 0.0001);
  let dir = delta / dist;
  let raw_forward = vec3<f32>(uniforms.view_tanfov.x, uniforms.view_tanfov.y, uniforms.view_tanfov.z);
  let forward = raw_forward / max(length(raw_forward), 0.000001);
  let up_ref_a = vec3<f32>(0.0, 1.0, 0.0);
  let up_ref_b = vec3<f32>(0.0, 0.0, 1.0);
  var up_seed = up_ref_a;
  if (abs(dot(forward, up_ref_a)) > 0.98) {
    up_seed = up_ref_b;
  }
  let right = normalize(cross(forward, up_seed));
  let up = normalize(cross(right, forward));
  let dot_forward = dot(dir, forward);
  let dot_right = dot(dir, right);
  let dot_up = dot(dir, up);
  let tan_x_rel = max(uniforms.view_tanfov.w, 0.0001);
  let tan_y_rel = max(uniforms.count_pad.x, 0.0001);
  let radius = length(size) * 0.5;
  let angular = radius / max(dist, 1.0);
  rel[0] = delta_norm.x;
  rel[1] = delta_norm.y;
  rel[2] = delta_norm.z;
  rel[3] = log(1.0 + dist / 100.0);
  rel[4] = dir.x;
  rel[5] = dir.y;
  rel[6] = dir.z;
  rel[7] = dot_forward;
  rel[8] = dot_right;
  rel[9] = dot_up;
  rel[10] = clamp(dot_right / max(dot_forward * tan_x_rel, 0.0001), -4.0, 4.0);
  rel[11] = clamp(dot_up / max(dot_forward * tan_y_rel, 0.0001), -4.0, 4.0);
  rel[12] = clamp(angular / tan_x_rel, 0.0, 4.0);
  rel[13] = clamp(angular / tan_y_rel, 0.0, 4.0);
  rel[14] = clamp(max(max(size.x / scene_scale.x, size.y / scene_scale.y), size.z / scene_scale.z), 0.0, 4.0);
  return rel;
}

${usesAttention ? `fn project_instance_feature(rel: array<f32, ${relativeDim}>, glb_id: u32, weight_offset: u32, bias_offset: u32, out_idx: u32) -> f32 {
  var sum = w(bias_offset + out_idx);
  for (var i = 0u; i < RELATIVE_DIM; i = i + 1u) {
    sum = sum + w(weight_offset + out_idx * (RELATIVE_DIM + GLB_GEO_DIM) + i) * rel[i];
  }
  for (var i = 0u; i < GLB_GEO_DIM; i = i + 1u) {
    sum = sum + w(weight_offset + out_idx * (RELATIVE_DIM + GLB_GEO_DIM) + RELATIVE_DIM + i) * w(OFFSET_GLB_GEO + glb_id * GLB_GEO_DIM + i);
  }
  return sum;
}` : ''}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = u32(uniforms.count_pad.y);
  if (idx >= count) { return; }
  let inst_id = candidate_ids[idx];
  let cam = uniforms.camera_threshold.xyz;
  let threshold = uniforms.camera_threshold.w;
  let glb_id = safe_glb_id(inst_id);
  let rel = compute_relative(inst_id);

  var fused: array<f32, ${inputDim}>;
${levelLines}
${viewCode}

  for (var i = 0u; i < RELATIVE_DIM; i = i + 1u) {
    fused[CAMERA_FEATURE_DIM + i] = rel[i];
  }
  for (var i = 0u; i < GLB_GEO_DIM; i = i + 1u) {
    fused[CAMERA_FEATURE_DIM + RELATIVE_DIM + i] = w(OFFSET_GLB_GEO + glb_id * GLB_GEO_DIM + i);
  }

${usesAttention ? `  var q: array<f32, ${attentionDim}>;
  for (var o = 0u; o < ATTN_DIM; o = o + 1u) {
    q[o] = project_instance_feature(rel, glb_id, OFFSET_ATTN_Q_W, OFFSET_ATTN_Q_B, o);
  }

  var context: array<f32, ${attentionDim}>;
  for (var i = 0u; i < ATTN_DIM; i = i + 1u) {
    context[i] = 0.0;
  }
  for (var h = 0u; h < ATTN_HEADS; h = h + 1u) {
    var logits: array<f32, ${k}>;
    var valid: array<u32, ${k}>;
    var max_logit = -10000000000.0;
    for (var n = 0u; n < OCCLUDER_K; n = n + 1u) {
      let local_index = neighbor_indices[idx * OCCLUDER_K + n];
      if (local_index == INVALID_INDEX || local_index >= count) {
        logits[n] = -10000000000.0;
        valid[n] = 0u;
      } else {
        let n_inst = candidate_ids[local_index];
        let n_rel = compute_relative(n_inst);
        let n_glb = safe_glb_id(n_inst);
        var dot_qk = 0.0;
        for (var d = 0u; d < ATTN_HEAD_DIM; d = d + 1u) {
          let out_idx = h * ATTN_HEAD_DIM + d;
          dot_qk = dot_qk + q[out_idx] * project_instance_feature(n_rel, n_glb, OFFSET_ATTN_K_W, OFFSET_ATTN_K_B, out_idx);
        }
        logits[n] = dot_qk / sqrt(f32(ATTN_HEAD_DIM));
        valid[n] = 1u;
        max_logit = max(max_logit, logits[n]);
      }
    }
    var denom = 0.0;
    for (var n = 0u; n < OCCLUDER_K; n = n + 1u) {
      if (valid[n] == 1u) {
        denom = denom + exp(logits[n] - max_logit);
      }
    }
    for (var d = 0u; d < ATTN_HEAD_DIM; d = d + 1u) {
      let out_idx = h * ATTN_HEAD_DIM + d;
      var value_sum = 0.0;
      if (denom > 0.0) {
        for (var n = 0u; n < OCCLUDER_K; n = n + 1u) {
          if (valid[n] == 1u) {
            let local_index = neighbor_indices[idx * OCCLUDER_K + n];
            let n_inst = candidate_ids[local_index];
            let n_rel = compute_relative(n_inst);
            let n_glb = safe_glb_id(n_inst);
            let attn_weight = exp(logits[n] - max_logit) / denom;
            value_sum = value_sum + attn_weight * project_instance_feature(n_rel, n_glb, OFFSET_ATTN_V_W, OFFSET_ATTN_V_B, out_idx);
          }
        }
      }
      context[out_idx] = value_sum;
    }
  }

  for (var o = 0u; o < ATTN_DIM; o = o + 1u) {
    var sum = w(OFFSET_ATTN_OUT_B + o);
    for (var i = 0u; i < ATTN_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_ATTN_OUT_W + o * ATTN_DIM + i) * context[i];
    }
    fused[CAMERA_FEATURE_DIM + RELATIVE_DIM + GLB_GEO_DIM + o] = sum;
  }` : ''}

  var h0: array<f32, ${hidden}>;
  for (var o = 0u; o < HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_B0 + o);
    for (var i = 0u; i < INPUT_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_W0 + o * INPUT_DIM + i) * fused[i];
    }
    h0[o] = max(sum, 0.0);
  }
  var h1: array<f32, ${hidden}>;
  for (var o = 0u; o < HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_B1 + o);
    for (var i = 0u; i < HIDDEN; i = i + 1u) {
      sum = sum + w(OFFSET_W1 + o * HIDDEN + i) * h0[i];
    }
    h1[o] = max(sum, 0.0);
  }
  var logit = w(OFFSET_B2);
  for (var i = 0u; i < HIDDEN; i = i + 1u) {
    logit = logit + w(OFFSET_W2 + i) * h1[i];
  }
  let p = 1.0 / (1.0 + exp(-logit));
  let q16 = u32(clamp(p, 0.0, 1.0) * 65535.0);
  let flag = select(0u, 1u, p >= threshold);
  output_values[idx * OUTPUT_VALUE_WORDS] = (q16 << 1u) | flag;
}
`;
  }

  _buildShaderDirectionalOcclusionProxy() {
    const meta = this.meta;
    const offsets = Object.fromEntries([...this.layout.entries()].map(([name, item]) => [name, item.floatOffset]));
    const runtimeDim = Number(meta.runtimeFeatureDim || 352);
    const geoDim = Number(meta.geoDim || 96);
    const contextDim = Number(meta.contextDim || 64);
    const proxyDim = Number(meta.proxyDim || 8);
    const directionBins = Number(meta.directionBins || 8);
    const depthShells = Number(meta.depthShells || 3);
    const proxyCells = directionBins * depthShells;
    const queryDim = Number(meta.queryFeatureDim || (geoDim + contextDim + proxyDim));
    const rayDirDim = Number(meta.rayDirectionFeatureDim || 63);
    const rayScalarDim = Number(meta.rayScalarFeatureDim || 54);
    const cameraLocationDim = Number(meta.cameraLocationFeatureDim || 0);
    const cameraDim = Number(meta.cameraFeatureDim || (rayDirDim + rayScalarDim));
    const rayBands = Number(meta.rayFourierBands || 10);
    const scalarBands = Number(meta.rayScalarFourierBands || 4);
    const gateHidden = Number(this.layout.get('ray_proxy_gate_b0')?.shape?.[0] || 80);
    const hidden = Number(this.layout.get('mlp_vis_b0')?.shape?.[0] || 128);
    const interactionDim = Number(this.layout.get('mlp_vis_cam_inter_b')?.shape?.[0] || 64);
    const utilityHidden = Number(this.layout.get('utility_b0')?.shape?.[0] || 96);
    const sceneSize = meta.sceneSizeM || this.sceneBounds.size;
    const visInputDim = cameraDim + queryDim + interactionDim;
    const inhibitionInputDim = queryDim + cameraDim;
    const utilityInputDim = queryDim + 1;
    const downloadInputDim = queryDim + 2;

    return `
struct Uniforms {
  camera_threshold: vec4<f32>,
  view_tanfov: vec4<f32>,
  count_pad: vec4<f32>,
  camera_world_pad: vec4<f32>,
};

struct QueryFeatures {
  cam: array<f32, ${cameraDim}>,
  ray_dir: array<f32, ${rayDirDim}>,
  query: array<f32, ${queryDim}>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> candidate_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_values: array<u32>;

const PI: f32 = 3.141592653589793;
const RUNTIME_DIM: u32 = ${runtimeDim}u;
const GEO_DIM: u32 = ${geoDim}u;
const CONTEXT_DIM: u32 = ${contextDim}u;
const PROXY_DIM: u32 = ${proxyDim}u;
const PROXY_CELLS: u32 = ${proxyCells}u;
const QUERY_DIM: u32 = ${queryDim}u;
const RAY_DIR_DIM: u32 = ${rayDirDim}u;
const RAY_SCALAR_DIM: u32 = ${rayScalarDim}u;
const CAMERA_LOCATION_DIM: u32 = ${cameraLocationDim}u;
const CAMERA_DIM: u32 = ${cameraDim}u;
const RAY_BANDS: u32 = ${rayBands}u;
const SCALAR_BANDS: u32 = ${scalarBands}u;
const GATE_HIDDEN: u32 = ${gateHidden}u;
const HIDDEN: u32 = ${hidden}u;
const INTERACTION_DIM: u32 = ${interactionDim}u;
const VIS_INPUT_DIM: u32 = ${visInputDim}u;
const INHIBITION_INPUT_DIM: u32 = ${inhibitionInputDim}u;
const UTILITY_HIDDEN: u32 = ${utilityHidden}u;
const UTILITY_INPUT_DIM: u32 = ${utilityInputDim}u;
const DOWNLOAD_INPUT_DIM: u32 = ${downloadInputDim}u;
const OUTPUT_VALUE_WORDS: u32 = ${this._outputValueWords()}u;
const NUM_INSTANCES: u32 = ${Number(meta.numInstances || 1)}u;
const SCENE_X: f32 = ${Number(sceneSize[0]).toFixed(8)};
const SCENE_Y: f32 = ${Number(sceneSize[1]).toFixed(8)};
const SCENE_Z: f32 = ${Number(sceneSize[2]).toFixed(8)};

const OFFSET_RUNTIME: u32 = ${offsets.runtime_features}u;
const OFFSET_AABBS: u32 = ${offsets.instance_world_aabbs}u;
const OFFSET_PROXY_GATE_W0: u32 = ${offsets.ray_proxy_gate_w0}u;
const OFFSET_PROXY_GATE_B0: u32 = ${offsets.ray_proxy_gate_b0}u;
const OFFSET_PROXY_GATE_W1: u32 = ${offsets.ray_proxy_gate_w1}u;
const OFFSET_PROXY_GATE_B1: u32 = ${offsets.ray_proxy_gate_b1}u;
const OFFSET_RUNTIME_GATE_W: u32 = ${offsets.ray_runtime_gate_w}u;
const OFFSET_RUNTIME_GATE_B: u32 = ${offsets.ray_runtime_gate_b}u;
const OFFSET_VIS_CAM_W: u32 = ${offsets.mlp_vis_cam_inter_w}u;
const OFFSET_VIS_CAM_B: u32 = ${offsets.mlp_vis_cam_inter_b}u;
const OFFSET_VIS_INST_W: u32 = ${offsets.mlp_vis_inst_inter_w}u;
const OFFSET_VIS_INST_B: u32 = ${offsets.mlp_vis_inst_inter_b}u;
const OFFSET_VIS_W0: u32 = ${offsets.mlp_vis_w0}u;
const OFFSET_VIS_B0: u32 = ${offsets.mlp_vis_b0}u;
const OFFSET_VIS_W1: u32 = ${offsets.mlp_vis_w1}u;
const OFFSET_VIS_B1: u32 = ${offsets.mlp_vis_b1}u;
const OFFSET_VIS_W2: u32 = ${offsets.mlp_vis_w2}u;
const OFFSET_VIS_B2: u32 = ${offsets.mlp_vis_b2}u;
const OFFSET_INHIB_W0: u32 = ${offsets.inhibition_w0}u;
const OFFSET_INHIB_B0: u32 = ${offsets.inhibition_b0}u;
const OFFSET_INHIB_W1: u32 = ${offsets.inhibition_w1}u;
const OFFSET_INHIB_B1: u32 = ${offsets.inhibition_b1}u;
const OFFSET_UTILITY_W0: u32 = ${offsets.utility_w0}u;
const OFFSET_UTILITY_B0: u32 = ${offsets.utility_b0}u;
const OFFSET_UTILITY_W1: u32 = ${offsets.utility_w1}u;
const OFFSET_UTILITY_B1: u32 = ${offsets.utility_b1}u;
const OFFSET_UTILITY_W2: u32 = ${offsets.utility_w2}u;
const OFFSET_UTILITY_B2: u32 = ${offsets.utility_b2}u;
const OFFSET_DOWNLOAD_W0: u32 = ${offsets.download_w0}u;
const OFFSET_DOWNLOAD_B0: u32 = ${offsets.download_b0}u;
const OFFSET_DOWNLOAD_W1: u32 = ${offsets.download_w1}u;
const OFFSET_DOWNLOAD_B1: u32 = ${offsets.download_b1}u;

fn w(index: u32) -> f32 {
  let pair = unpack2x16float(weights[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn softplus(x: f32) -> f32 {
  if (x > 20.0) { return x; }
  return log(1.0 + exp(x));
}

fn safe_inst_id(inst_id: u32) -> u32 {
  return min(inst_id, NUM_INSTANCES - 1u);
}

fn bounds_center(inst_id: u32) -> vec3<f32> {
  let base = OFFSET_AABBS + safe_inst_id(inst_id) * 6u;
  let bmin = vec3<f32>(w(base + 0u), w(base + 1u), w(base + 2u));
  let bmax = vec3<f32>(w(base + 3u), w(base + 4u), w(base + 5u));
  return (bmin + bmax) * 0.5;
}

fn bounds_size(inst_id: u32) -> vec3<f32> {
  let base = OFFSET_AABBS + safe_inst_id(inst_id) * 6u;
  let bmin = vec3<f32>(w(base + 0u), w(base + 1u), w(base + 2u));
  let bmax = vec3<f32>(w(base + 3u), w(base + 4u), w(base + 5u));
  return max(bmax - bmin, vec3<f32>(0.0001));
}

fn camera_basis() -> mat3x3<f32> {
  let raw_forward = vec3<f32>(uniforms.view_tanfov.x, uniforms.view_tanfov.y, uniforms.view_tanfov.z);
  let forward = raw_forward / max(length(raw_forward), 0.000001);
  let up_ref_a = vec3<f32>(0.0, 1.0, 0.0);
  let up_ref_b = vec3<f32>(0.0, 0.0, 1.0);
  var up_seed = up_ref_a;
  if (abs(dot(forward, up_ref_a)) > 0.98) {
    up_seed = up_ref_b;
  }
  let right = normalize(cross(forward, up_seed));
  let up = normalize(cross(right, forward));
  return mat3x3<f32>(forward, right, up);
}

fn make_query(inst_id: u32) -> QueryFeatures {
  var out: QueryFeatures;
  let center = bounds_center(inst_id);
  let size = bounds_size(inst_id);
  let delta = center - uniforms.camera_world_pad.xyz;
  let dist = max(length(delta), 0.0001);
  let ray_dir = delta / dist;
  var cursor = 0u;
  out.cam[cursor + 0u] = ray_dir.x;
  out.cam[cursor + 1u] = ray_dir.y;
  out.cam[cursor + 2u] = ray_dir.z;
  out.ray_dir[cursor + 0u] = ray_dir.x;
  out.ray_dir[cursor + 1u] = ray_dir.y;
  out.ray_dir[cursor + 2u] = ray_dir.z;
  cursor = cursor + 3u;
  for (var band = 0u; band < RAY_BANDS; band = band + 1u) {
    let freq = exp2(f32(band)) * PI;
    out.cam[cursor + 0u] = sin(ray_dir.x * freq);
    out.cam[cursor + 1u] = sin(ray_dir.y * freq);
    out.cam[cursor + 2u] = sin(ray_dir.z * freq);
    out.ray_dir[cursor + 0u] = out.cam[cursor + 0u];
    out.ray_dir[cursor + 1u] = out.cam[cursor + 1u];
    out.ray_dir[cursor + 2u] = out.cam[cursor + 2u];
    cursor = cursor + 3u;
    out.cam[cursor + 0u] = cos(ray_dir.x * freq);
    out.cam[cursor + 1u] = cos(ray_dir.y * freq);
    out.cam[cursor + 2u] = cos(ray_dir.z * freq);
    out.ray_dir[cursor + 0u] = out.cam[cursor + 0u];
    out.ray_dir[cursor + 1u] = out.cam[cursor + 1u];
    out.ray_dir[cursor + 2u] = out.cam[cursor + 2u];
    cursor = cursor + 3u;
  }

  let basis = camera_basis();
  let forward = basis[0];
  let right = basis[1];
  let up = basis[2];
  let dot_forward = dot(ray_dir, forward);
  let dot_right = dot(ray_dir, right);
  let dot_up = dot(ray_dir, up);
  let tan_x = max(uniforms.view_tanfov.w, 0.0001);
  let tan_y = max(uniforms.count_pad.x, 0.0001);
  let denom_x = max(abs(dot_forward) * tan_x, 0.0001);
  let denom_y = max(abs(dot_forward) * tan_y, 0.0001);
  let u = dot_right / denom_x;
  let v = dot_up / denom_y;
  let radius = length(size) * 0.5;
  let angular = radius / max(dist, 1.0);
  var scalars: array<f32, 6>;
  scalars[0] = clamp(log(1.0 + dist / 100.0) / 4.0, 0.0, 2.0) - 1.0;
  scalars[1] = clamp(dot_forward, -1.0, 1.0);
  scalars[2] = clamp(u, -4.0, 4.0) / 4.0;
  scalars[3] = clamp(v, -4.0, 4.0) / 4.0;
  scalars[4] = clamp(angular / tan_x, 0.0, 4.0) / 2.0 - 1.0;
  scalars[5] = clamp(angular / tan_y, 0.0, 4.0) / 2.0 - 1.0;
  for (var i = 0u; i < 6u; i = i + 1u) {
    out.cam[cursor + i] = scalars[i];
  }
  cursor = cursor + 6u;
  for (var band = 0u; band < SCALAR_BANDS; band = band + 1u) {
    let freq = exp2(f32(band)) * PI;
    for (var i = 0u; i < 6u; i = i + 1u) {
      out.cam[cursor + i] = sin(scalars[i] * freq);
    }
    cursor = cursor + 6u;
    for (var i = 0u; i < 6u; i = i + 1u) {
      out.cam[cursor + i] = cos(scalars[i] * freq);
    }
    cursor = cursor + 6u;
  }

  // Ray direction remains the primary query. The optional normalized scene
  // location only disambiguates distant regions in very large scenes.
  if (CAMERA_LOCATION_DIM > 0u) {
    let camera_location = clamp(uniforms.camera_threshold.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    if (CAMERA_LOCATION_DIM >= 1u) { out.cam[cursor + 0u] = camera_location.x; }
    if (CAMERA_LOCATION_DIM >= 2u) { out.cam[cursor + 1u] = camera_location.y; }
    if (CAMERA_LOCATION_DIM >= 3u) { out.cam[cursor + 2u] = camera_location.z; }
    cursor = cursor + CAMERA_LOCATION_DIM;
  }

  var gate_h0: array<f32, ${gateHidden}>;
  for (var o = 0u; o < GATE_HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_PROXY_GATE_B0 + o);
    for (var i = 0u; i < CAMERA_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_PROXY_GATE_W0 + o * CAMERA_DIM + i) * out.cam[i];
    }
    gate_h0[o] = max(sum, 0.0);
  }
  var gate_logits: array<f32, ${proxyCells}>;
  var max_gate = -10000000000.0;
  for (var o = 0u; o < PROXY_CELLS; o = o + 1u) {
    var sum = w(OFFSET_PROXY_GATE_B1 + o);
    for (var i = 0u; i < GATE_HIDDEN; i = i + 1u) {
      sum = sum + w(OFFSET_PROXY_GATE_W1 + o * GATE_HIDDEN + i) * gate_h0[i];
    }
    gate_logits[o] = sum;
    max_gate = max(max_gate, sum);
  }
  var denom = 0.0;
  for (var o = 0u; o < PROXY_CELLS; o = o + 1u) {
    denom = denom + exp(gate_logits[o] - max_gate);
  }

  let runtime_base = OFFSET_RUNTIME + safe_inst_id(inst_id) * RUNTIME_DIM;
  for (var i = 0u; i < GEO_DIM; i = i + 1u) {
    out.query[i] = w(runtime_base + i);
  }
  for (var i = 0u; i < CONTEXT_DIM; i = i + 1u) {
    out.query[GEO_DIM + i] = w(runtime_base + GEO_DIM + i);
  }
  for (var p = 0u; p < PROXY_DIM; p = p + 1u) {
    var selected = 0.0;
    if (denom > 0.0) {
      for (var cell = 0u; cell < PROXY_CELLS; cell = cell + 1u) {
        let soft = exp(gate_logits[cell] - max_gate) / denom;
        selected = selected + soft * w(runtime_base + GEO_DIM + CONTEXT_DIM + cell * PROXY_DIM + p);
      }
    }
    out.query[GEO_DIM + CONTEXT_DIM + p] = selected;
  }
  for (var o = 0u; o < QUERY_DIM; o = o + 1u) {
    var sum = w(OFFSET_RUNTIME_GATE_B + o);
    for (var i = 0u; i < RAY_DIR_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_RUNTIME_GATE_W + o * RAY_DIR_DIM + i) * out.ray_dir[i];
    }
    out.query[o] = out.query[o] * (1.0 + tanh(sum));
  }
  return out;
}

fn visibility_logit(query: QueryFeatures) -> f32 {
  var cam_i: array<f32, ${interactionDim}>;
  var inst_i: array<f32, ${interactionDim}>;
  for (var o = 0u; o < INTERACTION_DIM; o = o + 1u) {
    var cam_sum = w(OFFSET_VIS_CAM_B + o);
    for (var i = 0u; i < CAMERA_DIM; i = i + 1u) {
      cam_sum = cam_sum + w(OFFSET_VIS_CAM_W + o * CAMERA_DIM + i) * query.cam[i];
    }
    cam_i[o] = cam_sum;
    var inst_sum = w(OFFSET_VIS_INST_B + o);
    for (var i = 0u; i < QUERY_DIM; i = i + 1u) {
      inst_sum = inst_sum + w(OFFSET_VIS_INST_W + o * QUERY_DIM + i) * query.query[i];
    }
    inst_i[o] = inst_sum;
  }
  var h0: array<f32, ${hidden}>;
  for (var o = 0u; o < HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_VIS_B0 + o);
    for (var i = 0u; i < CAMERA_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_VIS_W0 + o * VIS_INPUT_DIM + i) * query.cam[i];
    }
    for (var i = 0u; i < QUERY_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_VIS_W0 + o * VIS_INPUT_DIM + CAMERA_DIM + i) * query.query[i];
    }
    for (var i = 0u; i < INTERACTION_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_VIS_W0 + o * VIS_INPUT_DIM + CAMERA_DIM + QUERY_DIM + i) * (cam_i[i] * inst_i[i]);
    }
    h0[o] = max(sum, 0.0);
  }
  var h1: array<f32, ${hidden}>;
  for (var o = 0u; o < HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_VIS_B1 + o);
    for (var i = 0u; i < HIDDEN; i = i + 1u) {
      sum = sum + w(OFFSET_VIS_W1 + o * HIDDEN + i) * h0[i];
    }
    h1[o] = max(sum, 0.0);
  }
  var logit = w(OFFSET_VIS_B2);
  for (var i = 0u; i < HIDDEN; i = i + 1u) {
    logit = logit + w(OFFSET_VIS_W2 + i) * h1[i];
  }
  return logit;
}

fn inhibition_value(query: QueryFeatures) -> f32 {
  var h0: array<f32, ${hidden}>;
  for (var o = 0u; o < HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_INHIB_B0 + o);
    for (var i = 0u; i < QUERY_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_INHIB_W0 + o * INHIBITION_INPUT_DIM + i) * query.query[i];
    }
    for (var i = 0u; i < CAMERA_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_INHIB_W0 + o * INHIBITION_INPUT_DIM + QUERY_DIM + i) * query.cam[i];
    }
    h0[o] = max(sum, 0.0);
  }
  var raw = w(OFFSET_INHIB_B1);
  for (var i = 0u; i < HIDDEN; i = i + 1u) {
    raw = raw + w(OFFSET_INHIB_W1 + i) * h0[i];
  }
  return clamp(softplus(raw), 0.0, 4.0);
}

fn utility_logit(query: QueryFeatures, final_prob: f32) -> f32 {
  var h0: array<f32, ${utilityHidden}>;
  for (var o = 0u; o < UTILITY_HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_UTILITY_B0 + o);
    for (var i = 0u; i < QUERY_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_UTILITY_W0 + o * UTILITY_INPUT_DIM + i) * query.query[i];
    }
    sum = sum + w(OFFSET_UTILITY_W0 + o * UTILITY_INPUT_DIM + QUERY_DIM) * final_prob;
    h0[o] = max(sum, 0.0);
  }
  var h1: array<f32, ${utilityHidden}>;
  for (var o = 0u; o < UTILITY_HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_UTILITY_B1 + o);
    for (var i = 0u; i < UTILITY_HIDDEN; i = i + 1u) {
      sum = sum + w(OFFSET_UTILITY_W1 + o * UTILITY_HIDDEN + i) * h0[i];
    }
    h1[o] = max(sum, 0.0);
  }
  var logit = w(OFFSET_UTILITY_B2);
  for (var i = 0u; i < UTILITY_HIDDEN; i = i + 1u) {
    logit = logit + w(OFFSET_UTILITY_W2 + i) * h1[i];
  }
  return logit;
}

fn download_logit(query: QueryFeatures, final_prob: f32, utility_prob: f32) -> f32 {
  var h0: array<f32, ${utilityHidden}>;
  for (var o = 0u; o < UTILITY_HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_DOWNLOAD_B0 + o);
    for (var i = 0u; i < QUERY_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_DOWNLOAD_W0 + o * DOWNLOAD_INPUT_DIM + i) * query.query[i];
    }
    sum = sum + w(OFFSET_DOWNLOAD_W0 + o * DOWNLOAD_INPUT_DIM + QUERY_DIM) * final_prob;
    sum = sum + w(OFFSET_DOWNLOAD_W0 + o * DOWNLOAD_INPUT_DIM + QUERY_DIM + 1u) * utility_prob;
    h0[o] = max(sum, 0.0);
  }
  var logit = w(OFFSET_DOWNLOAD_B1);
  for (var i = 0u; i < UTILITY_HIDDEN; i = i + 1u) {
    logit = logit + w(OFFSET_DOWNLOAD_W1 + i) * h0[i];
  }
  return logit;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = u32(uniforms.count_pad.y);
  if (idx >= count) { return; }
  let inst_id = safe_inst_id(candidate_ids[idx]);
  let threshold = uniforms.camera_threshold.w;
  let query = make_query(inst_id);
  let base_logit = visibility_logit(query);
  let final_logit = base_logit - inhibition_value(query);
  let final_prob = sigmoid(final_logit);
  let utility_prob = sigmoid(utility_logit(query, final_prob));
  let download_prob = sigmoid(download_logit(query, final_prob, utility_prob));
  let vis_q16 = u32(clamp(final_prob, 0.0, 1.0) * 65535.0);
  let dl_q16 = u32(clamp(download_prob, 0.0, 1.0) * 65535.0);
  let flag = select(0u, 1u, final_prob >= threshold);
  let out_base = idx * OUTPUT_VALUE_WORDS;
  output_values[out_base] = (vis_q16 << 1u) | flag;
  output_values[out_base + 1u] = dl_q16;
}
`;
  }

  _buildShaderUtilityDynamicPool() {
    const meta = this.meta;
    const offsets = Object.fromEntries([...this.layout.entries()].map(([name, item]) => [name, item.floatOffset]));
    const wordOffsets = Object.fromEntries([...this.layout.entries()].map(([name]) => [name, this._wordOffset(name)]));
    const runtimeDim = Number(meta.runtimeFeatureDim || 192);
    const cameraDim = Number(meta.cameraFeatureDim || 117);
    const rayDirDim = Number(meta.rayDirectionFeatureDim || 63);
    const rayScalarDim = Number(meta.rayScalarFeatureDim || 54);
    const rayBands = Number(meta.rayFourierBands || 10);
    const scalarBands = Number(meta.rayScalarFourierBands || 4);
    const hidden = Number(meta.mlpVisHiddenDim || 128);
    const interactionDim = Number(meta.visibilityInteractionDim || 64);
    const visInputDim = cameraDim + runtimeDim + interactionDim;
    const edgeHidden = Number(meta.edgeHiddenDim || 96);
    const edgeInputDim = runtimeDim * 2 + 6 + 2;
    const poolK = Number(meta.occlusionPoolK || 32);
    const edgeK = Number(meta.edgeK || 8);
    const edgeTop = Number(meta.edgeTopAggregate || 2);
    const utilityHidden = Number(meta.utilityHiddenDim || 96);
    const maxOcclusion = Number(meta.maxOcclusionResidual || 2.0);
    const sceneSize = meta.sceneSizeM || this.sceneBounds.size;

    return `
struct Uniforms {
  camera_threshold: vec4<f32>,
  view_tanfov: vec4<f32>,
  count_pad: vec4<f32>,
  camera_world_pad: vec4<f32>,
};

struct CameraQuery {
  cam: array<f32, ${cameraDim}>,
  gated: array<f32, ${runtimeDim}>,
};

struct EdgeRay {
  values: array<f32, 6>,
  score: f32,
};

struct EdgeChoice {
  source_id: u32,
  values: array<f32, 6>,
  score: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> candidate_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_values: array<u32>;

const PI: f32 = 3.141592653589793;
const RUNTIME_DIM: u32 = ${runtimeDim}u;
const CAMERA_DIM: u32 = ${cameraDim}u;
const RAY_DIR_DIM: u32 = ${rayDirDim}u;
const RAY_SCALAR_DIM: u32 = ${rayScalarDim}u;
const RAY_BANDS: u32 = ${rayBands}u;
const SCALAR_BANDS: u32 = ${scalarBands}u;
const HIDDEN: u32 = ${hidden}u;
const INTERACTION_DIM: u32 = ${interactionDim}u;
const VIS_INPUT_DIM: u32 = ${visInputDim}u;
const EDGE_HIDDEN: u32 = ${edgeHidden}u;
const EDGE_INPUT_DIM: u32 = ${edgeInputDim}u;
const POOL_K: u32 = ${poolK}u;
const EDGE_K: u32 = ${edgeK}u;
const EDGE_TOP: u32 = ${edgeTop}u;
const UTILITY_HIDDEN: u32 = ${utilityHidden}u;
const OUTPUT_VALUE_WORDS: u32 = ${this._outputValueWords()}u;
const NUM_INSTANCES: u32 = ${Number(meta.numInstances || 1)}u;
const NUM_GLBS: u32 = ${Number(meta.numGlbs || 1)}u;
const MAX_OCCLUSION: f32 = ${maxOcclusion.toFixed(8)};
const SCENE_X: f32 = ${Number(sceneSize[0]).toFixed(8)};
const SCENE_Y: f32 = ${Number(sceneSize[1]).toFixed(8)};
const SCENE_Z: f32 = ${Number(sceneSize[2]).toFixed(8)};

const OFFSET_RUNTIME: u32 = ${offsets.runtime_features}u;
const OFFSET_AABBS: u32 = ${offsets.instance_world_aabbs}u;
const OFFSET_POOL_WORD: u32 = ${wordOffsets.occlusion_pool_ids}u;
const OFFSET_GATE_W: u32 = ${offsets.ray_gate_w}u;
const OFFSET_GATE_B: u32 = ${offsets.ray_gate_b}u;
const OFFSET_VIS_CAM_W: u32 = ${offsets.mlp_vis_cam_inter_w}u;
const OFFSET_VIS_CAM_B: u32 = ${offsets.mlp_vis_cam_inter_b}u;
const OFFSET_VIS_INST_W: u32 = ${offsets.mlp_vis_inst_inter_w}u;
const OFFSET_VIS_INST_B: u32 = ${offsets.mlp_vis_inst_inter_b}u;
const OFFSET_VIS_W0: u32 = ${offsets.mlp_vis_w0}u;
const OFFSET_VIS_B0: u32 = ${offsets.mlp_vis_b0}u;
const OFFSET_VIS_W1: u32 = ${offsets.mlp_vis_w1}u;
const OFFSET_VIS_B1: u32 = ${offsets.mlp_vis_b1}u;
const OFFSET_VIS_W2: u32 = ${offsets.mlp_vis_w2}u;
const OFFSET_VIS_B2: u32 = ${offsets.mlp_vis_b2}u;
const OFFSET_EDGE_W0: u32 = ${offsets.edge_shared_w0}u;
const OFFSET_EDGE_B0: u32 = ${offsets.edge_shared_b0}u;
const OFFSET_EDGE_W1: u32 = ${offsets.edge_shared_w1}u;
const OFFSET_EDGE_B1: u32 = ${offsets.edge_shared_b1}u;
const OFFSET_EDGE_OCC_W: u32 = ${offsets.edge_occ_w}u;
const OFFSET_EDGE_OCC_B: u32 = ${offsets.edge_occ_b}u;
const OFFSET_EDGE_SUPP_W: u32 = ${offsets.edge_supp_w}u;
const OFFSET_EDGE_SUPP_B: u32 = ${offsets.edge_supp_b}u;
const OFFSET_UTILITY_W0: u32 = ${offsets.utility_w0}u;
const OFFSET_UTILITY_B0: u32 = ${offsets.utility_b0}u;
const OFFSET_UTILITY_W1: u32 = ${offsets.utility_w1}u;
const OFFSET_UTILITY_B1: u32 = ${offsets.utility_b1}u;
const OFFSET_UTILITY_W2: u32 = ${offsets.utility_w2}u;
const OFFSET_UTILITY_B2: u32 = ${offsets.utility_b2}u;
const OFFSET_DOWNLOAD_W0: u32 = ${offsets.download_w0}u;
const OFFSET_DOWNLOAD_B0: u32 = ${offsets.download_b0}u;
const OFFSET_DOWNLOAD_W1: u32 = ${offsets.download_w1}u;
const OFFSET_DOWNLOAD_B1: u32 = ${offsets.download_b1}u;

fn w(index: u32) -> f32 {
  let pair = unpack2x16float(weights[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn softplus(x: f32) -> f32 {
  if (x > 20.0) { return x; }
  return log(1.0 + exp(x));
}

fn safe_inst_id(inst_id: u32) -> u32 {
  return min(inst_id, NUM_INSTANCES - 1u);
}

fn pool_id(inst_id: u32, pool_index: u32) -> u32 {
  return safe_inst_id(weights[OFFSET_POOL_WORD + safe_inst_id(inst_id) * POOL_K + pool_index]);
}

fn load_runtime(inst_id: u32) -> array<f32, ${runtimeDim}> {
  var out: array<f32, ${runtimeDim}>;
  let base = OFFSET_RUNTIME + safe_inst_id(inst_id) * RUNTIME_DIM;
  for (var i = 0u; i < RUNTIME_DIM; i = i + 1u) {
    out[i] = w(base + i);
  }
  return out;
}

fn bounds_center(inst_id: u32) -> vec3<f32> {
  let base = OFFSET_AABBS + safe_inst_id(inst_id) * 6u;
  let bmin = vec3<f32>(w(base + 0u), w(base + 1u), w(base + 2u));
  let bmax = vec3<f32>(w(base + 3u), w(base + 4u), w(base + 5u));
  return (bmin + bmax) * 0.5;
}

fn bounds_size(inst_id: u32) -> vec3<f32> {
  let base = OFFSET_AABBS + safe_inst_id(inst_id) * 6u;
  let bmin = vec3<f32>(w(base + 0u), w(base + 1u), w(base + 2u));
  let bmax = vec3<f32>(w(base + 3u), w(base + 4u), w(base + 5u));
  return max(bmax - bmin, vec3<f32>(0.0001));
}

fn camera_basis() -> mat3x3<f32> {
  let raw_forward = vec3<f32>(uniforms.view_tanfov.x, uniforms.view_tanfov.y, uniforms.view_tanfov.z);
  let forward = raw_forward / max(length(raw_forward), 0.000001);
  let up_ref_a = vec3<f32>(0.0, 1.0, 0.0);
  let up_ref_b = vec3<f32>(0.0, 0.0, 1.0);
  var up_seed = up_ref_a;
  if (abs(dot(forward, up_ref_a)) > 0.98) {
    up_seed = up_ref_b;
  }
  let right = normalize(cross(forward, up_seed));
  let up = normalize(cross(right, forward));
  return mat3x3<f32>(forward, right, up);
}

fn make_camera_query(inst_id: u32, raw_feat: array<f32, ${runtimeDim}>) -> CameraQuery {
  var out: CameraQuery;
  let center = bounds_center(inst_id);
  let size = bounds_size(inst_id);
  let delta = center - uniforms.camera_world_pad.xyz;
  let dist = max(length(delta), 0.0001);
  let ray_dir = delta / dist;
  var cursor = 0u;
  out.cam[cursor + 0u] = ray_dir.x;
  out.cam[cursor + 1u] = ray_dir.y;
  out.cam[cursor + 2u] = ray_dir.z;
  cursor = cursor + 3u;
  for (var band = 0u; band < RAY_BANDS; band = band + 1u) {
    let freq = exp2(f32(band)) * PI;
    out.cam[cursor + 0u] = sin(ray_dir.x * freq);
    out.cam[cursor + 1u] = sin(ray_dir.y * freq);
    out.cam[cursor + 2u] = sin(ray_dir.z * freq);
    cursor = cursor + 3u;
    out.cam[cursor + 0u] = cos(ray_dir.x * freq);
    out.cam[cursor + 1u] = cos(ray_dir.y * freq);
    out.cam[cursor + 2u] = cos(ray_dir.z * freq);
    cursor = cursor + 3u;
  }

  let basis = camera_basis();
  let forward = basis[0];
  let right = basis[1];
  let up = basis[2];
  let dot_forward = dot(ray_dir, forward);
  let dot_right = dot(ray_dir, right);
  let dot_up = dot(ray_dir, up);
  let tan_x = max(uniforms.view_tanfov.w, 0.0001);
  let tan_y = max(uniforms.count_pad.x, 0.0001);
  let denom_x = max(abs(dot_forward) * tan_x, 0.0001);
  let denom_y = max(abs(dot_forward) * tan_y, 0.0001);
  let u = dot_right / denom_x;
  let v = dot_up / denom_y;
  let radius = length(size) * 0.5;
  let angular = radius / max(dist, 1.0);
  var scalars: array<f32, 6>;
  scalars[0] = clamp(log(1.0 + dist / 100.0) / 4.0, 0.0, 2.0) - 1.0;
  scalars[1] = clamp(dot_forward, -1.0, 1.0);
  scalars[2] = clamp(u, -4.0, 4.0) / 4.0;
  scalars[3] = clamp(v, -4.0, 4.0) / 4.0;
  scalars[4] = clamp(angular / tan_x, 0.0, 4.0) / 2.0 - 1.0;
  scalars[5] = clamp(angular / tan_y, 0.0, 4.0) / 2.0 - 1.0;
  for (var i = 0u; i < 6u; i = i + 1u) {
    out.cam[cursor + i] = scalars[i];
  }
  cursor = cursor + 6u;
  for (var band = 0u; band < SCALAR_BANDS; band = band + 1u) {
    let freq = exp2(f32(band)) * PI;
    for (var i = 0u; i < 6u; i = i + 1u) {
      out.cam[cursor + i] = sin(scalars[i] * freq);
    }
    cursor = cursor + 6u;
    for (var i = 0u; i < 6u; i = i + 1u) {
      out.cam[cursor + i] = cos(scalars[i] * freq);
    }
    cursor = cursor + 6u;
  }

  for (var o = 0u; o < RUNTIME_DIM; o = o + 1u) {
    var sum = w(OFFSET_GATE_B + o);
    for (var i = 0u; i < RAY_DIR_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_GATE_W + o * RAY_DIR_DIM + i) * out.cam[i];
    }
    out.gated[o] = raw_feat[o] * (1.0 + tanh(sum));
  }
  return out;
}

fn visibility_logit_from_feat(inst_id: u32, raw_feat: array<f32, ${runtimeDim}>) -> f32 {
  let query = make_camera_query(inst_id, raw_feat);
  var cam_i: array<f32, ${interactionDim}>;
  var inst_i: array<f32, ${interactionDim}>;
  for (var o = 0u; o < INTERACTION_DIM; o = o + 1u) {
    var cam_sum = w(OFFSET_VIS_CAM_B + o);
    for (var i = 0u; i < CAMERA_DIM; i = i + 1u) {
      cam_sum = cam_sum + w(OFFSET_VIS_CAM_W + o * CAMERA_DIM + i) * query.cam[i];
    }
    cam_i[o] = cam_sum;
    var inst_sum = w(OFFSET_VIS_INST_B + o);
    for (var i = 0u; i < RUNTIME_DIM; i = i + 1u) {
      inst_sum = inst_sum + w(OFFSET_VIS_INST_W + o * RUNTIME_DIM + i) * query.gated[i];
    }
    inst_i[o] = inst_sum;
  }

  var h0: array<f32, ${hidden}>;
  for (var o = 0u; o < HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_VIS_B0 + o);
    for (var i = 0u; i < CAMERA_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_VIS_W0 + o * VIS_INPUT_DIM + i) * query.cam[i];
    }
    for (var i = 0u; i < RUNTIME_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_VIS_W0 + o * VIS_INPUT_DIM + CAMERA_DIM + i) * query.gated[i];
    }
    for (var i = 0u; i < INTERACTION_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_VIS_W0 + o * VIS_INPUT_DIM + CAMERA_DIM + RUNTIME_DIM + i) * (cam_i[i] * inst_i[i]);
    }
    h0[o] = max(sum, 0.0);
  }
  var h1: array<f32, ${hidden}>;
  for (var o = 0u; o < HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_VIS_B1 + o);
    for (var i = 0u; i < HIDDEN; i = i + 1u) {
      sum = sum + w(OFFSET_VIS_W1 + o * HIDDEN + i) * h0[i];
    }
    h1[o] = max(sum, 0.0);
  }
  var logit = w(OFFSET_VIS_B2);
  for (var i = 0u; i < HIDDEN; i = i + 1u) {
    logit = logit + w(OFFSET_VIS_W2 + i) * h1[i];
  }
  return logit;
}

fn make_edge_ray(target_id: u32, source_id: u32) -> EdgeRay {
  var out: EdgeRay;
  let ct = bounds_center(target_id);
  let cs = bounds_center(source_id);
  let rt = length(bounds_size(target_id)) * 0.5;
  let rs = length(bounds_size(source_id)) * 0.5;
  let basis = camera_basis();
  let forward = basis[0];
  let rel_t = ct - uniforms.camera_world_pad.xyz;
  let rel_s = cs - uniforms.camera_world_pad.xyz;
  let depth_t = dot(rel_t, forward);
  let depth_s = dot(rel_s, forward);
  let gap_raw = depth_t - depth_s;
  let diff = rel_s - rel_t;
  let lateral_vec = diff - dot(diff, forward) * forward;
  let lateral_raw = length(lateral_vec);
  let overlap_raw = (rs + rt - lateral_raw) / max(rs + rt, 1.0);
  out.values[0] = select(0.0, 1.0, gap_raw > 0.0);
  out.values[1] = clamp(gap_raw / 500.0, -4.0, 4.0);
  out.values[2] = clamp(lateral_raw / 500.0, 0.0, 4.0);
  out.values[3] = clamp(overlap_raw, -2.0, 2.0);
  out.values[4] = clamp(rs / max(rt, 1.0), 0.0, 8.0) / 8.0;
  out.values[5] = clamp(depth_t / 1000.0, -4.0, 4.0);
  out.score = out.values[0] * 8.0 + out.values[3] * 3.0 + out.values[4] * 0.5 - out.values[2] * 0.35 - abs(out.values[1]) * 0.05;
  return out;
}

fn empty_choice() -> EdgeChoice {
  var c: EdgeChoice;
  c.source_id = 0u;
  c.score = -1000000000.0;
  for (var i = 0u; i < 6u; i = i + 1u) {
    c.values[i] = 0.0;
  }
  return c;
}

fn edge_suppression(target_feat: array<f32, ${runtimeDim}>, source_feat: array<f32, ${runtimeDim}>, ray: array<f32, 6>, target_prob: f32, source_prob: f32) -> f32 {
  var h0: array<f32, ${edgeHidden}>;
  for (var o = 0u; o < EDGE_HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_EDGE_B0 + o);
    for (var i = 0u; i < RUNTIME_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_EDGE_W0 + o * EDGE_INPUT_DIM + i) * target_feat[i];
    }
    for (var i = 0u; i < RUNTIME_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_EDGE_W0 + o * EDGE_INPUT_DIM + RUNTIME_DIM + i) * source_feat[i];
    }
    for (var i = 0u; i < 6u; i = i + 1u) {
      sum = sum + w(OFFSET_EDGE_W0 + o * EDGE_INPUT_DIM + RUNTIME_DIM * 2u + i) * ray[i];
    }
    sum = sum + w(OFFSET_EDGE_W0 + o * EDGE_INPUT_DIM + RUNTIME_DIM * 2u + 6u) * target_prob;
    sum = sum + w(OFFSET_EDGE_W0 + o * EDGE_INPUT_DIM + RUNTIME_DIM * 2u + 7u) * source_prob;
    h0[o] = max(sum, 0.0);
  }
  var h1: array<f32, ${edgeHidden}>;
  for (var o = 0u; o < EDGE_HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_EDGE_B1 + o);
    for (var i = 0u; i < EDGE_HIDDEN; i = i + 1u) {
      sum = sum + w(OFFSET_EDGE_W1 + o * EDGE_HIDDEN + i) * h0[i];
    }
    h1[o] = max(sum, 0.0);
  }
  var occ = w(OFFSET_EDGE_OCC_B);
  var supp = w(OFFSET_EDGE_SUPP_B);
  for (var i = 0u; i < EDGE_HIDDEN; i = i + 1u) {
    occ = occ + w(OFFSET_EDGE_OCC_W + i) * h1[i];
    supp = supp + w(OFFSET_EDGE_SUPP_W + i) * h1[i];
  }
  return softplus(supp) * sigmoid(occ) * ray[0];
}

fn utility_logit(raw_feat: array<f32, ${runtimeDim}>, final_prob: f32) -> f32 {
  var h0: array<f32, ${utilityHidden}>;
  for (var o = 0u; o < UTILITY_HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_UTILITY_B0 + o);
    for (var i = 0u; i < RUNTIME_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_UTILITY_W0 + o * (RUNTIME_DIM + 1u) + i) * raw_feat[i];
    }
    sum = sum + w(OFFSET_UTILITY_W0 + o * (RUNTIME_DIM + 1u) + RUNTIME_DIM) * final_prob;
    h0[o] = max(sum, 0.0);
  }
  var h1: array<f32, ${utilityHidden}>;
  for (var o = 0u; o < UTILITY_HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_UTILITY_B1 + o);
    for (var i = 0u; i < UTILITY_HIDDEN; i = i + 1u) {
      sum = sum + w(OFFSET_UTILITY_W1 + o * UTILITY_HIDDEN + i) * h0[i];
    }
    h1[o] = max(sum, 0.0);
  }
  var logit = w(OFFSET_UTILITY_B2);
  for (var i = 0u; i < UTILITY_HIDDEN; i = i + 1u) {
    logit = logit + w(OFFSET_UTILITY_W2 + i) * h1[i];
  }
  return logit;
}

fn download_logit(raw_feat: array<f32, ${runtimeDim}>, final_prob: f32, utility_prob: f32) -> f32 {
  var h0: array<f32, ${utilityHidden}>;
  for (var o = 0u; o < UTILITY_HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_DOWNLOAD_B0 + o);
    for (var i = 0u; i < RUNTIME_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_DOWNLOAD_W0 + o * (RUNTIME_DIM + 2u) + i) * raw_feat[i];
    }
    sum = sum + w(OFFSET_DOWNLOAD_W0 + o * (RUNTIME_DIM + 2u) + RUNTIME_DIM) * final_prob;
    sum = sum + w(OFFSET_DOWNLOAD_W0 + o * (RUNTIME_DIM + 2u) + RUNTIME_DIM + 1u) * utility_prob;
    h0[o] = max(sum, 0.0);
  }
  var logit = w(OFFSET_DOWNLOAD_B1);
  for (var i = 0u; i < UTILITY_HIDDEN; i = i + 1u) {
    logit = logit + w(OFFSET_DOWNLOAD_W1 + i) * h0[i];
  }
  return logit;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = u32(uniforms.count_pad.y);
  if (idx >= count) { return; }
  let inst_id = safe_inst_id(candidate_ids[idx]);
  let threshold = uniforms.camera_threshold.w;
  let target_feat = load_runtime(inst_id);
  let prior_logit = visibility_logit_from_feat(inst_id, target_feat);
  let prior_prob = sigmoid(prior_logit);

  var chosen: array<EdgeChoice, ${edgeK}>;
  for (var i = 0u; i < EDGE_K; i = i + 1u) {
    chosen[i] = empty_choice();
  }
  for (var p = 0u; p < POOL_K; p = p + 1u) {
    let source_id = pool_id(inst_id, p);
    let ray = make_edge_ray(inst_id, source_id);
    var insert_at = EDGE_K;
    for (var slot = 0u; slot < EDGE_K; slot = slot + 1u) {
      if (insert_at == EDGE_K && ray.score > chosen[slot].score) {
        insert_at = slot;
      }
    }
    if (insert_at < EDGE_K) {
      var j = EDGE_K - 1u;
      loop {
        if (j <= insert_at) { break; }
        chosen[j] = chosen[j - 1u];
        j = j - 1u;
      }
      chosen[insert_at].source_id = source_id;
      chosen[insert_at].score = ray.score;
      for (var r = 0u; r < 6u; r = r + 1u) {
        chosen[insert_at].values[r] = ray.values[r];
      }
    }
  }

  var top1 = 0.0;
  var top2 = 0.0;
  for (var e = 0u; e < EDGE_K; e = e + 1u) {
    let source_feat = load_runtime(chosen[e].source_id);
    let source_prob = sigmoid(visibility_logit_from_feat(chosen[e].source_id, source_feat));
    let s = edge_suppression(target_feat, source_feat, chosen[e].values, prior_prob, source_prob);
    if (s > top1) {
      top2 = top1;
      top1 = s;
    } else if (s > top2) {
      top2 = s;
    }
  }
  var inhibition = top1;
  if (EDGE_TOP > 1u) {
    inhibition = inhibition + top2;
  }
  inhibition = clamp(inhibition, 0.0, MAX_OCCLUSION);
  let final_logit = prior_logit - inhibition;
  let final_prob = sigmoid(final_logit);
  let util_prob = sigmoid(utility_logit(target_feat, final_prob));
  let download_prob = sigmoid(download_logit(target_feat, final_prob, util_prob));
  let vis_q16 = u32(clamp(final_prob, 0.0, 1.0) * 65535.0);
  let dl_q16 = u32(clamp(download_prob, 0.0, 1.0) * 65535.0);
  let flag = select(0u, 1u, final_prob >= threshold);
  let out_base = idx * OUTPUT_VALUE_WORDS;
  output_values[out_base] = (vis_q16 << 1u) | flag;
  output_values[out_base + 1u] = dl_q16;
}
`;
  }

  _buildShader() {
    if (this._isDirectionalOcclusionProxyRuntime()) return this._buildShaderDirectionalOcclusionProxy();
    if (this._isUtilityDynamicPoolRuntime()) return this._buildShaderUtilityDynamicPool();
    if (this._isV3Runtime()) return this._buildShaderV3();
    const meta = this.meta;
    const offsets = Object.fromEntries([...this.layout.entries()].map(([name, item]) => [name, item.floatOffset]));
    const hidden = meta.mlpVisHiddenDim;
    const featureDim = meta.cameraHashFeatureDimPerLevel;
    const interactionDim = Number(meta.visibilityInteractionDim || 0);
    const inputDim = meta.cameraFeatureDim + meta.latentDim + meta.geoDim + interactionDim;
    const hashDim = Number(meta.cameraHashOutputDim || meta.cameraFeatureDim);
    const viewBands = Number(meta.cameraViewFourierBands || 0);
    const viewDim = Number(meta.cameraViewFeatureDim || 0);
    const relativeDim = Number(meta.relativeFeatureDim || 0);
    const sceneSize = this.cameraBounds.size;
    const worldScale = meta.sceneSizeM || this.sceneBounds.size;
    const viewCode = viewDim > 0 ? this._buildViewFeatureShader(hashDim, viewBands) : '';
    const relativeCode = relativeDim > 0 ? this._buildRelativeFeatureShader(hashDim + viewDim, offsets.instance_world_aabbs, worldScale) : '';
    const interactionCode = interactionDim > 0 ? `
  for (var o = 0u; o < ${interactionDim}u; o = o + 1u) {
    var cam_sum = w(${offsets.mlp_vis_cam_inter_b}u + o);
    for (var i = 0u; i < CAMERA_DIM; i = i + 1u) {
      cam_sum = cam_sum + w(${offsets.mlp_vis_cam_inter_w}u + o * CAMERA_DIM + i) * fused[i];
    }
    var inst_sum = w(${offsets.mlp_vis_inst_inter_b}u + o);
    for (var i = 0u; i < ${meta.latentDim + meta.geoDim}u; i = i + 1u) {
      inst_sum = inst_sum + w(${offsets.mlp_vis_inst_inter_w}u + o * ${meta.latentDim + meta.geoDim}u + i) * fused[CAMERA_DIM + i];
    }
    fused[CAMERA_DIM + ${meta.latentDim + meta.geoDim}u + o] = cam_sum * inst_sum;
  }` : '';
    const levelLines = (meta.cameraHashCellSizesM || []).map((cell, level) => {
      const outOffset = level * featureDim;
      return `
  for (var c${level} = 0u; c${level} < ${featureDim}u; c${level} = c${level} + 1u) {
    fused[${outOffset}u + c${level}] = sample_camera_level(${level}u, ${offsets[`camera_hash_l${level}`]}u, ${meta.cameraHashTableSizes[level]}u, ${Number(cell).toFixed(8)}, cam, c${level});
  }`;
    }).join('\n');

    return `
struct Uniforms {
  camera_threshold: vec4<f32>,
  view_tanfov: vec4<f32>,
  count_pad: vec4<f32>,
  camera_world_pad: vec4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> candidate_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<u32>;

const SCENE_X: f32 = ${Number(sceneSize[0]).toFixed(8)};
const SCENE_Y: f32 = ${Number(sceneSize[1]).toFixed(8)};
const SCENE_Z: f32 = ${Number(sceneSize[2]).toFixed(8)};
const FEATURE_DIM: u32 = ${featureDim}u;
const HIDDEN: u32 = ${hidden}u;
const INPUT_DIM: u32 = ${inputDim}u;
const CAMERA_DIM: u32 = ${meta.cameraFeatureDim}u;
const CAMERA_HASH_DIM: u32 = ${hashDim}u;
const CAMERA_VIEW_DIM: u32 = ${viewDim}u;
const RELATIVE_DIM: u32 = ${relativeDim}u;
const OFFSET_W0: u32 = ${offsets.mlp_vis_w0}u;
const OFFSET_B0: u32 = ${offsets.mlp_vis_b0}u;
const OFFSET_W1: u32 = ${offsets.mlp_vis_w1}u;
const OFFSET_B1: u32 = ${offsets.mlp_vis_b1}u;
const OFFSET_W2: u32 = ${offsets.mlp_vis_w2}u;
const OFFSET_B2: u32 = ${offsets.mlp_vis_b2}u;
const OUTPUT_VALUE_WORDS: u32 = ${this._outputValueWords()}u;
const OFFSET_LATENT: u32 = ${offsets.latent_codes}u;
const OFFSET_GEO: u32 = ${offsets.geo_summaries}u;

fn w(index: u32) -> f32 {
  let pair = unpack2x16float(weights[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn hash3(ix: u32, iy: u32, iz: u32, level: u32, table_size: u32) -> u32 {
  let h = (ix * 73856093u) ^ (iy * 19349663u) ^ (iz * 83492791u) ^ (level * 2654435761u);
  return h % table_size;
}

fn table_fetch(offset: u32, table_size: u32, feature: u32, ix: u32, iy: u32, iz: u32, level: u32) -> f32 {
  return w(offset + hash3(ix, iy, iz, level, table_size) * FEATURE_DIM + feature);
}

fn sample_camera_level(level: u32, offset: u32, table_size: u32, cell: f32, cam_in: vec3<f32>, feature: u32) -> f32 {
  let res = max(ceil(vec3<f32>(SCENE_X, SCENE_Y, SCENE_Z) / cell), vec3<f32>(1.0));
  let cam = clamp(cam_in, vec3<f32>(0.0), vec3<f32>(1.0)) * max(res - vec3<f32>(1.0), vec3<f32>(1.0));
  let x0 = u32(floor(cam.x));
  let y0 = u32(floor(cam.y));
  let z0 = u32(floor(cam.z));
  let x1 = min(x0 + 1u, u32(res.x) - 1u);
  let y1 = min(y0 + 1u, u32(res.y) - 1u);
  let z1 = min(z0 + 1u, u32(res.z) - 1u);
  let wx = cam.x - f32(x0);
  let wy = cam.y - f32(y0);
  let wz = cam.z - f32(z0);
  let c000 = table_fetch(offset, table_size, feature, x0, y0, z0, level);
  let c100 = table_fetch(offset, table_size, feature, x1, y0, z0, level);
  let c010 = table_fetch(offset, table_size, feature, x0, y1, z0, level);
  let c110 = table_fetch(offset, table_size, feature, x1, y1, z0, level);
  let c001 = table_fetch(offset, table_size, feature, x0, y0, z1, level);
  let c101 = table_fetch(offset, table_size, feature, x1, y0, z1, level);
  let c011 = table_fetch(offset, table_size, feature, x0, y1, z1, level);
  let c111 = table_fetch(offset, table_size, feature, x1, y1, z1, level);
  let c00 = c000 * (1.0 - wx) + c100 * wx;
  let c10 = c010 * (1.0 - wx) + c110 * wx;
  let c01 = c001 * (1.0 - wx) + c101 * wx;
  let c11 = c011 * (1.0 - wx) + c111 * wx;
  let c0 = c00 * (1.0 - wy) + c10 * wy;
  let c1 = c01 * (1.0 - wy) + c11 * wy;
  return c0 * (1.0 - wz) + c1 * wz;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = u32(uniforms.count_pad.y);
  if (idx >= count) { return; }
  let inst_id = candidate_ids[idx];
  let cam = uniforms.camera_threshold.xyz;
  let threshold = uniforms.camera_threshold.w;
  var fused: array<f32, ${inputDim}>;
${levelLines}
${viewCode}
${relativeCode}
  for (var i = 0u; i < ${meta.latentDim}u; i = i + 1u) {
    fused[CAMERA_DIM + i] = w(OFFSET_LATENT + inst_id * ${meta.latentDim}u + i);
  }
  for (var i = 0u; i < ${meta.geoDim}u; i = i + 1u) {
    fused[CAMERA_DIM + ${meta.latentDim}u + i] = w(OFFSET_GEO + inst_id * ${meta.geoDim}u + i);
  }
${interactionCode}
  var h0: array<f32, ${hidden}>;
  for (var o = 0u; o < HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_B0 + o);
    for (var i = 0u; i < INPUT_DIM; i = i + 1u) {
      sum = sum + w(OFFSET_W0 + o * INPUT_DIM + i) * fused[i];
    }
    h0[o] = max(sum, 0.0);
  }
  var h1: array<f32, ${hidden}>;
  for (var o = 0u; o < HIDDEN; o = o + 1u) {
    var sum = w(OFFSET_B1 + o);
    for (var i = 0u; i < HIDDEN; i = i + 1u) {
      sum = sum + w(OFFSET_W1 + o * HIDDEN + i) * h0[i];
    }
    h1[o] = max(sum, 0.0);
  }
  var logit = w(OFFSET_B2);
  for (var i = 0u; i < HIDDEN; i = i + 1u) {
    logit = logit + w(OFFSET_W2 + i) * h1[i];
  }
  let p = 1.0 / (1.0 + exp(-logit));
  let q = u32(clamp(p, 0.0, 1.0) * 65535.0);
  let flag = select(0u, 1u, p >= threshold);
  output_values[idx * OUTPUT_VALUE_WORDS] = (q << 1u) | flag;
}
`;
  }

  _buildViewFeatureShader(hashDim, bands) {
    let code = `
  let raw_view = vec3<f32>(uniforms.view_tanfov.x, uniforms.view_tanfov.y, uniforms.view_tanfov.z);
  let view_len = max(length(raw_view), 1e-6);
  let view = raw_view / view_len;
  let tan_x = uniforms.view_tanfov.w;
  let tan_y = uniforms.count_pad.x;
  fused[${hashDim}u + 0u] = view.x;
  fused[${hashDim}u + 1u] = view.y;
  fused[${hashDim}u + 2u] = view.z;
  fused[${hashDim}u + 3u] = tan_x;
  fused[${hashDim}u + 4u] = tan_y;`;
    for (let band = 0; band < bands; band += 1) {
      const base = hashDim + 5 + band * 10;
      const freq = Math.PI * (2 ** band);
      code += `
  fused[${base}u + 0u] = sin(view.x * ${freq.toFixed(8)});
  fused[${base}u + 1u] = sin(view.y * ${freq.toFixed(8)});
  fused[${base}u + 2u] = sin(view.z * ${freq.toFixed(8)});
  fused[${base}u + 3u] = sin(tan_x * ${freq.toFixed(8)});
  fused[${base}u + 4u] = sin(tan_y * ${freq.toFixed(8)});
  fused[${base}u + 5u] = cos(view.x * ${freq.toFixed(8)});
  fused[${base}u + 6u] = cos(view.y * ${freq.toFixed(8)});
  fused[${base}u + 7u] = cos(view.z * ${freq.toFixed(8)});
  fused[${base}u + 8u] = cos(tan_x * ${freq.toFixed(8)});
  fused[${base}u + 9u] = cos(tan_y * ${freq.toFixed(8)});`;
    }
    return code;
  }

  _buildSphericalViewFeatureShader(hashDim, bands) {
    let code = `
  let raw_view = vec3<f32>(uniforms.view_tanfov.x, uniforms.view_tanfov.y, uniforms.view_tanfov.z);
  let view_len = max(length(raw_view), 1e-6);
  let view = raw_view / view_len;
  let xz_len = max(length(vec2<f32>(view.x, view.z)), 1e-6);
  let tan_x = clamp(uniforms.view_tanfov.w, 0.0001, 4.0);
  let tan_y = clamp(uniforms.count_pad.x, 0.0001, 4.0);
  var view_base: array<f32, 5>;
  view_base[0] = view.x / xz_len;
  view_base[1] = view.z / xz_len;
  view_base[2] = clamp(view.y, -1.0, 1.0);
  view_base[3] = tan_x;
  view_base[4] = tan_y;
  for (var i = 0u; i < 5u; i = i + 1u) {
    fused[${hashDim}u + i] = view_base[i];
  }`;
    for (let band = 0; band < bands; band += 1) {
      const base = hashDim + 5 + band * 10;
      const freq = Math.PI * (2 ** band);
      code += `
  for (var i${band} = 0u; i${band} < 5u; i${band} = i${band} + 1u) {
    fused[${base}u + i${band}] = sin(view_base[i${band}] * ${freq.toFixed(8)});
    fused[${base + 5}u + i${band}] = cos(view_base[i${band}] * ${freq.toFixed(8)});
  }`;
    }
    return code;
  }

  _buildRelativeFeatureShader(offset, boundsOffset, worldScale) {
    return `
  let bounds_base = ${boundsOffset}u + inst_id * 6u;
  let bmin = vec3<f32>(w(bounds_base + 0u), w(bounds_base + 1u), w(bounds_base + 2u));
  let bmax = vec3<f32>(w(bounds_base + 3u), w(bounds_base + 4u), w(bounds_base + 5u));
  let center = (bmin + bmax) * 0.5;
  let size = max(bmax - bmin, vec3<f32>(0.0001));
  let camera_world = uniforms.camera_world_pad.xyz;
  let delta = center - camera_world;
  let scene_scale = max(vec3<f32>(${Number(worldScale[0]).toFixed(8)}, ${Number(worldScale[1]).toFixed(8)}, ${Number(worldScale[2]).toFixed(8)}), vec3<f32>(0.0001));
  let delta_norm = clamp(delta / scene_scale, vec3<f32>(-4.0), vec3<f32>(4.0));
  let dist = max(length(delta), 0.0001);
  let dir = delta / dist;
  let raw_forward = vec3<f32>(uniforms.view_tanfov.x, uniforms.view_tanfov.y, uniforms.view_tanfov.z);
  let forward = raw_forward / max(length(raw_forward), 0.000001);
  let up_ref_a = vec3<f32>(0.0, 1.0, 0.0);
  let up_ref_b = vec3<f32>(0.0, 0.0, 1.0);
  var up_seed = up_ref_a;
  if (abs(dot(forward, up_ref_a)) > 0.98) {
    up_seed = up_ref_b;
  }
  let right = normalize(cross(forward, up_seed));
  let up = normalize(cross(right, forward));
  let dot_forward = dot(dir, forward);
  let dot_right = dot(dir, right);
  let dot_up = dot(dir, up);
  let tan_x_rel = max(uniforms.view_tanfov.w, 0.0001);
  let tan_y_rel = max(uniforms.count_pad.x, 0.0001);
  let radius = length(size) * 0.5;
  let angular = radius / max(dist, 1.0);
  fused[${offset}u + 0u] = delta_norm.x;
  fused[${offset}u + 1u] = delta_norm.y;
  fused[${offset}u + 2u] = delta_norm.z;
  fused[${offset}u + 3u] = log(1.0 + dist / 100.0);
  fused[${offset}u + 4u] = dir.x;
  fused[${offset}u + 5u] = dir.y;
  fused[${offset}u + 6u] = dir.z;
  fused[${offset}u + 7u] = dot_forward;
  fused[${offset}u + 8u] = dot_right;
  fused[${offset}u + 9u] = dot_up;
  fused[${offset}u + 10u] = clamp(dot_right / max(dot_forward * tan_x_rel, 0.0001), -4.0, 4.0);
  fused[${offset}u + 11u] = clamp(dot_up / max(dot_forward * tan_y_rel, 0.0001), -4.0, 4.0);
  fused[${offset}u + 12u] = clamp(angular / tan_x_rel, 0.0, 4.0);
  fused[${offset}u + 13u] = clamp(angular / tan_y_rel, 0.0, 4.0);
  fused[${offset}u + 14u] = clamp(max(max(size.x / scene_scale.x, size.y / scene_scale.y), size.z / scene_scale.z), 0.0, 4.0);`;
  }
}
