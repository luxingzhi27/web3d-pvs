import { FRONTEND_RENDER_FOV_Y_DEG } from './neuralPvsFovProtocol.js';
import { normalizeInstancePVSBackend } from './InstancePVSBackendPolicy.js';
import { PVSQuerySession } from './PVSQuerySession.js';

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function resolveUrl(url) {
  if (!url) return '';
  if (typeof window === 'undefined' || !window.location) return url;
  return new URL(url, window.location.href).toString();
}

function cameraSnapshot(camera) {
  if (!camera) throw new Error('PVSDispatcher requires a camera.');
  camera.updateMatrixWorld?.(true);
  camera.updateProjectionMatrix?.();
  return {
    position: [camera.position.x, camera.position.y, camera.position.z],
    quaternion: [camera.quaternion.x, camera.quaternion.y, camera.quaternion.z, camera.quaternion.w],
    fov: Number.isFinite(camera.fov) ? camera.fov : FRONTEND_RENDER_FOV_Y_DEG,
    aspect: Number.isFinite(camera.aspect) ? camera.aspect : 16 / 9,
    near: Number.isFinite(camera.near) ? camera.near : 0.1,
    far: Number.isFinite(camera.far) ? camera.far : 20000,
  };
}

export class PVSDispatcher {
  constructor(assetBaseUrl, options = {}) {
    this.assetBaseUrl = String(assetBaseUrl || '').replace(/\/$/, '');
    this.options = { ...options };
    this.backendPreference = normalizeInstancePVSBackend(options.backendPreference);
    this.webgpuContext = options.externalWebGPUContext || null;
    this.worker = null;
    this.session = null;
    this.isReady = false;
    this.backend = 'uninitialized';
    this.initError = null;
    this.initPromise = null;
    this.lastInitTimings = null;
    this.lastPredictTimings = null;
    this.lastFilterTimings = null;
    this.modelInfo = null;
    this.predictSerial = 0;
    this.pending = new Map();
    this.pendingInit = null;
    this.fallbackReason = null;
    this.renderComponentIds = new Set();
    this.renderGlbIds = new Set();
  }

  async init() {
    if (this.isReady) return this;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._init().catch((error) => {
      this.initPromise = null;
      this.initError = error;
      throw error;
    });
    return this.initPromise;
  }

  async _init() {
    const startedAt = nowMs();
    if (this.webgpuContext && this.backendPreference !== 'wasm') {
      try {
        await this._initDirect();
      } catch (error) {
        if (this.backendPreference === 'webgpu') throw error;
        await this._initWorker(error);
      }
    } else {
      if (this.backendPreference === 'webgpu') {
        throw new Error('WebGPU PVS was requested but the renderer has no WebGPU device.');
      }
      const reason = this.backendPreference === 'auto'
        ? new Error('The renderer is using WebGL2, so PVS is running in the Worker WASM SIMD backend.')
        : null;
      await this._initWorker(reason);
    }
    this.isReady = true;
    this.initError = null;
    this.lastInitTimings = {
      ...(this.lastInitTimings || {}),
      dispatcherTotalMs: nowMs() - startedAt,
    };
    return this;
  }

  async _initDirect() {
    this.session = new PVSQuerySession(resolveUrl(this.assetBaseUrl), {
      ...this.options,
      backendPreference: 'webgpu',
      externalWebGPUContext: this.webgpuContext,
      executionLocation: 'renderer',
    });
    await this.session.init();
    this.backend = this.session.backend;
    this.lastInitTimings = this.session.initTimings;
    this.modelInfo = this.session.getModelInfo();
  }

  async _initWorker(reason = null) {
    this.session?.dispose();
    this.session = null;
    this.fallbackReason = reason ? String(reason?.message || reason) : this.fallbackReason;
    this.worker?.terminate();
    this.worker = new Worker('./PVSWorker.js');
    this.worker.onmessage = (event) => this._handleWorkerMessage(event.data || {});
    this.worker.onerror = (event) => this._failWorker(new Error(event.message || 'PVS worker failed.'));
    await new Promise((resolve, reject) => {
      this.pendingInit = { resolve, reject };
      this.worker.postMessage({
        type: 'init',
        assetBaseUrl: resolveUrl(this.assetBaseUrl),
        assetVersion: this.options.assetVersion || null,
        debugLogging: Boolean(this.options.debugLogging),
        maxPrefetch: Number(this.options.maxPrefetch || 2048),
        prefetchThreshold: this.options.prefetchThreshold,
        downloadPlanMode: this.options.downloadPlanMode,
        fallbackReason: this.fallbackReason,
      });
    });
  }

  _failWorker(error) {
    this.pendingInit?.reject(error);
    this.pendingInit = null;
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    this.initError = error;
    this.backend = 'worker-error';
  }

  _handleWorkerMessage(data) {
    if (data.type === 'ready') {
      this.backend = data.backend || 'worker-wasm-simd-v4';
      this.lastInitTimings = data.timings || null;
      this.modelInfo = data.modelInfo || this.modelInfo;
      if (data.fallbackReason) this.fallbackReason = data.fallbackReason;
      this.pendingInit?.resolve(this);
      this.pendingInit = null;
      return;
    }
    if (data.type === 'error') {
      const error = new Error(data.message || 'PVS worker error.');
      error.stack = data.stack || error.stack;
      const pending = this.pending.get(data.serial);
      if (pending) {
        this.pending.delete(data.serial);
        pending.reject(error);
      } else {
        this._failWorker(error);
      }
      return;
    }
    if (data.type !== 'result' && data.type !== 'filter-result') return;
    const pending = this.pending.get(data.serial);
    if (!pending) return;
    this.pending.delete(data.serial);
    pending.resolve(data);
  }

  _requestWorker(type, snapshot, serial) {
    return new Promise((resolve, reject) => {
      this.pending.set(serial, { resolve, reject });
      this.worker.postMessage({ type, serial, snapshot });
    });
  }

  _rememberRenderState(result) {
    if (result.type === 'result') {
      this.renderComponentIds = new Set(Array.from(result.renderComponentModelList || [], Number));
      this.renderGlbIds = new Set(Array.from(result.renderModelList || [], Number));
      return;
    }
    for (const id of result.renderComponentRemovedIds || []) this.renderComponentIds.delete(Number(id));
    for (const id of result.renderComponentAddedIds || []) this.renderComponentIds.add(Number(id));
    for (const id of result.renderGlbRemovedIds || []) this.renderGlbIds.delete(Number(id));
    for (const id of result.renderGlbAddedIds || []) this.renderGlbIds.add(Number(id));
  }

  _finalize(result, serial) {
    result.stale = serial !== this.predictSerial;
    result.fallbackReason = result.fallbackReason || this.fallbackReason;
    this.backend = result.backend || this.backend;
    this._rememberRenderState(result);
    const timings = {
      serial,
      ...(result.timings || {}),
      candidateSelection: result.candidateSelection || result.timings?.candidateSelection || null,
      stale: result.stale,
      backend: this.backend,
      fallbackReason: result.fallbackReason || null,
      modelInfo: result.timings?.modelInfo || this.modelInfo || null,
    };
    if (result.type === 'filter-result') this.lastFilterTimings = timings;
    else this.lastPredictTimings = timings;
    return result;
  }

  async _switchToWorker(error) {
    if (this.backendPreference !== 'auto') throw error;
    await this._initWorker(error);
    this.isReady = true;
  }

  async predict(renderCamera) {
    if (!this.isReady) return null;
    const serial = ++this.predictSerial;
    const snapshot = cameraSnapshot(renderCamera);
    let result;
    if (this.session) {
      try {
        result = await this.session.predict(snapshot, serial);
      } catch (error) {
        await this._switchToWorker(error);
        result = await this._requestWorker('predict', snapshot, serial);
      }
    } else {
      result = await this._requestWorker('predict', snapshot, serial);
    }
    return this._finalize(result, serial);
  }

  async refilter(renderCamera) {
    if (!this.isReady) return null;
    const serial = ++this.predictSerial;
    const snapshot = cameraSnapshot(renderCamera);
    let result;
    if (this.session) {
      try {
        result = await this.session.refilter(snapshot, serial);
      } catch (error) {
        const oldComponents = this.renderComponentIds;
        const oldGlbs = this.renderGlbIds;
        await this._switchToWorker(error);
        const replacement = await this._requestWorker('predict', snapshot, serial);
        const nextComponents = new Set(Array.from(replacement.renderComponentModelList || [], Number));
        const nextGlbs = new Set(Array.from(replacement.renderModelList || [], Number));
        result = {
          type: 'filter-result',
          serial,
          idMode: replacement.idMode,
          backend: `${replacement.backend}-recovered-render-filter`,
          fallbackReason: this.fallbackReason,
          renderComponentAddedIds: Uint32Array.from([...nextComponents].filter((id) => !oldComponents.has(id))),
          renderComponentRemovedIds: Uint32Array.from([...oldComponents].filter((id) => !nextComponents.has(id))),
          renderGlbAddedIds: Uint32Array.from([...nextGlbs].filter((id) => !oldGlbs.has(id))),
          renderGlbRemovedIds: Uint32Array.from([...oldGlbs].filter((id) => !nextGlbs.has(id))),
          renderInstanceCount: nextComponents.size,
          renderGlbCount: nextGlbs.size,
          renderRevision: replacement.renderRevision,
          timings: replacement.timings,
        };
      }
    } else {
      result = await this._requestWorker('filter', snapshot, serial);
    }
    return this._finalize(result, serial);
  }

  setDownloadPlanMode(value) {
    const mode = value === 'raw-visible' ? 'raw-visible' : 'viewcell-priority';
    this.options.downloadPlanMode = mode;
    this.session?.setDownloadPlanMode(mode);
    this.worker?.postMessage({ type: 'setDownloadPlanMode', downloadPlanMode: mode });
    return mode;
  }

  dispose() {
    this.session?.dispose();
    this.session = null;
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
    this.pendingInit = null;
    this.isReady = false;
    this.backend = 'disposed';
  }
}
