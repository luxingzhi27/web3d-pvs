import { FRONTEND_RENDER_FOV_Y_DEG } from './neuralPvsFovProtocol.js';
import { normalizeInstancePVSBackend } from './InstancePVSBackendPolicy.js';

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function resolveUrl(url) {
  if (!url) return '';
  if (typeof window === 'undefined' || !window.location) return url;
  return new URL(url, window.location.href).toString();
}

function numberArrayFromVector3(value, fallback = [0, 0, 0]) {
  if (!value) return fallback.slice();
  return [
    Number.isFinite(value.x) ? value.x : fallback[0],
    Number.isFinite(value.y) ? value.y : fallback[1],
    Number.isFinite(value.z) ? value.z : fallback[2],
  ];
}

function numberArrayFromQuaternion(value) {
  if (!value) return [0, 0, 0, 1];
  return [
    Number.isFinite(value.x) ? value.x : 0,
    Number.isFinite(value.y) ? value.y : 0,
    Number.isFinite(value.z) ? value.z : 0,
    Number.isFinite(value.w) ? value.w : 1,
  ];
}

function matrixElements(matrix) {
  return matrix && matrix.elements ? Array.from(matrix.elements) : [];
}

export class LightweightPVSDispatcher {
  constructor(assetBaseUrl, options = {}) {
    this.assetBaseUrl = (assetBaseUrl || '').replace(/\/$/, '');
    this.assetVersion = options.assetVersion || null;
    this.debugLogging = Boolean(options.debugLogging);
    this.cpuPerfMode = options.cpuPerfMode === 'mobile' ? 'mobile' : 'balanced';
    this.maxPrefetch = Number(options.maxPrefetch || (this.cpuPerfMode === 'mobile' ? 768 : 2048));
    this.prefetchThreshold = options.prefetchThreshold !== undefined
      ? Number(options.prefetchThreshold)
      : null;
    this.downloadPlanMode = options.downloadPlanMode === 'raw-visible' ? 'raw-visible' : 'viewcell-priority';
    this.backendPreference = normalizeInstancePVSBackend(options.backendPreference);

    this.worker = null;
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
    this.pendingInitResolve = null;
    this.pendingInitReject = null;
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
    const start = nowMs();
    try {
      // Parcel v1 recognizes string-literal Worker URLs and bundles the worker
      // as a separate classic worker. Avoid import.meta.url because this app is
      // emitted as a classic script, not as an ES module bundle.
      this.worker = new Worker('./LightweightPVSWorker.js');
    } catch (error) {
      this.backend = 'worker-create-failed';
      this.initError = error;
      throw error;
    }

    this.worker.onmessage = (event) => this._handleWorkerMessage(event.data || {});
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'Lightweight PVS worker failed.');
      if (this.pendingInitReject) {
        this.pendingInitReject(error);
        this.pendingInitReject = null;
        this.pendingInitResolve = null;
      }
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
      this.initError = error;
      this.backend = 'worker-error';
    };

    await new Promise((resolve, reject) => {
      this.pendingInitResolve = resolve;
      this.pendingInitReject = reject;
      this.worker.postMessage({
        type: 'init',
        assetBaseUrl: resolveUrl(this.assetBaseUrl),
        assetVersion: this.assetVersion,
        debugLogging: this.debugLogging,
        maxPrefetch: this.maxPrefetch,
        prefetchThreshold: this.prefetchThreshold,
        downloadPlanMode: this.downloadPlanMode,
        backendPreference: this.backendPreference,
      });
    });
    this.isReady = true;
    this.lastInitTimings = {
      ...(this.lastInitTimings || {}),
      dispatcherTotalMs: nowMs() - start,
    };
    return this;
  }

  _handleWorkerMessage(data) {
    if (data.type === 'ready') {
      this.backend = data.backend || 'worker';
      this.lastInitTimings = data.timings || null;
      this.modelInfo = data.modelInfo || this.modelInfo;
      if (data.fallbackReason && this.lastInitTimings) {
        this.lastInitTimings.fallbackReason = data.fallbackReason;
      }
      this.isReady = true;
      this.initError = null;
      if (this.pendingInitResolve) {
        this.pendingInitResolve(this);
        this.pendingInitResolve = null;
        this.pendingInitReject = null;
      }
      return;
    }

    if (data.type === 'error') {
      const error = new Error(data.message || 'Lightweight PVS worker error.');
      error.stack = data.stack || error.stack;
      if (data.serial != null && this.pending.has(data.serial)) {
        const pending = this.pending.get(data.serial);
        this.pending.delete(data.serial);
        pending.reject(error);
        return;
      }
      if (this.pendingInitReject) {
        this.pendingInitReject(error);
        this.pendingInitReject = null;
        this.pendingInitResolve = null;
      }
      this.initError = error;
      return;
    }

    if (data.type !== 'result' && data.type !== 'filter-result') return;
    const pending = this.pending.get(data.serial);
    if (!pending) return;
    this.pending.delete(data.serial);
    data.stale = data.serial !== this.predictSerial;
    this.backend = data.backend || this.backend;
    const timings = {
      serial: data.serial,
      ...(data.timings || {}),
      candidateSelection: data.candidateSelection || data.timings?.candidateSelection || null,
      stale: data.stale,
      backend: data.backend || this.backend,
      fallbackReason: data.fallbackReason || null,
      modelInfo: (data.timings && data.timings.modelInfo) || data.modelInfo || this.modelInfo || null,
    };
    if (data.type === 'filter-result') this.lastFilterTimings = timings;
    else this.lastPredictTimings = timings;
    pending.resolve(data);
  }

  _cameraSnapshot(camera) {
    if (!camera) throw new Error('LightweightPVSDispatcher.predict requires a camera.');
    if (typeof camera.updateMatrixWorld === 'function') camera.updateMatrixWorld(true);
    if (typeof camera.updateProjectionMatrix === 'function') camera.updateProjectionMatrix();
    return {
      position: numberArrayFromVector3(camera.position),
      quaternion: numberArrayFromQuaternion(camera.quaternion),
      fov: Number.isFinite(camera.fov) ? camera.fov : FRONTEND_RENDER_FOV_Y_DEG,
      aspect: Number.isFinite(camera.aspect) ? camera.aspect : 16 / 9,
      near: Number.isFinite(camera.near) ? camera.near : 0.1,
      far: Number.isFinite(camera.far) ? camera.far : 20000,
      projectionMatrix: matrixElements(camera.projectionMatrix),
      matrixWorldInverse: matrixElements(camera.matrixWorldInverse),
    };
  }

  async predict(renderCamera) {
    if (!this.isReady) return null;
    const serial = ++this.predictSerial;
    const snapshot = this._cameraSnapshot(renderCamera);
    return new Promise((resolve, reject) => {
      this.pending.set(serial, { resolve, reject, startedAt: nowMs() });
      this.worker.postMessage({
        type: 'predict',
        serial,
        snapshot,
      });
    });
  }

  async refilter(renderCamera) {
    if (!this.isReady) return null;
    const serial = ++this.predictSerial;
    const snapshot = this._cameraSnapshot(renderCamera);
    return new Promise((resolve, reject) => {
      this.pending.set(serial, { resolve, reject, startedAt: nowMs(), operation: 'filter' });
      this.worker.postMessage({
        type: 'filter',
        serial,
        snapshot,
      });
    });
  }

  setDownloadPlanMode(value) {
    this.downloadPlanMode = value === 'raw-visible' ? 'raw-visible' : 'viewcell-priority';
    if (this.worker) {
      this.worker.postMessage({
        type: 'setDownloadPlanMode',
        downloadPlanMode: this.downloadPlanMode,
      });
    }
    return this.downloadPlanMode;
  }

  dispose() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.pending.clear();
    this.isReady = false;
  }
}
