import { InstancePVS } from './InstancePVS.js';
import { InstancePVSCPU } from './InstancePVSCPU.js';
import {
  canAttemptWebGPU,
  describeBackendFallback,
  isWebGPUBackendFailure,
  normalizeInstancePVSBackend,
} from './InstancePVSBackendPolicy.js';

export class InstancePVSRuntime {
  constructor(assetBaseUrl, options = {}) {
    this.assetBaseUrl = assetBaseUrl;
    this.options = { ...options };
    this.backendPreference = normalizeInstancePVSBackend(options.backendPreference);
    this.active = null;
    this.backend = 'uninitialized';
    this.fallbackReason = null;
    this.isReady = false;
    this.initError = null;
    this.lastInitTimings = null;
    this.lastPredictTimings = null;
    this.lastQuery = null;
    this.lastQueryOptions = null;
  }

  get meta() {
    return this.active?.meta || this.options.preloadedMeta || null;
  }

  async init() {
    if (this.isReady) return this;
    try {
      if (this.backendPreference === 'cpu') {
        await this._activateCPU();
      } else if (this.backendPreference === 'webgpu') {
        await this._activateWebGPU();
      } else if (canAttemptWebGPU()) {
        try {
          await this._activateWebGPU();
        } catch (error) {
          if (!isWebGPUBackendFailure(error)) throw error;
          await this._activateCPU(error);
        }
      } else {
        await this._activateCPU(new Error('WebGPU is unavailable in this browser.'));
      }
      this.isReady = true;
      this.initError = null;
      return this;
    } catch (error) {
      this.initError = error;
      throw error;
    }
  }

  async _activateWebGPU() {
    const runtime = new InstancePVS(this.assetBaseUrl, this.options);
    try {
      await runtime.init();
    } catch (error) {
      runtime.dispose();
      throw error;
    }
    this.active?.dispose();
    this.active = runtime;
    this.backend = runtime.backend;
    this.lastInitTimings = runtime.lastInitTimings;
  }

  async _activateCPU(reason = null) {
    const runtime = new InstancePVSCPU(this.assetBaseUrl, this.options);
    try {
      await runtime.init();
    } catch (error) {
      runtime.dispose();
      throw error;
    }
    this.active?.dispose();
    this.active = runtime;
    this.backend = runtime.backend;
    if (reason) this.fallbackReason = describeBackendFallback(reason);
    this.lastInitTimings = {
      ...(runtime.lastInitTimings || {}),
      fallbackReason: this.fallbackReason,
    };
  }

  async _fallbackAfterRuntimeFailure(error) {
    if (this.backendPreference !== 'auto' || !this.backend.includes('webgpu')) throw error;
    if (!isWebGPUBackendFailure(error, { runtimeFailure: true })) throw error;
    await this._activateCPU(error);
    this.isReady = true;
  }

  async predict(queryCamera, options = {}) {
    this.lastQuery = queryCamera;
    this.lastQueryOptions = options;
    try {
      const result = await this.active.predict(queryCamera, options);
      this.backend = this.active.backend;
      this.lastPredictTimings = result?.timings || this.active.lastPredictTimings;
      return result;
    } catch (error) {
      await this._fallbackAfterRuntimeFailure(error);
      const result = await this.active.predict(queryCamera, options);
      this.backend = this.active.backend;
      this.lastPredictTimings = result?.timings || this.active.lastPredictTimings;
      return result;
    }
  }

  async refilter(renderCamera) {
    try {
      return await this.active.refilter(renderCamera);
    } catch (error) {
      await this._fallbackAfterRuntimeFailure(error);
      if (!this.lastQuery) throw new Error('Cannot rebuild the neural prediction after WebGPU failure.');
      await this.active.predict(this.lastQuery, this.lastQueryOptions || {});
      return this.active.refilter(renderCamera);
    }
  }

  dispose() {
    this.active?.dispose();
    this.active = null;
    this.isReady = false;
    this.backend = 'disposed';
  }
}
