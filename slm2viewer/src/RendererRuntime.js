import { SRGBColorSpace, WebGPURenderer } from 'three/webgpu';
import { GPUFeatureName } from 'three/src/renderers/webgpu/utils/WebGPUConstants.js';
import { RendererEffects } from './RendererEffects.js';

const SOFTWARE_ADAPTER = /swiftshader|llvmpipe|softpipe|swrast|software/i;

function normalizeBackend(value) {
  const backend = String(value || 'auto').toLowerCase();
  return backend === 'webgpu' || backend === 'webgl' ? backend : 'auto';
}

async function createWebGPUContext() {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
    featureLevel: 'compatibility',
  });
  if (!adapter) return null;
  const adapterInfo = {
    vendor: adapter.info?.vendor || '',
    architecture: adapter.info?.architecture || '',
    device: adapter.info?.device || '',
    description: adapter.info?.description || '',
  };
  const softwareAdapter = SOFTWARE_ADAPTER.test(Object.values(adapterInfo).join(' '));
  if (softwareAdapter) {
    throw new Error(`Hardware WebGPU is unavailable (${adapterInfo.architecture || adapterInfo.vendor}).`);
  }
  const device = await adapter.requestDevice({
    requiredFeatures: Object.values(GPUFeatureName).filter((name) => adapter.features.has(name)),
  });
  return {
    adapter,
    device,
    adapterInfo,
  };
}

export class RendererRuntime {
  static async create(options = {}) {
    const runtime = new RendererRuntime(options);
    await runtime.init();
    return runtime;
  }

  constructor(options = {}) {
    this.options = { ...options };
    this.backendPreference = normalizeBackend(options.backendPreference);
    this.renderer = null;
    this.webgpuContext = null;
    this.effects = null;
    this.backend = 'uninitialized';
    this.fallbackReason = null;
    this.deviceLossInfo = null;
  }

  async init() {
    const rendererOptions = {
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    };

    if (this.backendPreference !== 'webgl') {
      try {
        this.webgpuContext = await createWebGPUContext();
      } catch (error) {
        if (this.backendPreference === 'webgpu') throw error;
        this.fallbackReason = String(error?.message || error);
      }
    }

    if (this.backendPreference === 'webgpu' && !this.webgpuContext) {
      throw new Error('WebGPU rendering was requested but no high-performance adapter is available.');
    }

    this.renderer = new WebGPURenderer({
      ...rendererOptions,
      ...(this.webgpuContext
        ? { device: this.webgpuContext.device }
        : { forceWebGL: true }),
    });
    await this.renderer.init();

    if (this.renderer.backend.isWebGPUBackend) {
      this.backend = 'webgpu';
      this.webgpuContext.device.lost.then((info) => {
        if (info.reason !== 'destroyed') {
          this.deviceLossInfo = {
            reason: info.reason || 'unknown',
            message: info.message || '',
          };
        }
      });
    } else {
      this.backend = 'webgl2-fallback';
      this.webgpuContext = null;
      if (!this.fallbackReason && this.backendPreference !== 'webgl') {
        this.fallbackReason = 'WebGPU is unavailable; Three.js selected its WebGL2 backend.';
      }
    }

    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.setClearColor(0xdddddd);
    return this;
  }

  configureScene(scene, camera, options = {}) {
    this.effects?.dispose();
    this.effects = new RendererEffects(this.renderer, scene, camera, options);
    return this.effects;
  }

  configureEffects(options = {}) {
    this.effects?.configure(options);
  }

  configureSurface(surface) {
    this.renderer.setDrawingBufferSize(
      surface.cssWidth,
      surface.cssHeight,
      surface.pixelRatio,
    );
    this.renderer.domElement.style.width = `${surface.cssWidth}px`;
    this.renderer.domElement.style.height = `${surface.cssHeight}px`;
  }

  render(scene, camera) {
    if (this.effects && this.effects.scene === scene && this.effects.camera === camera) {
      this.effects.render();
    } else {
      this.renderer.render(scene, camera);
    }
  }

  getSharedWebGPUContext() {
    if (this.backend !== 'webgpu' || !this.webgpuContext) return null;
    return this.webgpuContext;
  }

  getInfo() {
    return {
      backend: this.backend,
      fallbackReason: this.fallbackReason,
      adapter: this.webgpuContext?.adapterInfo || null,
      sharedDevice: Boolean(this.getSharedWebGPUContext()),
      deviceLimits: this.webgpuContext ? {
        maxBufferSize: Number(this.webgpuContext.device.limits.maxBufferSize),
        maxStorageBufferBindingSize: Number(
          this.webgpuContext.device.limits.maxStorageBufferBindingSize,
        ),
      } : null,
      deviceLossInfo: this.deviceLossInfo,
    };
  }

  dispose() {
    this.effects?.dispose();
    this.effects = null;
    this.renderer?.dispose();
    this.renderer = null;
    this.webgpuContext?.device?.destroy();
    this.webgpuContext = null;
    this.backend = 'disposed';
  }
}
