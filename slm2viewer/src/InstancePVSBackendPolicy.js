const WEBGPU_ERROR_TOKENS = Object.freeze([
  'webgpu',
  'gpuadapter',
  'gpudevice',
  'gpubuffer',
  'requestadapter',
  'requestdevice',
  'wgsl',
  'device lost',
  'device was lost',
  'operationerror',
  'validation error',
  'mapasync',
]);

export function normalizeInstancePVSBackend(value) {
  const normalized = String(value || 'auto').trim().toLowerCase();
  if (normalized === 'wasm' || normalized === 'webgpu') return normalized;
  return 'auto';
}

export function canAttemptWebGPU(scope = globalThis) {
  return Boolean(scope && scope.navigator && scope.navigator.gpu);
}

export function isWebGPUBackendFailure(error, options = {}) {
  if (options.runtimeFailure === true) return true;
  if (!canAttemptWebGPU(options.scope || globalThis)) return true;
  const text = `${error?.name || ''} ${error?.message || error || ''}`.toLowerCase();
  return WEBGPU_ERROR_TOKENS.some((token) => text.includes(token));
}

export function describeBackendFallback(error) {
  const message = String(error?.message || error || 'WebGPU is unavailable')
    .replace(/\s+/g, ' ')
    .trim();
  return message.slice(0, 240) || 'WebGPU is unavailable';
}
