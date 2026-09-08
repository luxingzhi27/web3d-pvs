import { LoadingManager } from 'three';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';

const manager = new LoadingManager();

export const DRACO_LOADER = new DRACOLoader(manager).setDecoderPath('./assets/three/draco/gltf/');
export const KTX2_LOADER = new KTX2Loader(manager).setTranscoderPath('./assets/three/basis/');
export const LOCAL_RUNTIME_ASSET_VERSION = 'pvs-mainline-v4-hkust-20260825';
export const INITIAL_GLB_PRELOAD_LIMIT = 100;

export function withLocalRuntimeVersion(url) {
  if (!url) return url;
  return `${url}${url.includes('?') ? '&' : '?'}v=${encodeURIComponent(LOCAL_RUNTIME_ASSET_VERSION)}`;
}

export function joinUrlPath(baseUrl, pathPart) {
  const base = String(baseUrl || '').replace(/\/+$/, '');
  const suffix = String(pathPart || '').replace(/^\/+/, '');
  return `${base}/${suffix}`;
}
