#!/usr/bin/env node
/* Lightweight integrity smoke for the currently retained scenes and models. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const viewerDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const readJson = (relativePath) => JSON.parse(
  fs.readFileSync(path.join(viewerDir, relativePath), 'utf8'),
);
const requireFile = (relativePath) => {
  const absolute = path.join(viewerDir, relativePath);
  if (!fs.existsSync(absolute)) {
    throw new Error(`Missing required file: ${relativePath}`);
  }
  return absolute;
};

const modelByScene = {
  'hkust-v3': 'pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best',
  'ifcbench_fantasy_metropolis_instanced_v2':
    'pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best',
};

for (const legacyPath of ['assets/glbIndex.json', 'assets/runtimeVisibilityMeta.json']) {
  if (fs.existsSync(path.join(viewerDir, legacyPath))) {
    throw new Error(`Legacy root runtime metadata must not be present: ${legacyPath}`);
  }
}

const config = readJson('assets/config.json');
if (!config.scenes || typeof config.scenes !== 'object') {
  throw new Error('assets/config.json has no scenes object.');
}

for (const [scene, modelName] of Object.entries(modelByScene)) {
  const sceneConfig = config.scenes[scene];
  if (!sceneConfig?.loaderConfig?.resourcesBaseUrl) {
    throw new Error(`Scene ${scene} is missing loaderConfig.resourcesBaseUrl.`);
  }

  const sceneRoot = `assets/scenes/${scene}`;
  const glbIndex = readJson(`${sceneRoot}/glbIndex.json`);
  const runtimeMeta = readJson(`${sceneRoot}/runtimeVisibilityMeta.json`);
  const modelMeta = readJson(`assets/neural_instance_culling/${modelName}/instance_model_meta.json`);
  requireFile(`assets/neural_instance_culling/${modelName}/instance_pvs_assets.bin`);

  const glbCount = Number(glbIndex.total);
  const instanceCount = Number(runtimeMeta.instanceCount);
  const runtimeGlbCount = Number(runtimeMeta.globalGlbCount);
  if (!Number.isInteger(glbCount) || glbCount <= 0) {
    throw new Error(`Invalid GLB count for ${scene}: ${glbIndex.total}`);
  }
  if (!Number.isInteger(instanceCount) || instanceCount <= 0) {
    throw new Error(`Invalid instance count for ${scene}: ${runtimeMeta.instanceCount}`);
  }
  if (glbCount !== runtimeGlbCount || glbCount !== Number(modelMeta.numGlbs)) {
    throw new Error(`GLB count mismatch for ${scene}: index=${glbCount}, runtime=${runtimeGlbCount}, model=${modelMeta.numGlbs}`);
  }
  if (instanceCount !== Number(modelMeta.numInstances)) {
    throw new Error(`Instance count mismatch for ${scene}: runtime=${instanceCount}, model=${modelMeta.numInstances}`);
  }
  if (!Number.isFinite(Number(modelMeta.visibilityThreshold))) {
    throw new Error(`Missing visibility threshold for ${scene}.`);
  }

  console.log(`${scene}: ${instanceCount} instances, ${glbCount} GLBs, threshold=${modelMeta.visibilityThreshold}`);
}

console.log('Current scene/model integrity smoke passed.');
