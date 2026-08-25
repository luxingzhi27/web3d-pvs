#!/usr/bin/env node
/* Static contract smoke for the current frontend and its single V4 model. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const viewerDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modelName = 'pvs_mainline_v4';
const modelRoot = `assets/neural_instance_culling/${modelName}`;
const runtimeSchema = 'pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4';
const requiredModelFiles = [
  'model_meta.json',
  'instance_runtime_features_fp16.bin',
  'instance_aabb_fp32.bin',
  'instance_to_glb_uint32.bin',
  'query_weights_fp16.bin',
  'frequency_cycles_fp32.bin',
  'chi_table_fp32.bin',
];

function requireFile(relativePath) {
  const absolute = path.join(viewerDir, relativePath);
  if (!fs.existsSync(absolute)) throw new Error(`Missing required file: ${relativePath}`);
  return absolute;
}

function readJson(relativePath) {
  return JSON.parse(fs.readFileSync(requireFile(relativePath), 'utf8'));
}

const config = readJson('assets/config.json');
if (!config.scenes || typeof config.scenes !== 'object') {
  throw new Error('assets/config.json has no scenes object.');
}
const checkedSceneResources = new Set();
for (const [sceneName, sceneConfig] of Object.entries(config.scenes)) {
  const resourcesBaseUrl = sceneConfig?.loaderConfig?.resourcesBaseUrl;
  if (!resourcesBaseUrl) {
    throw new Error(`Scene ${sceneName} is missing loaderConfig.resourcesBaseUrl.`);
  }
  const match = /^\.\/assets\/scenes\/([^/]+)$/.exec(resourcesBaseUrl);
  if (!match) {
    throw new Error(`Scene ${sceneName} uses an unsupported resource path: ${resourcesBaseUrl}`);
  }
  const resourceScene = match[1];
  if (checkedSceneResources.has(resourceScene)) continue;
  checkedSceneResources.add(resourceScene);
  readJson(`assets/scenes/${resourceScene}/glbIndex.json`);
  readJson(`assets/scenes/${resourceScene}/runtimeVisibilityMeta.json`);
}

for (const file of requiredModelFiles) requireFile(`${modelRoot}/${file}`);
const meta = readJson(`${modelRoot}/model_meta.json`);
const runtimeMeta = readJson('assets/scenes/hkust-v3/runtimeVisibilityMeta.json');
const glbIndex = readJson('assets/scenes/hkust-v3/glbIndex.json');
if (meta.schema !== runtimeSchema || meta.modelConfig?.runtimeFeatureDim !== 124
    || meta.modelConfig?.runtimeHeadInputDim !== 130) {
  throw new Error('HKUST frontend asset is not the current V4 runtime.');
}
if (meta.safety?.safe !== true || !Number.isFinite(Number(meta.threshold))) {
  throw new Error('V4 frontend asset has no safe calibration threshold.');
}
if (Number(meta.query?.modelInputFovYDeg) !== 66
    || Number(meta.query?.frontendRenderFovYDeg) !== 60
    || !Number.isFinite(Number(meta.query?.candidateCameraBackOffsetM))) {
  throw new Error('V4 camera contract is incomplete.');
}
if (Number(meta.numInstances) !== Number(runtimeMeta.instanceCount)
    || Number(meta.numGlbs) !== Number(runtimeMeta.globalGlbCount)
    || Number(meta.numGlbs) !== Number(glbIndex.total)) {
  throw new Error('HKUST V4 model and scene metadata counts do not match.');
}
for (const descriptor of Object.values(meta.files || {})) {
  if (!descriptor?.file || descriptor.file === 'model_meta.json') continue;
  const absolute = requireFile(`${modelRoot}/${descriptor.file}`);
  if (fs.statSync(absolute).size !== Number(descriptor.byteLength)) {
    throw new Error(`V4 asset byte length mismatch: ${descriptor.file}`);
  }
}

const backendSource = fs.readFileSync(requireFile('src/neuralCullingBackendMode.js'), 'utf8');
const runtimeSource = fs.readFileSync(requireFile('src/InstancePVS.js'), 'utf8');
const workerSource = fs.readFileSync(requireFile('src/LightweightPVSWorker.js'), 'utf8');
const dispatcherSource = fs.readFileSync(requireFile('src/LightweightPVSDispatcher.js'), 'utf8');
const loaderSource = fs.readFileSync(requireFile('slm2/SLM2Loader.js'), 'utf8');
const viewerSource = fs.readFileSync(requireFile('src/viewer.js'), 'utf8');
if (fs.existsSync(path.join(viewerDir, 'src/NeuralPVS.js'))) {
  throw new Error('The unused legacy ONNX visibility backend must not return.');
}
for (const forbidden of [
  'directional-occlusion-proxy-scheduler-v1',
  'camera-hash-latent-geo',
  'dynamic-occlusion-pool',
  'spatial-feature-pages',
]) {
  if ([backendSource, runtimeSource, workerSource, dispatcherSource, loaderSource, viewerSource]
    .some((source) => source.includes(forbidden))) {
    throw new Error(`Old runtime compatibility remains in the frontend: ${forbidden}`);
  }
}

for (const forbidden of [
  'setRuntimeMeta(',
  'requestWebGPUUpgrade(',
  "data.type === 'upgraded'",
  '_checkNeuralBackendUpgrade',
  '_buildPriorityCandidates(',
  'refreshLoadingTaskAsync(',
  'Array.isArray(pred.modelList)',
  'lightweight-worker-pvs-v1',
  'predict(this.activeCamera.position',
  'predict(pose.position',
]) {
  if (dispatcherSource.includes(forbidden) || loaderSource.includes(forbidden)) {
    throw new Error(`Removed model compatibility API returned: ${forbidden}`);
  }
}
if (viewerSource.includes('Directional Proxy full40')
    || !viewerSource.includes('PVS V4 固定实例特征与查询网络')) {
  throw new Error('The debug panel still describes an old frontend model.');
}
if (workerSource.includes('testWorkpoint')
    || !workerSource.includes('calibrationWorkpoint')
    || !viewerSource.includes('Calibration安全工作点')) {
  throw new Error('The debug panel does not report the V4 calibration workpoint.');
}
if (!runtimeSource.includes('let shared_variance = 0.5 * (1.0 - chi_squared);')
    || runtimeSource.includes('second_sin - mean_sin * mean_sin')) {
  throw new Error('The V4 WebGPU spectral variance is not using the stable formulation.');
}
if (!workerSource.includes('buildCamera(snapshot, FRONTEND_RENDER_FOV_Y_DEG)')
    || !workerSource.includes('buildCandidateCamera(snapshot)')
    || !workerSource.includes('renderCamera: activeCamera')
    || !workerSource.includes('renderComponentModelList: renderComponentIds')) {
  throw new Error('The worker no longer follows the 60-degree render / 66-degree candidate contract.');
}
if (!runtimeSource.includes("source: 'gpu_v4_back_frustum_aabb'")
    || !runtimeSource.includes('fn intersects_frustum(instance_id: u32, render_frustum: bool)')
    || !runtimeSource.includes('atomicMax(&results[')
    || !runtimeSource.includes('_buildGlbCompactionShader()')) {
  throw new Error('The V4 runtime no longer performs candidate, render and GLB aggregation on the GPU.');
}
for (const forbidden of [
  'function intersectsComponent(',
  'function filterComponentsByFrustum(',
  'function componentIdsToGlbIds(',
  'worker-aabb-fallback',
  'this.device.queue.writeBuffer(this.candidateBuffer',
  'TrajectoryPrefetcher',
  '_scheduleTrajectoryPrefetch',
]) {
  if (runtimeSource.includes(forbidden) || workerSource.includes(forbidden) || loaderSource.includes(forbidden)) {
    throw new Error(`CPU neural-culling compatibility path returned: ${forbidden}`);
  }
}
if (!loaderSource.includes('predictionPayload.renderComponentModelList')
    || !loaderSource.includes('this._applyInstancedVisibility(renderComponentIds)')) {
  throw new Error('The main thread no longer applies the worker result at instance granularity.');
}

console.log(
  `Current frontend contract passed: ${meta.numInstances} instances, ${meta.numGlbs} GLBs, `
  + `threshold=${meta.threshold}, backOffset=${meta.query.candidateCameraBackOffsetM}m.`,
);
