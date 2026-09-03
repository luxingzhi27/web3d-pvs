#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtimeSchema = 'pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4';
const modelRoot = 'assets/neural_instance_culling/pvs_mainline_v4';

function file(relativePath) {
  const absolute = path.join(root, relativePath);
  if (!fs.existsSync(absolute)) throw new Error(`Missing required file: ${relativePath}`);
  return absolute;
}

function source(relativePath) {
  return fs.readFileSync(file(relativePath), 'utf8');
}

function json(relativePath) {
  return JSON.parse(source(relativePath));
}

function requireText(text, tokens, label) {
  for (const token of tokens) {
    if (!text.includes(token)) throw new Error(`${label} is missing contract: ${token}`);
  }
}

const config = json('assets/config.json');
const meta = json(`${modelRoot}/model_meta.json`);
const runtimeMeta = json('assets/scenes/hkust-v3/runtimeVisibilityMeta.json');
const glbIndex = json('assets/scenes/hkust-v3/glbIndex.json');
const initialOrder = json('assets/scenes/hkust-v3/initialGlbLoadOrder.json');

for (const relativePath of [
  `${modelRoot}/instance_runtime_features_fp16.bin`,
  `${modelRoot}/instance_aabb_fp32.bin`,
  `${modelRoot}/instance_to_glb_uint32.bin`,
  `${modelRoot}/query_weights_fp16.bin`,
  `${modelRoot}/frequency_cycles_fp32.bin`,
  `${modelRoot}/chi_table_fp32.bin`,
  'assets/wasm/instance_pvs_v4.wasm',
]) file(relativePath);

if (meta.schema !== runtimeSchema
    || Number(meta.modelConfig?.runtimeFeatureDim) !== 124
    || Number(meta.modelConfig?.runtimeHeadInputDim) !== 130
    || meta.safety?.safe !== true
    || !Number.isFinite(Number(meta.threshold))) {
  throw new Error('HKUST frontend model is not the frozen safe V4 runtime.');
}
if (Number(meta.query?.modelInputFovYDeg) !== 66
    || Number(meta.query?.frontendRenderFovYDeg) !== 60
    || !Number.isFinite(Number(meta.query?.candidateCameraBackOffsetM))) {
  throw new Error('The 66-degree candidate / 60-degree render camera contract is incomplete.');
}
if (Number(meta.numInstances) !== Number(runtimeMeta.instanceCount)
    || Number(meta.numGlbs) !== Number(runtimeMeta.globalGlbCount)
    || Number(meta.numGlbs) !== Number(glbIndex.total)) {
  throw new Error('Model, visibility metadata and GLB index counts do not agree.');
}

const initialIds = Array.isArray(initialOrder.initialLoadIds)
  ? initialOrder.initialLoadIds.map(Number)
  : [];
if (initialOrder.schemaVersion !== 1
    || initialOrder.scene !== 'hkust-v3'
    || initialIds.length !== 100
    || new Set(initialIds).size !== initialIds.length
    || initialIds.some((id) => !Number.isInteger(id) || id < 0 || !glbIndex.entries[id])) {
  throw new Error('The independent 100-GLB startup queue is invalid.');
}

const hkustGlbBaseUrl = 'https://www.liteweb3d.com/data/hkust-v3/';
for (const name of ['default_config', 'hkust-v3']) {
  if (config.scenes?.[name]?.loaderConfig?.glbResourcesBaseUrl !== hkustGlbBaseUrl) {
    throw new Error(`Scene ${name} does not use the configured HKUST GLB origin.`);
  }
}

const packageJson = json('package.json');
for (const dependency of ['n8ao', 'postprocessing', '@monogrid/gainmap-js']) {
  if (packageJson.dependencies?.[dependency]) throw new Error(`Removed WebGL dependency returned: ${dependency}`);
}
requireText(JSON.stringify(packageJson.alias || {}), ['three.webgpu.js', 'three.tsl.js'], 'Parcel Three.js alias');

for (const removedPath of [
  'src/LightweightPVSDispatcher.js',
  'src/LightweightPVSWorker.js',
  'src/InstancePVSCPU.js',
  'src/InstancePVSCPUMath.js',
  'lib/three-vignette.js',
  'lib/three-vignette_1.js',
]) {
  if (fs.existsSync(path.join(root, removedPath))) throw new Error(`Removed runtime returned: ${removedPath}`);
}

const renderer = source('src/RendererRuntime.js');
const effects = source('src/RendererEffects.js');
const background = source('src/SceneBackground.js');
const dispatcher = source('src/PVSDispatcher.js');
const session = source('src/PVSQuerySession.js');
const worker = source('src/PVSWorker.js');
const runtime = source('src/InstancePVS.js');
const wasm = source('src/InstancePVSWasm.js');
const loader = source('slm2/SLM2Loader.js');
const viewer = source('src/viewer.js');

requireText(renderer, [
  'new WebGPURenderer({',
  'forceWebGL: true',
  'getSharedWebGPUContext()',
  'new RendererEffects(',
], 'RendererRuntime');
requireText(effects, [
  'new RenderPipeline(',
  'pass(this.scene, this.camera, { samples: 0 })',
  'ao(this.scenePass.getTextureNode',
  'pow(this.aoNode.getTextureNode().r.clamp(0, 1), this.aoStrength)',
  'fxaa(outputNode)',
], 'TSL post-processing');
requireText(background, ['viewportUV.y', 'smoothstep(', 'mix('], 'TSL background');
requireText(dispatcher, [
  "backendPreference: 'webgpu'",
  "new Worker('./PVSWorker.js')",
  "backendPreference !== 'auto'",
], 'PVS dispatcher');
requireText(session, [
  'renderer-shared',
  'MODEL_INPUT_FOV_Y_DEG',
  'FRONTEND_RENDER_FOV_Y_DEG',
  'candidateCameraBackOffsetM',
  'renderComponentAddedIds',
  'renderGlbRemovedIds',
], 'PVS query session');
requireText(worker, ["backendPreference: 'wasm'", 'session.predict(', 'session.refilter('], 'PVS worker');
requireText(loader, [
  "import { PVSDispatcher }",
  'externalWebGPUContext: this.options.pvsWebGPUContext || null',
  'this.glbResourceScheduler.setPlan({',
  'this.glbResourceScheduler.takeNext()',
  'applyDenseInstancedDelta(',
], 'SLM2 loader');
requireText(viewer, [
  'this.rendererRuntime.render(',
  'this.rendererRuntime.configureSurface(',
  'pvsWebGPUContext: scope.rendererRuntime.getSharedWebGPUContext()',
], 'Viewer');
requireText(runtime, [
  "source: 'gpu_v4_back_frustum_aabb'",
  '_buildGlbCompactionShader()',
  '_buildRenderFilterShader()',
  'sharedRendererDevice:',
], 'WebGPU PVS runtime');
requireText(wasm, ["this.backend = 'wasm-simd-v4'", 'this.wasm.predict_v4('], 'WASM PVS runtime');

for (const forbidden of [
  'EffectComposer',
  'N8AOPostPass',
  'WebGLRenderer',
  'HDRJPGLoader',
  'cpu-js-v4',
]) {
  if ([viewer, loader, dispatcher, worker].some((text) => text.includes(forbidden))) {
    throw new Error(`Old frontend path remains: ${forbidden}`);
  }
}

console.log(
  `Current frontend contract passed: WebGPU/WebGL2 renderer, shared-device/Web Worker PVS, `
  + `${meta.numInstances} instances and ${meta.numGlbs} GLBs.`,
);
