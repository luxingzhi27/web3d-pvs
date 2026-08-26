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
  const loaderConfig = sceneConfig?.loaderConfig;
  const resourcesBaseUrl = loaderConfig?.resourcesBaseUrl;
  if (!resourcesBaseUrl) {
    throw new Error(`Scene ${sceneName} is missing loaderConfig.resourcesBaseUrl.`);
  }
  for (const legacyField of ['resourcesWS', 'remoteResourcesWS', 'rcServerAddress']) {
    if (Object.prototype.hasOwnProperty.call(loaderConfig, legacyField)) {
      throw new Error(`Scene ${sceneName} still declares removed loader field ${legacyField}.`);
    }
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
const initialGlbOrder = readJson('assets/scenes/hkust-v3/initialGlbLoadOrder.json');
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
if (Object.prototype.hasOwnProperty.call(glbIndex, 'initialLoadIds')) {
  throw new Error('glbIndex.json must remain a GLB mapping index, not an initial download queue.');
}
const initialLoadIds = Array.isArray(initialGlbOrder.initialLoadIds)
  ? initialGlbOrder.initialLoadIds.map(Number)
  : [];
if (initialGlbOrder.schemaVersion !== 1
    || initialGlbOrder.scene !== 'hkust-v3'
    || initialLoadIds.length === 0
    || Number(initialGlbOrder.visibleGlbCount) < initialLoadIds.length
    || Number(initialGlbOrder.initialLoadLimit) !== 100
    || Number(initialGlbOrder.initialLoadCount) !== 100
    || initialLoadIds.length !== 100
    || new Set(initialLoadIds).size !== initialLoadIds.length
    || initialLoadIds.some((id) => !Number.isInteger(id) || id < 0 || !glbIndex.entries[id])) {
  throw new Error('HKUST initial GLB download order is invalid.');
}
const hkustScene = config.scenes?.['hkust-v3'];
if (JSON.stringify(initialGlbOrder.camera?.position) !== JSON.stringify(hkustScene?.cameraPostion)
    || JSON.stringify(initialGlbOrder.camera?.target) !== JSON.stringify(hkustScene?.cameraTarget)
    || Number(initialGlbOrder.camera?.fovYDeg) !== 60
    || Math.abs(Number(initialGlbOrder.camera?.aspect) - 694 / 552) > 1e-12) {
  throw new Error('HKUST initial GLB order was not captured at the configured 60-degree startup camera.');
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
const renderVisibilitySource = fs.readFileSync(requireFile('src/RenderVisibilitySystem.js'), 'utf8');
const sortedIdDeltaSource = fs.readFileSync(requireFile('src/sortedIdDelta.js'), 'utf8');
const denseSlotsSource = fs.readFileSync(requireFile('src/DenseInstancedSlots.js'), 'utf8');
const bitsetStateSource = fs.readFileSync(requireFile('src/IdBitsetState.js'), 'utf8');
const staticSceneOptimizerSource = fs.readFileSync(requireFile('src/StaticSceneOptimizer.js'), 'utf8');
const renderSurfaceSource = fs.readFileSync(requireFile('src/RenderSurfacePolicy.js'), 'utf8');
const buildSource = fs.readFileSync(requireFile('scripts/build_parcel.mjs'), 'utf8');
const hkustGlbBaseUrl = 'https://www.liteweb3d.com/data/hkust-v3/';
if (config.scenes?.default_config?.loaderConfig?.glbResourcesBaseUrl !== hkustGlbBaseUrl
    || config.scenes?.['hkust-v3']?.loaderConfig?.glbResourcesBaseUrl !== hkustGlbBaseUrl) {
  throw new Error('All HKUST source entries must use the liteweb3d GLB origin.');
}
if (fs.existsSync(path.join(viewerDir, 'src/NeuralPVS.js'))) {
  throw new Error('The unused legacy ONNX visibility backend must not return.');
}
for (const forbidden of [
  'directional-occlusion-proxy-scheduler-v1',
  'camera-hash-latent-geo',
  'dynamic-occlusion-pool',
  'spatial-feature-pages',
  'RGBELoader',
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
    || !workerSource.includes('renderComponentModelList: renderComponentIds')
    || !workerSource.includes("message.type === 'filter'")) {
  throw new Error('The worker no longer follows the 60-degree render / 66-degree candidate contract.');
}
if (!runtimeSource.includes("source: 'gpu_v4_back_frustum_aabb'")
    || !runtimeSource.includes('fn intersects_frustum(instance_id: u32, render_frustum: bool)')
    || !runtimeSource.includes('atomicMax(&results[')
    || !runtimeSource.includes('_buildGlbCompactionShader()')
    || !runtimeSource.includes('_buildRenderFilterShader()')
    || !runtimeSource.includes('cached-render-filter')) {
  throw new Error('The V4 runtime no longer performs candidate, render and GLB aggregation on the GPU.');
}
if (!runtimeSource.includes('layout.instanceBitWords = Math.ceil(numInstances / 32)')
    || !runtimeSource.includes('layout.glbBitWords = Math.ceil(numGlbs / 32)')
    || !runtimeSource.includes('renderComponentBitset')
    || runtimeSource.includes('_buildRenderGlbCompactionShader()')
    || runtimeSource.includes('filterReadbackLayout')) {
  throw new Error('Cached 60-degree refiltering is not using the compact instance/GLB bitset readback.');
}
if (!workerSource.includes('diffIdBitsets(')
    || !workerSource.includes('renderComponentAddedIds: componentDelta.added')
    || !workerSource.includes('renderGlbRemovedIds: glbDelta.removed')
    || !sortedIdDeltaSource.includes('export function diffIdBitsets(')) {
  throw new Error('The worker no longer sends only the refilter ID delta.');
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
    || !loaderSource.includes('this._applyInstancedVisibility(renderComponentIds)')
    || !loaderSource.includes('_applyLightweightNeuralRefilter(')
    || !loaderSource.includes('applyDenseInstancedDelta(state, delta.added, delta.removed)')
    || !denseSlotsSource.includes('active[removedSlot] = lastSourceIndex')
    || !denseSlotsSource.includes('mesh.instanceMatrix.addUpdateRange(slot * 16, 16)')) {
  throw new Error('The main thread no longer applies the worker result at instance granularity.');
}
if (!loaderSource.includes('this.renderVisibilitySystem.applyDelta(')
    || !loaderSource.includes('this._applyInstancedVisibilityDelta(')
    || !loaderSource.includes('this._promoteCurrentRenderGlbs(glbAddedIds)')
    || !renderVisibilitySource.includes('applyDelta(addedModelInfos, removedModelInfos')
    || !loaderSource.includes('this.renderComponentState.applyDelta(added, removed)')
    || !loaderSource.includes('this.renderGlbState.applyDelta(glbAddedIds, glbRemovedIds)')
    || !bitsetStateSource.includes('toIds()')) {
  throw new Error('Cached refilter results are rebuilding full main-thread visibility sets.');
}
if (!staticSceneOptimizerSource.includes('new BatchedMesh(')
    || !staticSceneOptimizerSource.includes('batch.setVisibleAt(')
    || !staticSceneOptimizerSource.includes('mesh.matrixAutoUpdate = false')
    || !staticSceneOptimizerSource.includes('mesh.matrixWorldAutoUpdate = false')
    || !loaderSource.includes('this.staticSceneOptimizer = new StaticSceneOptimizer(this)')
    || !loaderSource.includes('scope.staticSceneOptimizer.optimizeResident(extras.hashCode, gltf)')
    || !loaderSource.includes('staticBatching: this.staticSceneOptimizer.getStats()')) {
  throw new Error('Static scene flattening or material/task BatchedMesh rendering was removed.');
}
const animateSource = viewerSource.match(/animate\(time\)[\s\S]*?\n  createQueueDebugPanel\(\)/)?.[0] || '';
if (!animateSource.includes('this.animationFrameId = null')
    || !animateSource.includes('this.slm2Loader.hasImmediateFrameWork()')
    || !animateSource.includes('const debugPanelExpanded = Boolean(this.gui && !this.gui.closed)')
    || animateSource.includes('requestAnimationFrame( this.animate )')
    || !loaderSource.includes('if (this.cameraUpdatePending && this.cullingUpdateDelta > 80)')) {
  throw new Error('Demand-driven rendering or event-driven camera culling was removed.');
}
for (const removedFile of [
  'src/AdaptiveRenderQuality.js',
  'src/AdaptiveResolutionController.js',
  'src/WebGLGpuFrameTimer.js',
]) {
  if (fs.existsSync(path.join(viewerDir, removedFile))) {
    throw new Error(`Removed render-quality implementation returned: ${removedFile}`);
  }
}
if (renderVisibilitySource.includes('item.meshObject.removeFromParent()')
    || viewerSource.includes('adaptiveResolution')
    || viewerSource.includes('gpuFrameTimer')
    || viewerSource.includes('sceneFog')
    || viewerSource.includes('distanceRendering')
    || viewerSource.includes('applyDistanceRendering')
    || viewerSource.includes('动态分辨率')
    || viewerSource.includes('雾化')
    || !viewerSource.includes('this.n8aopass.enabled = true')
    || !viewerSource.includes('this.smaaPass.enabled = true')) {
  throw new Error('Dynamic resolution or fog returned to the frontend runtime.');
}
if (!viewerSource.includes("window.addEventListener('resize', this.scheduleResize")
    || !viewerSource.includes('this.renderer.setDrawingBufferSize(')
    || !viewerSource.includes('sameRenderSurface(this.renderSurface, nextSurface)')
    || renderSurfaceSource.includes('maxDrawingBufferPixels')
    || renderSurfaceSource.includes('resolutionScale')
    || !renderSurfaceSource.includes('finitePositive(options.devicePixelRatio, 1)')
    || !viewerSource.includes('devicePixelRatio: window.devicePixelRatio')) {
  throw new Error('Fixed native-DPR render-surface sizing was removed.');
}
if (Object.values(config.scenes).some((sceneConfig) => (
  Object.prototype.hasOwnProperty.call(sceneConfig || {}, 'distanceRendering')
))) {
  throw new Error('Fog and distance-rendering configuration must stay removed.');
}
if (!loaderSource.includes('new AbortController()')
    || !loaderSource.includes('fetch(modelURL, {')
    || !loaderSource.includes('pendingGlbParseQueue.push({')
    || !loaderSource.includes('loader.parse(item.buffer')
    || !loaderSource.includes('pendingSceneInsertions.push({')) {
  throw new Error('GLB download, parse and scene insertion are no longer separate bounded stages.');
}
for (const forbidden of [
  '_startDirectModelLoad(',
  'neuralDisableLoadLimits',
  'processNeuralWSParseQueue',
  'pendingWSParse',
  'activeWSParse',
  'gltfLoaders[capturedLoaderIdx].load(',
  'startConnectAsset(',
  'getRVCServerUrl(',
  'fetchCameraVisibilityList(',
  'loadStatic(',
  'rvcServer',
  'resourcesWS',
  'lastRenderableComponentIdsForInstancing',
  'lastBenchmarkVisibilityIds.renderComponentIds',
  'applySortedIdDelta',
]) {
  if (loaderSource.includes(forbidden)) {
    throw new Error(`Replaced GLB loading path remains in the frontend: ${forbidden}`);
  }
}
const initialOrderLoadIndex = loaderSource.indexOf('scope._loadInitialGlbLoadOrder(function()');
const initialPreloadIndex = loaderSource.indexOf('scope._startNeuralInitialGlbPreload();', initialOrderLoadIndex);
const runtimeMetaStartIndex = loaderSource.indexOf('scope._scheduleRuntimeVisibilityMetaLoad(function()', initialPreloadIndex);
if (initialOrderLoadIndex < 0 || initialPreloadIndex < initialOrderLoadIndex
    || runtimeMetaStartIndex < initialPreloadIndex
    || !loaderSource.includes('const INITIAL_GLB_PRELOAD_LIMIT = 100;')
    || !loaderSource.includes('.slice(0, INITIAL_GLB_PRELOAD_LIMIT)')
    || !loaderSource.includes('this.processLoadingList();')
    || loaderSource.includes('parsedIndex.initialLoadIds')
    || loaderSource.includes('payload[1] && payload[1].initialLoadIds')) {
  throw new Error('Initial GLB requests must start from the independent queue before model initialization.');
}
for (const forbidden of [
  '_projectedAreaForHash(',
  '_preciseProjectedAreaForBox(',
  '_fastProjectedAreaForBox(',
  'areaRejected++',
  'state.lastKey',
  'mesh.setMatrixAt(visibleIndex',
  'computeBoundingSphere()',
  'neuralRenderRetainMs',
]) {
  if (renderVisibilitySource.includes(forbidden) || loaderSource.includes(forbidden)) {
    throw new Error(`Duplicate CPU culling or full instanced update returned: ${forbidden}`);
  }
}
if (!loaderSource.includes('var imageConfigUrl = joinUrlPath(')
    || !loaderSource.includes('scope.glbResourcesBaseUrl || scope.resourcesBaseUrl')
    || !loaderSource.includes("'task-' + item.groupId + '/images/LOD'")) {
  throw new Error('Material metadata or textures no longer use the configured scene resource origin.');
}
if (!buildSource.includes('cache: false')
    || !buildSource.includes("source.includes('__parcel__error__overlay__')")) {
  throw new Error('Production builds are no longer protected from Parcel HMR cache contamination.');
}
if (!viewerSource.includes('ids.renderComponentIds')
    || !viewerSource.includes('ids.renderGlbIds')
    || !viewerSource.includes("ids.mode === 'neural' && currentMode === 'neural'")
    || !viewerSource.includes('startFrozenPredictionInspectSession(\n        this.predictionDebugFrozenComponentIds,\n        this.predictionDebugFrozenGlbIds')) {
  throw new Error('Frozen inspection no longer captures the final render-frustum instance and GLB sets.');
}
if (!loaderSource.includes('this.frozenPredictionInspectComponentIds = this._normalizeIdList(componentIds)')
    || !loaderSource.includes('this.frozenPredictionInspectGlbIds = this._normalizeIdList(glbIds)')
    || !loaderSource.includes('this._applyInstancedVisibility(this.frozenPredictionInspectComponentIds)')
    || !renderVisibilitySource.includes('duplicateCullRemoved: true')) {
  throw new Error('Frozen inspection no longer preserves its instance-level visibility snapshot.');
}
const frozenSceneBranch = loaderSource.match(/else if \(this\.frozenPredictionInspectActive\)[\s\S]*?\r?\n    else\r?\n    \{/);
if (!frozenSceneBranch
    || frozenSceneBranch[0].includes('_showAllResidentObjects()')
    || viewerSource.includes('_getRawPredictionDebugComponentIds')
    || viewerSource.includes('freeze all')
    || viewerSource.includes('download/show all GLBs')) {
  throw new Error('Frozen inspection returned to raw back-frustum or all-resident rendering.');
}

console.log(
  `Current frontend contract passed: ${meta.numInstances} instances, ${meta.numGlbs} GLBs, `
  + `threshold=${meta.threshold}, backOffset=${meta.query.candidateCameraBackOffsetM}m.`,
);
