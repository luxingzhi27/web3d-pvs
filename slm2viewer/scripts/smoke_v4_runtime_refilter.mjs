#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const viewerRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const publicRoot = path.join(viewerRoot, 'public');
const loadDiagnosticArg = process.argv.find((arg) => arg.startsWith('--load-diagnostic-ms='));
const loadDiagnosticMs = loadDiagnosticArg
  ? Math.max(0, Number(loadDiagnosticArg.split('=')[1]) || 0)
  : 0;
const forceWasm = process.argv.includes('--force-wasm');
const renderBackendArg = process.argv.find((arg) => arg.startsWith('--render-backend='));
const renderBackend = renderBackendArg ? renderBackendArg.split('=')[1] : 'webgl';
if (renderBackend !== 'webgl' && renderBackend !== 'webgpu') {
  throw new Error('--render-backend must be webgl or webgpu.');
}
const chromePath = [
  process.env.CHROME_PATH,
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
].find((candidate) => candidate && fs.existsSync(candidate));
if (!chromePath) throw new Error('Chrome/Chromium was not found.');
if (!fs.existsSync(path.join(publicRoot, 'index.html'))) {
  throw new Error('Run npm run build before the runtime refilter smoke.');
}

const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
};
const server = http.createServer((request, response) => {
  try {
    const url = new URL(request.url, 'http://127.0.0.1');
    const relative = decodeURIComponent(url.pathname).replace(/^\/+/, '') || 'index.html';
    const file = path.resolve(publicRoot, relative);
    if (!file.startsWith(`${publicRoot}${path.sep}`) && file !== publicRoot) throw new Error('Forbidden');
    response.writeHead(200, {
      'Content-Type': mimeTypes[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    fs.createReadStream(file).pipe(response);
  } catch {
    response.writeHead(404);
    response.end('Not found');
  }
});
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});

let browser;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: chromePath,
    args: [
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--ignore-gpu-blocklist',
      '--enable-gpu',
      '--enable-webgpu',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--use-vulkan',
      '--use-angle=vulkan',
      '--enable-accelerated-2d-canvas',
      '--enable-zero-copy',
      '--disable-software-rasterizer',
    ],
  });
  const page = await browser.newPage({ viewport: { width: 694, height: 552 } });
  const pageErrors = [];
  const consoleErrors = [];
  const rendererContractWarnings = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
    if (message.type() === 'warning'
        && /AttributeNode: Vertex attribute "uv" not found|arrayStride|Invalid ShaderModule|Invalid RenderPipeline/i.test(message.text())) {
      rendererContractWarnings.push(message.text());
    }
  });
  const port = server.address().port;
  const runtimeQuery = forceWasm ? '&neuralRuntimeBackend=wasm' : '';
  await page.goto(`http://127.0.0.1:${port}/?scene=hkust-v3&renderBackend=${renderBackend}${runtimeQuery}`, {
    waitUntil: 'domcontentloaded',
    timeout: 120000,
  });
  await page.waitForFunction(
    () => window.__slmApp?.viewer?.slm2Loader?.lastNeuralPrediction?.renderComponentModelList?.length > 0,
    null,
    { timeout: 180000 },
  );

  const result = await page.evaluate(async () => {
    const viewer = window.__slmApp.viewer;
    const loader = viewer.slm2Loader;
    const dispatcher = loader.neuralPVS;
    const full = loader.lastNeuralPrediction;
    const sorted = (values) => Array.from(values || []).sort((a, b) => a - b);
    const same = (a, b) => a.length === b.length && a.every((value, index) => value === b[index]);
    const difference = (values, excluded) => {
      const excludedSet = new Set(excluded);
      return values.filter((value) => !excludedSet.has(value));
    };
    viewer.controls.enabled = false;

    const fullInstances = sorted(full.renderComponentModelList);
    const fullGlbs = sorted(full.renderModelList);
    const initialBenchmark = loader.getBenchmarkVisibilityIds();
    const previousFilterSerial = Number(dispatcher.lastFilterTimings?.serial || 0);

    const movedCamera = loader.activeCamera.clone();
    movedCamera.position.x += 0.5;
    movedCamera.updateMatrixWorld(true);
    const aabbResponse = await fetch(
      './assets/neural_instance_culling/pvs_mainline_v4/instance_aabb_fp32.bin',
    );
    const aabbs = new Float32Array(await aabbResponse.arrayBuffer());
    loader._updateRuntimeVisibilityFrustum(movedCamera);
    const cpuReference = sorted(Array.from(full.componentModelList).filter((instanceId) => {
      const offset = instanceId * 6;
      for (const plane of loader._runtimeFrustum.planes) {
        const x = plane.normal.x >= 0 ? aabbs[offset + 3] : aabbs[offset];
        const y = plane.normal.y >= 0 ? aabbs[offset + 4] : aabbs[offset + 1];
        const z = plane.normal.z >= 0 ? aabbs[offset + 5] : aabbs[offset + 2];
        if (plane.normal.x * x + plane.normal.y * y + plane.normal.z * z + plane.constant < 0) {
          return false;
        }
      }
      return true;
    }));

    loader.activeCamera.copy(movedCamera);
    loader.activeCamera.updateMatrixWorld(true);
    loader.lastNeuralRefilter = null;
    loader.sceneCulling();
    const deadline = performance.now() + 30000;
    while (!loader.lastNeuralRefilter && performance.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    if (!loader.lastNeuralRefilter) throw new Error('Loader did not apply the cached refilter.');
    const benchmark = loader.getBenchmarkVisibilityIds();
    const movedInstances = sorted(benchmark.renderComponentIds);
    const movedGlbs = sorted(benchmark.renderGlbIds);
    const componentAdded = sorted(loader.lastNeuralRefilter.renderComponentAddedIds);
    const componentRemoved = sorted(loader.lastNeuralRefilter.renderComponentRemovedIds);
    const glbAdded = sorted(loader.lastNeuralRefilter.renderGlbAddedIds);
    const glbRemoved = sorted(loader.lastNeuralRefilter.renderGlbRemovedIds);
    const expectedComponentAdded = difference(movedInstances, fullInstances);
    const expectedComponentRemoved = difference(fullInstances, movedInstances);
    const expectedGlbAdded = difference(movedGlbs, fullGlbs);
    const expectedGlbRemoved = difference(fullGlbs, movedGlbs);
    const transferredIdCount = componentAdded.length + componentRemoved.length
      + glbAdded.length + glbRemoved.length;
    const rendererInfo = viewer.rendererRuntime.getInfo();
    const adapterInfo = rendererInfo.adapter || {};
    const gl = rendererInfo.backend === 'webgl2-fallback' ? viewer.renderer.getContext() : null;
    const glDebug = gl?.getExtension('WEBGL_debug_renderer_info');
    const denseStates = Object.values(loader.loadedInstancedVisibilityStatesByHash || {});
    const denseSlotsConsistent = denseStates.every((state) => state.disabled || (
      Array.isArray(state.activeSourceIndices)
      && state.activeSourceIndices.every((sourceIndex, slotIndex) => (
        state.slotBySourceIndex?.[sourceIndex] === slotIndex
      ))
      && state.meshStates.every((meshState) => meshState.mesh.count === state.activeSourceIndices.length)
    ));
    viewer._resizeRenderTargets();
    const nativePixelRatio = Number(window.devicePixelRatio || 1);
    const renderPixelRatio = Number(viewer.renderer.getPixelRatio() || 0);
    const cssWidth = Math.max(1, Math.round(viewer.renderer.domElement.clientWidth));
    const cssHeight = Math.max(1, Math.round(viewer.renderer.domElement.clientHeight));
    const runtimeAfterRefilter = loader.getRuntimeStats();
    return {
      adapter: {
        vendor: String(adapterInfo.vendor || ''),
        architecture: String(adapterInfo.architecture || ''),
        device: String(adapterInfo.device || ''),
        description: String(adapterInfo.description || ''),
      },
      rendererBackend: rendererInfo.backend,
      rendererFallbackReason: rendererInfo.fallbackReason,
      webglRenderer: glDebug
        ? String(gl.getParameter(glDebug.UNMASKED_RENDERER_WEBGL) || '')
        : '',
      sharedRendererDevice: Boolean(
        viewer.rendererRuntime.getSharedWebGPUContext()?.device
        && dispatcher.session?.pvs?.active?.device
          === viewer.rendererRuntime.getSharedWebGPUContext().device,
      ),
      initialGlbLoadOrderCount: loader.neuralInitialLoadOrder.length,
      initialGlbPreloadCount: Number(loader.startupMetrics.initialGlbPreloadCount || 0),
      fullInstanceCount: fullInstances.length,
      fullGlbCount: fullGlbs.length,
      initialLoaderInstanceParity: same(fullInstances, sorted(initialBenchmark.renderComponentIds)),
      initialLoaderGlbParity: same(fullGlbs, sorted(initialBenchmark.renderGlbIds)),
      movedInstanceCount: movedInstances.length,
      movedGlbCount: movedGlbs.length,
      movedCpuParity: same(movedInstances, cpuReference),
      componentDeltaParity: same(componentAdded, expectedComponentAdded)
        && same(componentRemoved, expectedComponentRemoved),
      glbDeltaParity: same(glbAdded, expectedGlbAdded)
        && same(glbRemoved, expectedGlbRemoved),
      readbackBytes: Number(loader.lastNeuralRefilter.timings?.readbackBytes || 0),
      transferredIdCount,
      reportedTransferredIdCount: Number(
        loader.lastNeuralRefilter.timings?.transferredIdCount || 0,
      ),
      promotedGlbCount: Number(
        loader.lastPvsSchedulerStats?.promotedGlbCount || 0,
      ),
      expectedPromotedGlbCount: glbAdded.length,
      renderComponentBitsetCount: loader.renderComponentState?.count,
      renderGlbBitsetCount: loader.renderGlbState?.count,
      noPersistentRenderIdArray: loader.lastRenderableComponentIdsForInstancing === undefined,
      denseSlotsConsistent,
      nativePixelRatio,
      renderPixelRatio,
      nativeDrawingBuffer: viewer.renderer.domElement.width === Math.floor(cssWidth * nativePixelRatio)
        && viewer.renderer.domElement.height === Math.floor(cssHeight * nativePixelRatio),
      fogAbsent: viewer.scene.fog == null,
      legacyQualityControlsAbsent: viewer.adaptiveResolution === undefined
        && viewer.gpuFrameTimer === undefined
        && viewer.distanceRendering === undefined,
      aoEnabled: viewer.rendererRuntime.effects?.options?.aoEnabled === true,
      filterSerialAdvanced: dispatcher.lastFilterTimings.serial > previousFilterSerial,
      filterInferenceMs: Number(dispatcher.lastFilterTimings.inferenceMs || 0),
      duplicateCullRemoved: loader.renderVisibilitySystem.lastStats?.duplicateCullRemoved === true,
      staticBatching: runtimeAfterRefilter.neural?.staticBatching || {},
      resourceSchedule: runtimeAfterRefilter.load?.resourceSchedule || {},
    };
  });

  const captureCanvasStats = async () => page.evaluate(async () => {
    const viewer = window.__slmApp.viewer;
    viewer._resizeRenderTargets();
    let renderError = null;
    try {
      viewer.render();
    } catch (error) {
      renderError = String(error?.stack || error);
    }
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
    const source = viewer.renderer.domElement;
    const width = source.width;
    const height = source.height;
    const copy = document.createElement('canvas');
    copy.width = width;
    copy.height = height;
    const context = copy.getContext('2d', { willReadFrequently: true });
    let pixels = new Uint8Array(width * height * 4);
    if (!renderError) {
      if (viewer.rendererRuntime.backend === 'webgl2-fallback') {
        const gl = viewer.renderer.getContext();
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      } else {
        context.drawImage(source, 0, 0);
        pixels = context.getImageData(0, 0, width, height).data;
      }
    }
    let sum = 0;
    let sumSquared = 0;
    let opaquePixels = 0;
    for (let offset = 0; offset < pixels.length; offset += 4) {
      const luminance = 0.2126 * pixels[offset]
        + 0.7152 * pixels[offset + 1]
        + 0.0722 * pixels[offset + 2];
      sum += luminance;
      sumSquared += luminance * luminance;
      if (pixels[offset + 3] > 0) opaquePixels += 1;
    }
    const count = Math.max(1, width * height);
    const meanLuminance = sum / count;
    return {
      width,
      height,
      meanLuminance,
      luminanceVariance: Math.max(0, sumSquared / count - meanLuminance * meanLuminance),
      opaqueRatio: opaquePixels / count,
      renderError,
      rendererInfo: viewer.rendererRuntime.getInfo(),
      aoEnabled: viewer.rendererRuntime.effects?.options?.aoEnabled === true,
    };
  });

  const desktopNative = await captureCanvasStats();
  await page.screenshot({ path: '/tmp/slm_v4_native_dpr_desktop.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.__slmApp.viewer.resize());
  const mobileNative = await captureCanvasStats();
  const mobileLayout = await page.evaluate(() => ({
    innerWidth: window.innerWidth,
    documentClientWidth: document.documentElement.clientWidth,
    documentScrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
    wrapWidth: document.querySelector('.wrap')?.getBoundingClientRect().width || 0,
    viewerWidth: document.querySelector('.viewer')?.getBoundingClientRect().width || 0,
    canvasWidth: window.__slmApp.viewer.renderer.domElement.getBoundingClientRect().width,
    guiWidth: document.querySelector('.gui-wrap')?.getBoundingClientRect().width || 0,
  }));
  await page.screenshot({ path: '/tmp/slm_v4_native_dpr_mobile.png' });
  result.canvasChecks = {
    desktopNative,
    mobileNative,
    mobileLayout,
  };
  if (desktopNative.renderError || mobileNative.renderError) {
    throw new Error(`WebGPU render failed: ${JSON.stringify({ pageErrors, result })}`);
  }

  await page.setViewportSize({ width: 694, height: 552 });
  await page.evaluate(() => {
    const viewer = window.__slmApp.viewer;
    viewer.resize();
    viewer._resizeRenderTargets();
  });
  result.runtimeDiagnostic = await page.evaluate(async () => {
    const viewer = window.__slmApp.viewer;
    const rendererInfo = viewer.rendererRuntime.getInfo();
    const runtime = viewer.slm2Loader.getRuntimeStats();
    viewer.renderer.info.reset();
    viewer.render();
    await new Promise((resolve) => setTimeout(resolve, 100));
    return {
      rendererBackend: rendererInfo.backend,
      adapter: rendererInfo.adapter,
      drawCalls: Number(viewer.renderer.info.render.calls || 0),
      triangles: Number(viewer.renderer.info.render.triangles || 0),
      actualRender: runtime.neural?.actualRender || {},
      staticBatching: runtime.neural?.staticBatching || {},
      load: runtime.load || {},
    };
  });
  if (loadDiagnosticMs > 0) {
    result.loadProgress = [];
    const loadStartedAt = Date.now();
    while (Date.now() - loadStartedAt < loadDiagnosticMs) {
      await page.waitForTimeout(Math.min(5000, loadDiagnosticMs - (Date.now() - loadStartedAt)));
      result.loadProgress.push(await page.evaluate(() => {
        const viewer = window.__slmApp.viewer;
        const runtime = viewer.slm2Loader.getRuntimeStats();
        return {
          elapsedMs: performance.now(),
          queueLength: Number(runtime.load?.queueLength || 0),
          prefetchQueueLength: Number(runtime.load?.prefetchQueueLength || 0),
          inflightCount: Number(runtime.load?.inflightCount || 0),
          pendingParseCount: Number(runtime.load?.pendingParseCount || 0),
          pendingSceneInsertions: Number(runtime.load?.pendingSceneInsertions || 0),
          totalIntegrated: Number(runtime.load?.totalIntegrated || 0),
          lastSuccess: runtime.load?.httpAdaptive?.lastSuccess !== false,
          visibleMeshCount: Number(runtime.neural?.actualRender?.visibleMeshCount || 0),
        };
      }));
    }
  }

  const validWebGPUFallback = renderBackend === 'webgpu'
    && result.rendererBackend === 'webgl2-fallback'
    && /WebGPU|adapter|swiftshader/i.test(String(result.rendererFallbackReason || ''));
  const passed = result.initialLoaderInstanceParity
    && (result.rendererBackend === (renderBackend === 'webgpu' ? 'webgpu' : 'webgl2-fallback')
      || validWebGPUFallback)
    && (result.rendererBackend !== 'webgl2-fallback' || (
      Boolean(result.webglRenderer)
      && !/swiftshader|llvmpipe|softpipe|swrast|software/i.test(result.webglRenderer)
    ))
    && (renderBackend !== 'webgpu' || forceWasm || result.sharedRendererDevice)
    && result.initialGlbLoadOrderCount === 100
    && result.initialGlbPreloadCount === 100
    && result.initialLoaderGlbParity
    && result.movedCpuParity
    && result.componentDeltaParity
    && result.glbDeltaParity
    && (renderBackend !== 'webgpu' || result.readbackBytes === 2776)
    && result.transferredIdCount === result.reportedTransferredIdCount
    && result.transferredIdCount < result.fullInstanceCount + result.fullGlbCount
    && result.promotedGlbCount === result.expectedPromotedGlbCount
    && result.resourceSchedule.urgentWanted === result.movedGlbCount
    && result.resourceSchedule.urgentMissing
      === result.resourceSchedule.urgentWanted - result.resourceSchedule.urgentResident
    && result.resourceSchedule.urgentFailed === 0
    && result.renderComponentBitsetCount === result.movedInstanceCount
    && result.renderGlbBitsetCount === result.movedGlbCount
    && result.noPersistentRenderIdArray
    && result.denseSlotsConsistent
    && Math.abs(result.renderPixelRatio - result.nativePixelRatio) < 1e-9
    && result.nativeDrawingBuffer
    && result.fogAbsent
    && result.legacyQualityControlsAbsent
    && result.aoEnabled
    && result.filterSerialAdvanced
    && result.filterInferenceMs === 0
    && result.duplicateCullRemoved
    && result.staticBatching.flattenedGlbs > 0
    && result.canvasChecks.desktopNative.luminanceVariance > 1
    && result.canvasChecks.mobileNative.luminanceVariance > 1
    && result.canvasChecks.desktopNative.opaqueRatio > 0.99
    && result.canvasChecks.mobileNative.opaqueRatio > 0.99
    && result.canvasChecks.desktopNative.aoEnabled
    && result.canvasChecks.mobileNative.aoEnabled
    && result.canvasChecks.mobileLayout.documentScrollWidth <= result.canvasChecks.mobileLayout.innerWidth;
  if (pageErrors.length || consoleErrors.length || rendererContractWarnings.length || !passed) {
    throw new Error(`Runtime refilter smoke failed: ${JSON.stringify({ pageErrors, consoleErrors, rendererContractWarnings, result })}`);
  }
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}
