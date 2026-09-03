#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

if (!process.argv.includes('--allow-software-gpu')) {
  throw new Error('This functional smoke requires the explicit --allow-software-gpu flag.');
}

const viewerRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const publicRoot = path.join(viewerRoot, 'public');
const loadDiagnosticArg = process.argv.find((arg) => arg.startsWith('--load-diagnostic-ms='));
const loadDiagnosticMs = loadDiagnosticArg
  ? Math.max(0, Number(loadDiagnosticArg.split('=')[1]) || 0)
  : 0;
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
    ],
  });
  const page = await browser.newPage({ viewport: { width: 694, height: 552 } });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  const port = server.address().port;
  await page.goto(`http://127.0.0.1:${port}/?scene=hkust-v3`, {
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
    const fullSerial = dispatcher.lastPredictTimings.serial;
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
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const adapterInfo = adapter?.info || {};
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
    const gl = viewer.renderer.getContext();
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
        loader.lastLightweightPVSSchedulerStats?.promotedGlbCount || 0,
      ),
      expectedPromotedGlbCount: glbAdded.length,
      renderComponentBitsetCount: loader.renderComponentState?.count,
      renderGlbBitsetCount: loader.renderGlbState?.count,
      noPersistentRenderIdArray: loader.lastRenderableComponentIdsForInstancing === undefined,
      denseSlotsConsistent,
      nativePixelRatio,
      renderPixelRatio,
      nativeDrawingBuffer: gl.drawingBufferWidth === Math.floor(cssWidth * nativePixelRatio)
        && gl.drawingBufferHeight === Math.floor(cssHeight * nativePixelRatio),
      fogAbsent: viewer.scene.fog == null,
      legacyQualityControlsAbsent: viewer.adaptiveResolution === undefined
        && viewer.gpuFrameTimer === undefined
        && viewer.distanceRendering === undefined,
      aoEnabled: viewer.n8aopass?.enabled === true,
      smaaEnabled: viewer.smaaPass?.enabled === true,
      fullPredictSerialUnchanged: dispatcher.lastPredictTimings.serial === fullSerial,
      filterSerialAdvanced: dispatcher.lastFilterTimings.serial > previousFilterSerial,
      filterInferenceMs: Number(dispatcher.lastFilterTimings.inferenceMs || 0),
      duplicateCullRemoved: loader.renderVisibilitySystem.lastStats?.duplicateCullRemoved === true,
      staticBatching: runtimeAfterRefilter.neural?.staticBatching || {},
      resourceSchedule: runtimeAfterRefilter.load?.resourceSchedule || {},
    };
  });

  const captureCanvasStats = async () => page.evaluate(() => {
    const viewer = window.__slmApp.viewer;
    viewer._resizeRenderTargets();
    viewer.render();
    const gl = viewer.renderer.getContext();
    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
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
      aoEnabled: viewer.n8aopass?.enabled === true,
      smaaEnabled: viewer.smaaPass?.enabled === true,
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

  await page.setViewportSize({ width: 694, height: 552 });
  await page.evaluate(() => {
    const viewer = window.__slmApp.viewer;
    viewer.resize();
    viewer._resizeRenderTargets();
  });
  result.runtimeDiagnostic = await page.evaluate(() => new Promise((resolve) => {
    const viewer = window.__slmApp.viewer;
    const frameIntervals = [];
    let previous = performance.now();
    const startedAt = previous;
    const sample = (now) => {
      frameIntervals.push(now - previous);
      previous = now;
      if (now - startedAt < 2000) {
        requestAnimationFrame(sample);
        return;
      }
      const sortedIntervals = frameIntervals.slice().sort((a, b) => a - b);
      const percentile = (ratio) => sortedIntervals[Math.min(
        sortedIntervals.length - 1,
        Math.floor(sortedIntervals.length * ratio),
      )] || 0;
      const gl = viewer.renderer.getContext();
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      const runtime = viewer.slm2Loader.getRuntimeStats();
      const previousAutoReset = viewer.renderer.info.autoReset;
      viewer.renderer.info.autoReset = false;
      viewer.renderer.info.reset();
      viewer.render();
      const drawCalls = Number(viewer.renderer.info.render.calls || 0);
      const triangles = Number(viewer.renderer.info.render.triangles || 0);
      viewer.renderer.info.autoReset = previousAutoReset;
      viewer.renderer.info.reset();
      resolve({
        frameCount: frameIntervals.length,
        avgFrameMs: frameIntervals.reduce((sum, value) => sum + value, 0)
          / Math.max(1, frameIntervals.length),
        p95FrameMs: percentile(0.95),
        webglRenderer: debugInfo
          ? String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || '')
          : '',
        drawCalls,
        triangles,
        actualRender: runtime.neural?.actualRender || {},
        staticBatching: runtime.neural?.staticBatching || {},
        load: runtime.load || {},
      });
    };
    requestAnimationFrame(sample);
  }));
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

  const passed = result.initialLoaderInstanceParity
    && result.initialGlbLoadOrderCount === 100
    && result.initialGlbPreloadCount === 100
    && result.initialLoaderGlbParity
    && result.movedCpuParity
    && result.componentDeltaParity
    && result.glbDeltaParity
    && result.readbackBytes === 2776
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
    && result.smaaEnabled
    && result.fullPredictSerialUnchanged
    && result.filterSerialAdvanced
    && result.filterInferenceMs === 0
    && result.duplicateCullRemoved
    && result.staticBatching.flattenedGlbs > 0
    && result.canvasChecks.desktopNative.luminanceVariance > 1
    && result.canvasChecks.mobileNative.luminanceVariance > 1
    && result.canvasChecks.desktopNative.opaqueRatio > 0.99
    && result.canvasChecks.mobileNative.opaqueRatio > 0.99
    && result.canvasChecks.desktopNative.aoEnabled
    && result.canvasChecks.desktopNative.smaaEnabled
    && result.canvasChecks.mobileNative.aoEnabled
    && result.canvasChecks.mobileNative.smaaEnabled
    && result.canvasChecks.mobileLayout.documentScrollWidth <= result.canvasChecks.mobileLayout.innerWidth;
  if (pageErrors.length || !passed) {
    throw new Error(`Runtime refilter smoke failed: ${JSON.stringify({ pageErrors, result })}`);
  }
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}
