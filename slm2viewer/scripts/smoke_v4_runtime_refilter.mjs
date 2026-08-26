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
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
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
    viewer.controls.enabled = false;

    const fullInstances = sorted(full.renderComponentModelList);
    const fullGlbs = sorted(full.renderModelList);
    const sameCamera = await dispatcher.refilter(loader.activeCamera.clone());
    const fullSerial = dispatcher.lastPredictTimings.serial;
    const previousFilterSerial = dispatcher.lastFilterTimings.serial;

    const movedCamera = loader.activeCamera.clone();
    movedCamera.position.x += 0.5;
    movedCamera.updateMatrixWorld(true);
    const moved = await dispatcher.refilter(movedCamera.clone());
    const movedInstances = sorted(moved.renderComponentModelList);
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
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const adapterInfo = adapter?.info || {};
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
      sameCameraInstanceParity: same(fullInstances, sorted(sameCamera.renderComponentModelList)),
      sameCameraGlbParity: same(fullGlbs, sorted(sameCamera.renderModelList)),
      movedInstanceCount: movedInstances.length,
      movedCpuParity: same(movedInstances, cpuReference),
      loaderParity: same(movedInstances, sorted(benchmark.renderComponentIds)),
      fullPredictSerialUnchanged: dispatcher.lastPredictTimings.serial === fullSerial,
      filterSerialAdvanced: dispatcher.lastFilterTimings.serial > previousFilterSerial,
      filterInferenceMs: dispatcher.lastFilterTimings.inferenceMs,
      duplicateCullRemoved: loader.renderVisibilitySystem.lastStats?.duplicateCullRemoved === true,
    };
  });

  const passed = result.sameCameraInstanceParity
    && result.initialGlbLoadOrderCount === 100
    && result.initialGlbPreloadCount === 100
    && result.sameCameraGlbParity
    && result.movedCpuParity
    && result.loaderParity
    && result.fullPredictSerialUnchanged
    && result.filterSerialAdvanced
    && result.filterInferenceMs === 0
    && result.duplicateCullRemoved;
  if (pageErrors.length || !passed) {
    throw new Error(`Runtime refilter smoke failed: ${JSON.stringify({ pageErrors, result })}`);
  }
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}
