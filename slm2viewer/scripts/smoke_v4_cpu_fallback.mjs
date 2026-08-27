#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const viewerRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const viewerDirArg = process.argv.find((value) => value.startsWith('--viewer-dir='));
const publicRoot = viewerDirArg
  ? path.resolve(viewerDirArg.slice('--viewer-dir='.length))
  : path.join(viewerRoot, 'public');
const chromePath = [
  process.env.CHROME_PATH,
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
].find((candidate) => candidate && fs.existsSync(candidate));
if (!chromePath) throw new Error('Chrome/Chromium was not found.');
if (!fs.existsSync(path.join(publicRoot, 'index.html'))) {
  throw new Error('Run npm run build before the CPU fallback smoke.');
}

const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.glb': 'model/gltf-binary',
  '.wasm': 'application/wasm',
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
      '--enable-webgl',
      '--use-angle=vulkan',
    ],
  });
  const port = server.address().port;

  async function runCase(name, query, hideWebGPU) {
    const context = await browser.newContext({ viewport: { width: 694, height: 552 } });
    if (hideWebGPU) {
      await context.addInitScript(() => {
        Object.defineProperty(Navigator.prototype, 'gpu', {
          configurable: true,
          get: () => undefined,
        });
      });
    }
    const page = await context.newPage();
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(String(error)));
    await page.route('https://www.liteweb3d.com/data/hkust-v3/**', (route) => route.abort());
    await page.goto(`http://127.0.0.1:${port}/?scene=hkust-v3&showGUI=false&${query}`, {
      waitUntil: 'domcontentloaded',
      timeout: 120000,
    });
    await page.waitForFunction(
      () => window.__slmApp?.viewer?.slm2Loader?.lastNeuralPrediction?.componentModelList?.length > 0,
      null,
      { timeout: 180000 },
    );
    const result = await page.evaluate(async () => {
      const loader = window.__slmApp.viewer.slm2Loader;
      const dispatcher = loader.neuralPVS;
      const camera = loader.activeCamera;
      let timerTicks = 0;
      const timer = setInterval(() => { timerTicks += 1; }, 10);
      const prediction = await dispatcher.predict(camera);
      clearInterval(timer);
      const filtered = await dispatcher.refilter(camera);
      return {
        backend: prediction.backend,
        dispatcherBackend: dispatcher.backend,
        fallbackReason: prediction.fallbackReason || dispatcher.lastInitTimings?.fallbackReason || null,
        candidateCount: prediction.candidateCount,
        visibleInstanceCount: prediction.componentModelList.length,
        visibleGlbCount: prediction.modelList.length,
        renderInstanceCount: prediction.renderComponentModelList.length,
        refilterInstanceCount: filtered.renderInstanceCount,
        inferenceMs: Number(prediction.timings?.inferenceMs || 0),
        timerTicks,
      };
    });
    assert.match(result.backend, /worker-cpu-js-v4/);
    assert.doesNotMatch(result.backend, /aabb/i);
    assert.ok(result.candidateCount > result.visibleInstanceCount);
    assert.ok(result.visibleInstanceCount > 0);
    assert.ok(result.visibleGlbCount > 0);
    assert.equal(result.refilterInstanceCount, result.renderInstanceCount);
    assert.ok(result.timerTicks >= 2, 'CPU neural inference blocked the browser main thread.');
    assert.deepEqual(pageErrors, []);
    if (hideWebGPU) assert.ok(result.fallbackReason);
    await context.close();
    return { name, ...result };
  }

  const forced = await runCase('forced-cpu', 'neuralRuntimeBackend=cpu', false);
  const automatic = await runCase('auto-without-webgpu', 'neuralRuntimeBackend=auto', true);
  console.log(JSON.stringify({ forced, automatic }, null, 2));
  console.log('V4 Worker CPU fallback browser smoke passed.');
} finally {
  if (browser) await browser.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
}
