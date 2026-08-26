#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

function argument(name, fallback = null) {
  const prefix = `--${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

const allowSoftware = process.argv.includes('--allow-software-gpu');
if (!allowSoftware) {
  throw new Error('Initial-order capture requires explicit --allow-software-gpu for functional WebGPU capture.');
}

const pageUrl = argument('url', 'http://127.0.0.1:3001/?scene=hkust-v3');
const initialLoadLimit = Math.max(1, Number.parseInt(argument('limit', '100'), 10) || 100);
const viewportWidth = Math.max(1, Number.parseInt(argument('viewport-width', '1280'), 10) || 1280);
const viewportHeight = Math.max(1, Number.parseInt(argument('viewport-height', '720'), 10) || 720);
const outputPath = path.resolve(argument(
  'out',
  'assets/scenes/hkust-v3/initialGlbLoadOrder.json',
));
const chromePath = [
  process.env.CHROME_PATH,
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
].find((candidate) => candidate && fs.existsSync(candidate));
if (!chromePath) throw new Error('Chrome/Chromium was not found.');

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
  const page = await browser.newPage({
    viewport: { width: viewportWidth, height: viewportHeight },
  });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  await page.goto(pageUrl, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForFunction(
    () => window.__slmApp?.viewer?.slm2Loader?.lastNeuralPrediction?.renderModelList?.length > 0,
    null,
    { timeout: 180000 },
  );

  const capture = await page.evaluate(async () => {
    const viewer = window.__slmApp.viewer;
    const loader = viewer.slm2Loader;
    const prediction = loader.lastNeuralPrediction;
    const scoreByGlb = new Map();
    const predictedIds = Array.from(prediction.modelList || []);
    const predictedWeights = Array.from(prediction.weightList || []);
    for (let index = 0; index < predictedIds.length; index += 1) {
      scoreByGlb.set(Number(predictedIds[index]), Number(predictedWeights[index] || 0));
    }
    const renderIds = Array.from(prediction.renderModelList || []).map(Number);
    renderIds.sort((a, b) => (scoreByGlb.get(b) || 0) - (scoreByGlb.get(a) || 0) || a - b);
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const adapterInfo = adapter?.info || {};
    return {
      cameraPosition: viewer.activeCamera.position.toArray(),
      cameraTarget: viewer.controls.target.toArray(),
      fovYDeg: Number(viewer.activeCamera.fov),
      aspect: Number(viewer.activeCamera.aspect),
      renderInstanceCount: Number(prediction.renderComponentModelList?.length || 0),
      initialLoadIds: renderIds,
      adapter: {
        vendor: String(adapterInfo.vendor || ''),
        architecture: String(adapterInfo.architecture || ''),
        device: String(adapterInfo.device || ''),
        description: String(adapterInfo.description || ''),
      },
    };
  });
  if (pageErrors.length) {
    throw new Error(`Page errors during initial-order capture: ${pageErrors.join('; ')}`);
  }
  if (new Set(capture.initialLoadIds).size !== capture.initialLoadIds.length) {
    throw new Error('Captured initial GLB order contains duplicate IDs.');
  }

  const payload = {
    schemaVersion: 1,
    scene: 'hkust-v3',
    prioritySource: 'pvs-v4-max-instance-visibility-probability',
    visibilitySet: 'model-visible-and-real-60deg-frustum',
    camera: {
      position: capture.cameraPosition,
      target: capture.cameraTarget,
      fovYDeg: capture.fovYDeg,
      aspect: capture.aspect,
    },
    visibleInstanceCount: capture.renderInstanceCount,
    visibleGlbCount: capture.initialLoadIds.length,
    initialLoadLimit,
    initialLoadCount: Math.min(initialLoadLimit, capture.initialLoadIds.length),
    initialLoadIds: capture.initialLoadIds.slice(0, initialLoadLimit),
  };
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`);
  console.log(JSON.stringify({ outputPath, adapter: capture.adapter, ...payload }, null, 2));
} finally {
  await browser?.close();
}
