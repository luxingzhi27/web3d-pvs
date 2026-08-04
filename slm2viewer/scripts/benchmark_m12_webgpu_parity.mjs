#!/usr/bin/env node
/*
 * Run the M12 probe through the production Worker/WebGPU path.
 * The probe is intentionally separate from the normal browser audit: it sends
 * preselected cameras and instance IDs to the opt-in benchmark method and
 * stores the raw WGSL logits for comparison with the PyTorch reference.
 */
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_DIR = path.resolve(SCRIPT_DIR, '..');
const DEFAULT_PORT = 3137;

function parseArgs(argv) {
  const options = {
    cases: null,
    out: null,
    port: DEFAULT_PORT,
    url: null,
    startServer: false,
    headed: false,
    debugStages: false,
    executablePath: process.env.CHROME_PATH || process.env.CHROMIUM_PATH || null,
    timeoutMs: 180000,
    // Formal parity must prove the browser selected a hardware adapter.  The
    // software opt-out is retained only for historical semantic debugging.
    requireHardwareGpu: true,
  };
  const valueOptions = new Set(['cases', 'out', 'port', 'url', 'executable-path', 'timeout-ms']);
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--start-server') { options.startServer = true; continue; }
    if (token === '--headed') { options.headed = true; continue; }
    if (token === '--debug-stages') { options.debugStages = true; continue; }
    if (token === '--require-hardware-gpu') { options.requireHardwareGpu = true; continue; }
    if (token === '--allow-software-gpu') { options.requireHardwareGpu = false; continue; }
    if (token === '--help' || token === '-h') { options.help = true; continue; }
    const equal = token.indexOf('=');
    const name = token.slice(0, equal >= 0 ? equal : undefined).replace(/^--/, '');
    if (!valueOptions.has(name)) throw new Error(`Unknown argument: ${token}`);
    const value = equal >= 0 ? token.slice(equal + 1) : argv[++i];
    if (value == null || String(value).startsWith('--')) throw new Error(`Missing value for --${name}`);
    if (name === 'cases') options.cases = path.resolve(String(value));
    else if (name === 'out') options.out = path.resolve(String(value));
    else if (name === 'port') options.port = Number(value);
    else if (name === 'url') options.url = String(value);
    else if (name === 'executable-path') options.executablePath = path.resolve(String(value));
    else if (name === 'timeout-ms') options.timeoutMs = Number(value);
  }
  if (!options.cases || !options.out) throw new Error('--cases and --out are required');
  if (!options.url) options.url = `http://127.0.0.1:${options.port}/?scene=hkust-v3&culling=neural`;
  return options;
}

function usage() {
  return 'node scripts/benchmark_m12_webgpu_parity.mjs --start-server --cases PATH --out PATH [--require-hardware-gpu|--allow-software-gpu]';
}

function classifyGpuBackend(adapterInfo, webglBackend) {
  const adapter = adapterInfo && typeof adapterInfo === 'object' ? adapterInfo : {};
  const webgl = webglBackend && typeof webglBackend === 'object' ? webglBackend : {};
  const text = [
    adapter.vendor,
    adapter.architecture,
    adapter.device,
    adapter.description,
    webgl.vendor,
    webgl.renderer,
    webgl.version,
  ].filter(Boolean).join(' ');
  const softwarePattern = /swiftshader|llvmpipe|softpipe|swrast|software(?:\s+webgpu|\s+webgl|\s+rasterizer)?|no-webgpu|no-webgl/i;
  return {
    hardware: Boolean(text) && !softwarePattern.test(text),
    softwareMarkers: text.match(softwarePattern)?.[0] || null,
    evidenceText: text,
  };
}

function loadPlaywright() {
  try { return import('playwright'); } catch (_) { return import('playwright-core'); }
}

function findChromiumExecutable(explicit) {
  const candidates = [
    explicit,
    process.env.CHROME_PATH,
    process.env.CHROMIUM_PATH,
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);
  return candidates.find((candidate) => fs.existsSync(candidate)) || null;
}

function waitForHttp(url, timeoutMs = 30000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const request = http.get(url, (response) => {
        response.resume();
        if (response.statusCode && response.statusCode < 500) { resolve(); return; }
        retry();
      });
      request.on('error', retry);
      request.setTimeout(1000, () => { request.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() - start > timeoutMs) reject(new Error(`Timed out waiting for ${url}`));
      else setTimeout(attempt, 250);
    };
    attempt();
  });
}

async function waitFor(page, predicate, timeoutMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = await page.evaluate(predicate);
    if (value && value.ready) return value;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return page.evaluate(predicate);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) { console.log(usage()); return; }
  const casesPayload = JSON.parse(fs.readFileSync(options.cases, 'utf8'));
  const cases = Array.isArray(casesPayload.cases) ? casesPayload.cases : [];
  if (!cases.length) throw new Error('M12 cases file contains no cases');

  let server = null;
  let browser = null;
  try {
    if (options.startServer) {
      server = spawn('npm', ['run', 'dev', '--', '--port', String(options.port)], {
        cwd: VIEWER_DIR,
        detached: true,
        stdio: ['ignore', 'ignore', 'ignore'],
      });
      await waitForHttp(`http://127.0.0.1:${options.port}/`, 30000);
    }
    const playwright = await loadPlaywright();
    const executablePath = findChromiumExecutable(options.executablePath);
    const launchOptions = {
      headless: !options.headed,
      ...(executablePath ? { executablePath } : {}),
      args: [
        '--enable-unsafe-webgpu',
        '--enable-gpu',
        '--enable-webgpu',
        '--enable-webgl',
        '--use-angle=vulkan',
        '--enable-accelerated-2d-canvas',
        '--enable-zero-copy',
        '--ignore-gpu-blocklist',
        ...(options.requireHardwareGpu ? ['--disable-software-rasterizer'] : []),
        '--disable-gpu-sandbox',
        '--no-sandbox',
      ],
    };
    browser = await playwright.chromium.launch(launchOptions);
    const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
    const page = await context.newPage();
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push({ message: error.message, stack: error.stack || null }));
    await page.goto(options.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    const ready = await waitFor(page, () => {
      const dispatcher = window.__slmApp?.viewer?.slm2Loader?.neuralPVS;
      return dispatcher ? {
        ready: Boolean(dispatcher.isReady),
        backend: dispatcher.backend,
        modelInfo: dispatcher.modelInfo,
        initError: dispatcher.initError ? String(dispatcher.initError.message || dispatcher.initError) : null,
      } : null;
    }, options.timeoutMs);
    const result = await page.evaluate(async ({ probeCases, debugStages }) => {
      const dispatcher = window.__slmApp?.viewer?.slm2Loader?.neuralPVS;
      if (!dispatcher || typeof dispatcher.benchmarkM12 !== 'function') {
        throw new Error('The loaded frontend does not expose the M12 probe method. Rebuild the source bundle.');
      }
      const probe = await dispatcher.benchmarkM12(probeCases, { debugStages: debugStages });
      let adapterInfo = null;
      try {
        const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
        adapterInfo = adapter?.info ? {
          vendor: adapter.info.vendor || null,
          architecture: adapter.info.architecture || null,
          device: adapter.info.device || null,
          description: adapter.info.description || null,
        } : null;
      } catch (_) { /* adapter metadata is optional */ }
      let webglBackend = null;
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        const extension = gl?.getExtension('WEBGL_debug_renderer_info');
        webglBackend = gl ? {
          vendor: extension ? gl.getParameter(extension.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
          renderer: extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
          version: gl.getParameter(gl.VERSION),
        } : null;
      } catch (_) { /* WebGL evidence is optional for the WebGPU parity probe. */ }
      return { probe, adapterInfo, webglBackend };
    }, { probeCases: cases, debugStages: options.debugStages });
    const gpuGate = {
      required: options.requireHardwareGpu,
      ...classifyGpuBackend(result.adapterInfo, result.webglBackend),
    };
    const payload = {
      schema: 'm12-webgpu-parity-capture-v1',
      cases: options.cases,
      capturedAt: new Date().toISOString(),
      browser: {
        version: browser.version(),
        executablePath: executablePath || 'playwright-managed',
        headless: !options.headed,
        launchArgs: launchOptions.args,
        requireHardwareGpu: options.requireHardwareGpu,
      },
      frontendReady: ready,
      adapterInfo: result.adapterInfo,
      webglBackend: result.webglBackend,
      gpuGate,
      pageErrors,
      debugStages: options.debugStages,
      probe: result.probe,
    };
    fs.mkdirSync(path.dirname(options.out), { recursive: true });
    fs.writeFileSync(options.out, JSON.stringify(payload, null, 2) + '\n');
    console.log(JSON.stringify({ output: options.out, backend: result.probe?.backend, cases: result.probe?.results?.length || 0, pageErrors: pageErrors.length }, null, 2));
    if (options.requireHardwareGpu && !gpuGate.hardware) {
      console.error(`[m12-webgpu-parity] hardware GPU required, evidence=${gpuGate.evidenceText || 'none'}`);
      process.exitCode = 2;
    }
    if (String(result.probe?.backend || '') !== 'webgpu') process.exitCode = 2;
  } finally {
    if (browser) await browser.close().catch(() => {});
    if (server) {
      try { process.kill(-server.pid, 'SIGTERM'); } catch (_) { try { server.kill('SIGTERM'); } catch (__) {} }
    }
  }
}

main().catch((error) => {
  console.error(`[m12-webgpu-parity] ${error.stack || error.message || error}`);
  process.exitCode = 1;
});
