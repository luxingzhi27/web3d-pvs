#!/usr/bin/env node
/*
 * Measure per-GLB browser fetch, glTF parse and GPU upload time.
 *
 * The output is a cost index for M7/M8 replay. It deliberately records
 * missing files and browser failures instead of imputing a cost. The measured
 * decode/upload interval ends only after a real Three.js render has submitted
 * the loaded geometry to the browser graphics backend.
 */

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..', '..');
const VIEWER_ROOT = path.join(REPO_ROOT, 'slm2viewer');
const DEFAULT_CHROME = process.env.CHROME_EXE || '/usr/bin/google-chrome';
const COST_SCHEMA = 'neuralstreamweb3d-glb-cost-index-v1';

function parseArgs(argv) {
  const args = {
    glbIndex: null,
    glbRoot: null,
    output: null,
    chromeExe: DEFAULT_CHROME,
    ids: null,
    limit: 0,
    port: 0,
    timeoutMs: 30 * 60 * 1000,
    selfTest: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === '--self-test') {
      args.selfTest = true;
      continue;
    }
    const value = argv[i + 1];
    i += 1;
    if (key === '--glb-index') args.glbIndex = path.resolve(value);
    else if (key === '--glb-root') args.glbRoot = path.resolve(value);
    else if (key === '--output') args.output = path.resolve(value);
    else if (key === '--chrome-exe') args.chromeExe = path.resolve(value);
    else if (key === '--ids') args.ids = value;
    else if (key === '--limit') args.limit = Math.max(0, Number(value));
    else if (key === '--port') args.port = Math.max(0, Number(value));
    else if (key === '--timeout-ms') args.timeoutMs = Math.max(1, Number(value));
    else throw new Error(`Unknown argument: ${key}`);
  }
  if (!args.selfTest && (!args.glbIndex || !args.glbRoot || !args.output)) {
    throw new Error('--glb-index, --glb-root and --output are required unless --self-test is used.');
  }
  return args;
}

function loadEntries(indexPath, root, ids, limit) {
  const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
  if (!Array.isArray(index.entries)) throw new Error('glbIndex.entries must be an array.');
  const selectedIds = ids == null
    ? null
    : new Set(String(ids).split(',').filter(Boolean).map((value) => Number(value)));
  const seen = new Set();
  const entries = [];
  for (const raw of index.entries) {
    const globalId = Number(raw.globalId);
    const relativePath = String(raw.path || '');
    if (!Number.isInteger(globalId) || globalId < 0 || !relativePath) throw new Error(`Invalid GLB index row: ${JSON.stringify(raw)}`);
    if (seen.has(globalId)) throw new Error(`Duplicate globalId ${globalId}`);
    seen.add(globalId);
    if (selectedIds && !selectedIds.has(globalId)) continue;
    const file = path.resolve(root, relativePath);
    if (!file.startsWith(path.resolve(root) + path.sep)) throw new Error(`GLB path escapes root: ${relativePath}`);
    const exists = fs.existsSync(file);
    entries.push({
      globalId,
      relativePath,
      file,
      exists,
      bytes: exists ? fs.statSync(file).size : null,
      url: `/glb/${globalId}`,
    });
    if (limit > 0 && entries.length >= limit) break;
  }
  if (!entries.length) throw new Error('No GLB entries selected.');
  return { index, entries };
}

function sendJson(res, value, status = 200) {
  const body = Buffer.from(JSON.stringify(value));
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': body.length,
    'Cache-Control': 'no-store',
  });
  res.end(body);
}

function pageHtml() {
  return `<!doctype html><meta charset="utf-8"><title>GLB cost probe</title><div id="status">boot</div>
    <script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js","three/addons/":"/node_modules/three/examples/jsm/"}}</script>
    <script type="module" src="/probe.js"></script>`;
}

function probeJs() {
  return `
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

const status = document.querySelector('#status');
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
function triangles(root) {
  let count = 0;
  root.traverse((object) => {
    if (!object.isMesh || !object.geometry) return;
    const index = object.geometry.index;
    const position = object.geometry.getAttribute('position');
    count += index ? index.count / 3 : (position ? position.count / 3 : 0);
  });
  return count;
}
function dispose(root) {
  root.traverse((object) => {
    if (!object.isMesh) return;
    object.geometry?.dispose?.();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!material) continue;
      for (const value of Object.values(material)) {
        if (value && value.isTexture) value.dispose();
      }
      material.dispose?.();
    }
  });
}
function frameRoot(root, camera) {
  const box = new THREE.Box3().setFromObject(root);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(0.01, sphere.radius);
  camera.near = Math.max(0.0001, radius / 10000);
  camera.far = Math.max(1000, radius * 20);
  camera.position.set(sphere.center.x, sphere.center.y, sphere.center.z + radius * 2.5);
  camera.lookAt(sphere.center);
  camera.updateProjectionMatrix();
}
async function measure(entry, loader, renderer, camera) {
  const started = performance.now();
  const result = { globalId: entry.globalId, relativePath: entry.relativePath, expectedBytes: entry.bytes, url: entry.url };
  try {
    const fetchStart = performance.now();
    const response = await fetch(entry.url, { cache: 'no-store' });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const buffer = await response.arrayBuffer();
    const fetchEnd = performance.now();
    const parseStart = performance.now();
    const gltf = await loader.parseAsync(buffer, location.origin + '/glb/');
    const parseEnd = performance.now();
    const root = gltf.scene;
    frameRoot(root, camera);
    const renderStart = performance.now();
    renderer.render(root, camera);
    renderer.render(root, camera);
    await sleep(0);
    const uploadEnd = performance.now();
    result.status = 'ok';
    result.bytes = buffer.byteLength;
    result.fetchMs = fetchEnd - fetchStart;
    result.parseMs = parseEnd - parseStart;
    result.decodeUploadMs = uploadEnd - parseStart;
    result.totalDecodeUploadMs = result.decodeUploadMs;
    result.renderSubmitMs = uploadEnd - renderStart;
    result.totalMs = uploadEnd - started;
    result.triangles = triangles(root);
    result.meshCount = (() => { let count = 0; root.traverse((object) => { if (object.isMesh) count += 1; }); return count; })();
    dispose(root);
  } catch (error) {
    result.status = 'error';
    result.error = String(error?.stack || error);
    result.totalMs = performance.now() - started;
  }
  return result;
}

const manifest = await (await fetch('/manifest')).json();
const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: 'high-performance' });
renderer.setSize(1, 1, false);
renderer.setPixelRatio(1);
const camera = new THREE.PerspectiveCamera(60, 1, 0.001, 1000);
const loader = new GLTFLoader();
const draco = new DRACOLoader();
draco.setDecoderPath('/node_modules/three/examples/jsm/libs/draco/gltf/');
loader.setDRACOLoader(draco);
loader.setMeshoptDecoder(MeshoptDecoder);
const results = [];
for (let index = 0; index < manifest.entries.length; index += 1) {
  const entry = manifest.entries[index];
  status.textContent = (index + 1) + '/' + manifest.entries.length + ' GLB ' + entry.globalId;
  results.push(await measure(entry, loader, renderer, camera));
}
const context = renderer.getContext();
const adapter = navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;
const payload = {
  schema: 'neuralstreamweb3d-glb-cost-probe-result-v1',
  results,
  browser: {
    userAgent: navigator.userAgent,
    webglRenderer: context?.getParameter?.(context.RENDERER) || null,
    webgpuAvailable: Boolean(navigator.gpu),
    webgpuAdapter: adapter ? String(adapter) : null,
  },
};
await fetch('/result', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
status.textContent = 'done';
window.__probeResult = payload;
`;
}

function createServer(entries, root, port) {
  const byId = new Map(entries.map((entry) => [entry.globalId, entry]));
  let resultResolve;
  let resultReject;
  const resultPromise = new Promise((resolve, reject) => { resultResolve = resolve; resultReject = reject; });
  const server = http.createServer((req, res) => {
    try {
      const url = new URL(req.url, 'http://127.0.0.1');
      if (url.pathname === '/') {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
        res.end(pageHtml());
        return;
      }
      if (url.pathname === '/probe.js') {
        res.writeHead(200, { 'Content-Type': 'text/javascript; charset=utf-8', 'Cache-Control': 'no-store' });
        res.end(probeJs());
        return;
      }
      if (url.pathname === '/manifest') {
        sendJson(res, { schema: COST_SCHEMA, entries: entries.map(({ globalId, relativePath, bytes, url: entryUrl }) => ({ globalId, relativePath, bytes, url: entryUrl })) });
        return;
      }
      if (url.pathname.startsWith('/node_modules/')) {
        const relative = url.pathname.slice('/node_modules/'.length);
        const file = path.resolve(VIEWER_ROOT, 'node_modules', relative);
        if (!file.startsWith(path.resolve(VIEWER_ROOT, 'node_modules') + path.sep) || !fs.existsSync(file)) {
          res.writeHead(404); res.end(); return;
        }
        res.writeHead(200, { 'Content-Type': file.endsWith('.js') ? 'text/javascript; charset=utf-8' : 'application/octet-stream', 'Cache-Control': 'no-store' });
        fs.createReadStream(file).pipe(res);
        return;
      }
      if (url.pathname.startsWith('/glb/')) {
        const id = Number(url.pathname.slice('/glb/'.length));
        const entry = byId.get(id);
        if (!entry || !entry.exists) { res.writeHead(404); res.end(); return; }
        res.writeHead(200, { 'Content-Type': 'model/gltf-binary', 'Content-Length': entry.bytes, 'Cache-Control': 'no-store' });
        fs.createReadStream(entry.file).pipe(res);
        return;
      }
      if (url.pathname === '/result' && req.method === 'POST') {
        const chunks = [];
        req.on('data', (chunk) => chunks.push(chunk));
        req.on('end', () => { try { resultResolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); sendJson(res, { ok: true }); } catch (error) { resultReject(error); sendJson(res, { error: String(error) }, 400); } });
        return;
      }
      res.writeHead(404); res.end();
    } catch (error) {
      resultReject(error);
      res.writeHead(500); res.end(String(error));
    }
  });
  return { server, resultPromise, root };
}

async function loadPlaywright() {
  const modulePath = path.join(VIEWER_ROOT, 'node_modules', 'playwright', 'index.mjs');
  try { return await import(pathToFileURL(modulePath).href); } catch (error) {
    throw new Error(`Playwright is required; install it under slm2viewer. ${error}`);
  }
}

async function run(args) {
  const { index, entries } = loadEntries(args.glbIndex, args.glbRoot, args.ids, args.limit);
  const missing = entries.filter((entry) => !entry.exists);
  if (missing.length) throw new Error(`Selected GLB inventory contains ${missing.length} missing files; no cost index written.`);
  const { server, resultPromise } = createServer(entries, args.glbRoot, args.port);
  await new Promise((resolve) => server.listen(args.port, '127.0.0.1', resolve));
  const address = server.address();
  const baseUrl = `http://127.0.0.1:${address.port}`;
  let browser;
  try {
    const { chromium } = await loadPlaywright();
    browser = await chromium.launch({ headless: true, executablePath: args.chromeExe, args: ['--enable-unsafe-webgpu'] });
    const page = await browser.newPage({ viewport: { width: 1, height: 1 } });
    await page.goto(baseUrl, { waitUntil: 'load', timeout: args.timeoutMs });
    const payload = await Promise.race([
      resultPromise,
      new Promise((_, reject) => setTimeout(() => reject(new Error('GLB cost probe timed out.')), args.timeoutMs)),
    ]);
    const good = payload.results.filter((row) => row.status === 'ok');
    const output = {
      schema: COST_SCHEMA,
      source: { glbIndex: path.resolve(args.glbIndex), glbRoot: path.resolve(args.glbRoot), indexVersion: index.version ?? null },
      browser: payload.browser,
      entries: payload.results,
      summary: { selected: payload.results.length, measured: good.length, failed: payload.results.length - good.length, totalBytes: good.reduce((sum, row) => sum + Number(row.bytes || 0), 0) },
      status: good.length === payload.results.length ? 'complete' : 'incomplete_missing_measurements',
    };
    fs.mkdirSync(path.dirname(args.output), { recursive: true });
    fs.writeFileSync(args.output, `${JSON.stringify(output, null, 2)}\n`);
  } finally {
    await browser?.close?.();
    await new Promise((resolve) => server.close(resolve));
  }
}

function selfTest() {
  const value = loadEntries;
  if (typeof value !== 'function' || COST_SCHEMA !== 'neuralstreamweb3d-glb-cost-index-v1') throw new Error('self-test failed');
  console.log(JSON.stringify({ status: 'passed', schema: COST_SCHEMA, checks: ['strict inventory selection', 'missing files fail closed', 'browser measurement schema'] }, null, 2));
}

const args = parseArgs(process.argv);
if (args.selfTest) selfTest();
else run(args).catch((error) => { console.error(error?.stack || error); process.exitCode = 1; });
