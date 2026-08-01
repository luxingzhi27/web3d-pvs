#!/usr/bin/env node
/*
 * M9 browser runtime and Cold-0 request audit.
 *
 * This script is deliberately external to the viewer runtime. It observes the
 * existing page through Playwright, so a report cannot silently make the
 * production scheduler faster or change its visibility semantics.
 */

import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_DIR = path.resolve(SCRIPT_DIR, '..');
const SCRIPT_VERSION = 'm9-browser-runtime-audit-v2';
const DEFAULT_SCENE = 'hkust-v3';
const DEFAULT_PORT = 3000;
const DEFAULT_DURATION_MS = 15000;
const DEFAULT_POLL_MS = 100;
const DEFAULT_SETTLE_MS = 3000;
const DEFAULT_NAVIGATION_TIMEOUT_MS = 30000;
const DEFAULT_MAX_EVENTS = 100000;
const MAX_TARGET_GLB_BODY_PROBES = 8;
const MAX_TARGET_GLB_BODY_PROBE_BYTES = 8 * 1024 * 1024;

const MODEL_BY_SCENE = Object.freeze({
  'hkust-v3': 'pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best',
  'ifcbench_fantasy_metropolis_instanced_v2':
    'pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best',
});

function nowMs() {
  return Number(process.hrtime.bigint()) / 1e6;
}

function isoNow() {
  return new Date().toISOString();
}

function asBool(value, fallback = false) {
  if (value == null) return fallback;
  if (typeof value === 'boolean') return value;
  return !['0', 'false', 'no', 'off'].includes(String(value).toLowerCase());
}

function asInt(value, fallback, min = -Infinity, max = Infinity) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(min, Math.min(max, Math.trunc(number)));
}

function parseArgs(argv) {
  const options = {
    selfTest: false,
    url: null,
    scene: DEFAULT_SCENE,
    startServer: false,
    serverCommand: null,
    serverCwd: VIEWER_DIR,
    port: DEFAULT_PORT,
    durationMs: DEFAULT_DURATION_MS,
    pollMs: DEFAULT_POLL_MS,
    settleMs: DEFAULT_SETTLE_MS,
    navigationTimeoutMs: DEFAULT_NAVIGATION_TIMEOUT_MS,
    waitFor: 'prediction',
    headed: false,
    executablePath: process.env.CHROME_PATH || process.env.CHROMIUM_PATH || null,
    viewport: { width: 1280, height: 720 },
    out: null,
    jsonlOut: null,
    maxEvents: DEFAULT_MAX_EVENTS,
    glbRegex: null,
    workerEvaluate: true,
    debugQuery: true,
    help: false,
  };

  const valueOptions = new Set([
    'url', 'scene', 'server-command', 'server-cwd', 'port', 'duration-ms', 'poll-ms', 'settle-ms',
    'navigation-timeout-ms', 'wait-for', 'executable-path', 'viewport', 'out',
    'jsonl-out', 'max-events', 'glb-regex',
  ]);

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') {
      options.help = true;
      continue;
    }
    if (token === '--self-test') {
      options.selfTest = true;
      continue;
    }
    if (token === '--start-server') {
      options.startServer = true;
      continue;
    }
    if (token === '--headed') {
      options.headed = true;
      continue;
    }
    if (token === '--no-worker-evaluate') {
      options.workerEvaluate = false;
      continue;
    }
    if (token === '--no-debug-query') {
      options.debugQuery = false;
      continue;
    }

    const equalIndex = token.indexOf('=');
    const rawName = equalIndex >= 0 ? token.slice(0, equalIndex) : token;
    const name = rawName.replace(/^--/, '');
    if (!valueOptions.has(name)) {
      throw new Error(`Unknown argument: ${token}`);
    }
    const value = equalIndex >= 0 ? token.slice(equalIndex + 1) : argv[++index];
    if (value == null || String(value).startsWith('--')) {
      throw new Error(`Missing value for --${name}`);
    }
    switch (name) {
      case 'url': options.url = String(value); break;
      case 'scene': options.scene = String(value); break;
      case 'server-command': options.serverCommand = String(value); break;
      case 'server-cwd': options.serverCwd = path.resolve(String(value)); break;
      case 'port': options.port = asInt(value, DEFAULT_PORT, 1, 65535); break;
      case 'duration-ms': options.durationMs = asInt(value, DEFAULT_DURATION_MS, 0, 3600000); break;
      case 'poll-ms': options.pollMs = asInt(value, DEFAULT_POLL_MS, 10, 10000); break;
      case 'settle-ms': options.settleMs = asInt(value, DEFAULT_SETTLE_MS, 0, 60000); break;
      case 'navigation-timeout-ms': options.navigationTimeoutMs = asInt(value, DEFAULT_NAVIGATION_TIMEOUT_MS, 1000, 300000); break;
      case 'wait-for': options.waitFor = String(value); break;
      case 'executable-path': options.executablePath = path.resolve(String(value)); break;
      case 'viewport': options.viewport = parseViewport(String(value)); break;
      case 'out': options.out = path.resolve(String(value)); break;
      case 'jsonl-out': options.jsonlOut = path.resolve(String(value)); break;
      case 'max-events': options.maxEvents = asInt(value, DEFAULT_MAX_EVENTS, 100, 1000000); break;
      case 'glb-regex': options.glbRegex = String(value); break;
      default: break;
    }
  }

  if (!['prediction', 'webgpu', 'none'].includes(options.waitFor)) {
    throw new Error(`--wait-for must be prediction, webgpu, or none; got ${options.waitFor}`);
  }
  if (!options.url && !options.startServer) {
    options.url = `http://127.0.0.1:${options.port}/?scene=${encodeURIComponent(options.scene)}`;
  }
  return options;
}

function parseViewport(value) {
  const match = String(value).match(/^(\d+)x(\d+)$/i);
  if (!match) throw new Error(`Invalid --viewport ${value}; expected WIDTHxHEIGHT`);
  return { width: Number(match[1]), height: Number(match[2]) };
}

function usage() {
  return `Usage:
  node scripts/benchmark_webgpu_runtime.mjs --self-test
  node scripts/benchmark_webgpu_runtime.mjs --url http://127.0.0.1:3000/?scene=hkust-v3 \\
    --duration-ms 15000 --out /tmp/m9_hkust.json
  node scripts/benchmark_webgpu_runtime.mjs --start-server --scene hkust-v3 \\
    --duration-ms 15000 --out /tmp/m9_local.json

Options:
  --self-test                  Run offline parser/classifier/report checks only.
  --url URL                    Existing viewer URL. No server is started implicitly.
  --scene NAME                 Scene used by the default local URL.
  --start-server               Start the local Parcel server on PORT and stop it afterward.
  --server-command COMMAND     Override the command used with --start-server.
  --server-cwd DIR             Working directory for the local server.
  --port PORT                  Local server port (default: 3000).
  --duration-ms N              Observation duration after navigation (default: 15000).
  --settle-ms N                Continue observing after the required event (default: 3000).
  --wait-for prediction|webgpu|none
  --headed                     Show Chromium instead of running headless.
  --executable-path PATH       Chromium/Chrome executable.
  --viewport WIDTHxHEIGHT      Browser viewport (default: 1280x720).
  --out FILE                   Write one JSON report.
  --jsonl-out FILE             Write the event stream as JSON Lines.
  --glb-regex REGEX            Override target-GLB URL classification.
  --no-worker-evaluate         Do not attempt Playwright injection into workers.
  --no-debug-query              Do not append neuralDebugLogs/debugNeural query flags.
`;
}

function safeUrl(value, base = null) {
  try {
    return new URL(String(value), base || undefined).toString();
  } catch {
    return String(value ?? '');
  }
}

function addQuery(url, key, value) {
  const parsed = new URL(url);
  if (!parsed.searchParams.has(key)) parsed.searchParams.set(key, value);
  return parsed.toString();
}

function buildAuditUrl(options) {
  let url = options.url;
  if (!url) url = `http://127.0.0.1:${options.port}/?scene=${encodeURIComponent(options.scene)}`;
  url = safeUrl(url);
  if (!new URL(url).searchParams.has('scene') && options.scene) url = addQuery(url, 'scene', options.scene);
  if (options.debugQuery) {
    url = addQuery(url, 'culling', 'neural');
    url = addQuery(url, 'neuralDebugLogs', 'true');
    url = addQuery(url, 'debugNeural', 'true');
  }
  return url;
}

function classifyResourceUrl(rawUrl, overrideRegex = null) {
  const url = String(rawUrl || '');
  const lower = url.toLowerCase();
  let targetGlb = false;
  if (overrideRegex) {
    try {
      targetGlb = new RegExp(overrideRegex, 'i').test(url);
    } catch (error) {
      throw new Error(`Invalid --glb-regex: ${error.message}`);
    }
  } else {
    const isGlb = /\.glb(?:[?#]|$)/i.test(lower);
    const isProxy = /(?:\/|^)proxy(?:\/|$)/i.test(lower) || /\/proxy\.glb(?:[?#]|$)/i.test(lower);
    const isSceneGeometry = /\/scene_glbs\//i.test(lower) ||
      /\/glb\/(?:lod\d+|raw)\//i.test(lower) ||
      /\/sub_[^/]+\.glb(?:[?#]|$)/i.test(lower);
    targetGlb = isGlb && isSceneGeometry && !isProxy;
  }
  if (targetGlb) return 'target-glb';
  if (/\.glb(?:[?#]|$)/i.test(lower)) return 'scene-proxy-or-other-glb';
  if (/instance_pvs_assets.*\.bin(?:[?#]|$)/i.test(lower)) return 'neural-weight';
  if (/runtimevisibilitymeta\.json(?:[?#]|$)/i.test(lower)) return 'runtime-visibility-meta';
  if (/glbindex\.json(?:[?#]|$)/i.test(lower)) return 'glb-index';
  if (/sceneweb\.json(?:[?#]|$)/i.test(lower)) return 'scene-web';
  if (/config\.json(?:[?#]|$)/i.test(lower)) return 'viewer-config';
  if (/\.json(?:[?#]|$)/i.test(lower)) return 'json';
  if (/\.wasm(?:[?#]|$)/i.test(lower)) return 'wasm';
  if (/\.hdr|\.jpg|\.jpeg|\.png|\.ktx2|\.basis/i.test(lower)) return 'texture-or-environment';
  if (/\.js(?:[?#]|$)/i.test(lower)) return 'script';
  if (/\.css(?:[?#]|$)/i.test(lower)) return 'stylesheet';
  return 'other';
}

function createStaticAudit(scene) {
  const sceneRoot = path.join(VIEWER_DIR, 'assets', 'scenes', scene);
  const modelName = MODEL_BY_SCENE[scene] || null;
  const modelRoot = modelName
    ? path.join(VIEWER_DIR, 'assets', 'neural_instance_culling', modelName)
    : null;
  const files = [
    ['viewer index', path.join(VIEWER_DIR, 'index.html')],
    ['viewer config', path.join(VIEWER_DIR, 'assets', 'config.json')],
    ['sceneWeb', path.join(sceneRoot, 'sceneWeb.json')],
    ['glbIndex', path.join(sceneRoot, 'glbIndex.json')],
    ['runtimeVisibilityMeta', path.join(sceneRoot, 'runtimeVisibilityMeta.json')],
    ['model meta', modelRoot && path.join(modelRoot, 'instance_model_meta.json')],
    ['model weights', modelRoot && path.join(modelRoot, 'instance_pvs_assets.bin')],
  ];
  const checks = files.map(([label, file]) => ({
    label,
    path: file,
    exists: Boolean(file && fs.existsSync(file)),
    bytes: file && fs.existsSync(file) ? fs.statSync(file).size : 0,
  }));
  const sourceGeometryRoot = path.join(sceneRoot, 'task-0', 'glb');
  const configuredTargetRoot = path.join(VIEWER_DIR, 'scene_glbs', scene);
  return {
    scene,
    modelName,
    checks,
    sourceGeometryRoot: {
      path: sourceGeometryRoot,
      exists: fs.existsSync(sourceGeometryRoot),
      note: 'Source assets may exist here, but the current config requests glbResourcesBaseUrl.'
    },
    configuredTargetRoot: {
      path: configuredTargetRoot,
      exists: fs.existsSync(configuredTargetRoot),
      note: 'This is the local path corresponding to the default ./scene_glbs URL.'
    },
    allRequiredLocalFilesPresent: checks.every((check) => check.exists),
    targetGlbDeploymentPresentForDefaultLocalServer: fs.existsSync(configuredTargetRoot),
  };
}

function makeBrowserInitScript(maxEvents) {
  return `(() => {
  const MAX_EVENTS = ${Math.max(100, Number(maxEvents) || DEFAULT_MAX_EVENTS)};
  const root = window.__m9RuntimeAudit = window.__m9RuntimeAudit || {
    schemaVersion: 'm9-browser-runtime-event-v1', events: [], sequence: 0,
    droppedEvents: 0, workerIds: 0,
  };
  const nativeDate = Date;
  const emit = (name, details = {}) => {
    if (root.events.length >= MAX_EVENTS) { root.droppedEvents += 1; return null; }
    const event = {
      seq: ++root.sequence,
      name,
      browserTimeMs: typeof performance !== 'undefined' && performance.now ? performance.now() : null,
      wallTime: new nativeDate().toISOString(),
      details: details && typeof details === 'object' ? details : { value: details },
    };
    root.events.push(event);
    return event;
  };
  root.emit = emit;
  root.classifyResource = (url) => {
    const value = String(url || '');
    const lower = value.toLowerCase();
    const isProxy = /(?:\\/|^)proxy(?:\\/|$)/i.test(lower) || /\\/proxy\\.glb(?:[?#]|$)/i.test(lower);
    const isGeometry = /\\/scene_glbs\\//i.test(lower) || /\\/glb\\/(?:lod\\d+|raw)\\//i.test(lower) || /\\/sub_[^/]+\\.glb(?:[?#]|$)/i.test(lower);
    if (/\\.glb(?:[?#]|$)/i.test(lower) && isGeometry && !isProxy) return 'target-glb';
    if (/\\.glb(?:[?#]|$)/i.test(lower)) return 'scene-proxy-or-other-glb';
    if (/instance_pvs_assets.*\\.bin(?:[?#]|$)/i.test(lower)) return 'neural-weight';
    if (/runtimevisibilitymeta\\.json(?:[?#]|$)/i.test(lower)) return 'runtime-visibility-meta';
    if (/glbindex\\.json(?:[?#]|$)/i.test(lower)) return 'glb-index';
    if (/sceneweb\\.json(?:[?#]|$)/i.test(lower)) return 'scene-web';
    if (/config\\.json(?:[?#]|$)/i.test(lower)) return 'viewer-config';
    return 'other';
  };
  const toUrl = (value) => {
    try { return new URL(String(value), location.href).toString(); } catch (_) { return String(value || ''); }
  };
  const requestRecord = (transport, method, rawUrl, extra = {}) => ({
    transport, method: String(method || 'GET').toUpperCase(), url: toUrl(rawUrl),
    resourceClass: root.classifyResource(toUrl(rawUrl)), ...extra,
  });

  // FileLoader/GLTFLoader use XHR for scene metadata and target GLBs.
  try {
    const open = XMLHttpRequest.prototype.open;
    const send = XMLHttpRequest.prototype.send;
    const abort = XMLHttpRequest.prototype.abort;
    const state = new WeakMap();
    XMLHttpRequest.prototype.open = function(method, url, ...rest) {
      state.set(this, { method, url: toUrl(url), started: false, ended: false });
      return open.call(this, method, url, ...rest);
    };
    XMLHttpRequest.prototype.send = function(...args) {
      const current = state.get(this) || { method: 'GET', url: '' };
      if (!current.started) {
        current.started = true;
        emit('network.request.start', requestRecord('xhr', current.method, current.url));
        const done = () => {
          if (current.ended) return;
          current.ended = true;
          emit('network.request.end', requestRecord('xhr', current.method, current.url, {
            status: Number(this.status || 0), ok: this.readyState === 4 && this.status >= 200 && this.status < 400,
            responseUrl: String(this.responseURL || current.url),
          }));
        };
        this.addEventListener('loadend', done, { once: true });
        this.addEventListener('error', done, { once: true });
        this.addEventListener('abort', done, { once: true });
      }
      return send.apply(this, args);
    };
    XMLHttpRequest.prototype.abort = function(...args) {
      const result = abort.apply(this, args);
      return result;
    };
  } catch (error) {
    emit('instrumentation.error', { area: 'xhr', message: String(error && error.message || error) });
  }

  try {
    const fetchImpl = window.fetch;
    if (typeof fetchImpl === 'function') {
      window.fetch = function(input, init) {
        const rawUrl = typeof input === 'string' ? input : (input && input.url) || '';
        const method = (init && init.method) || (input && input.method) || 'GET';
        const request = requestRecord('fetch', method, rawUrl);
        emit('network.request.start', request);
        let result;
        try { result = fetchImpl.call(this, input, init); }
        catch (error) {
          emit('network.request.end', { ...request, ok: false, error: String(error && error.message || error) });
          throw error;
        }
        return Promise.resolve(result).then((response) => {
          emit('network.request.end', { ...request, status: Number(response.status || 0), ok: response.ok, responseUrl: response.url || request.url });
          return response;
        }, (error) => {
          emit('network.request.end', { ...request, ok: false, error: String(error && error.message || error) });
          throw error;
        });
      };
    }
  } catch (error) {
    emit('instrumentation.error', { area: 'fetch', message: String(error && error.message || error) });
  }

  // Worker messages are the stable, external boundary of visibility scheduling.
  try {
    const NativeWorker = window.Worker;
    if (typeof NativeWorker === 'function') {
      const AuditedWorker = function(...args) {
        const worker = new NativeWorker(...args);
        const workerId = ++root.workerIds;
        try {
          const post = worker.postMessage.bind(worker);
          worker.postMessage = function(message, ...postArgs) {
            const type = message && typeof message === 'object' ? message.type : null;
            if (type === 'init' || type === 'predict' || type === 'setRuntimeMeta' || type === 'setModelMeta') {
              emit('visibility.worker.postMessage', {
                workerId, type, serial: message.serial == null ? null : Number(message.serial),
                candidateCount: message.candidateCount == null ? null : Number(message.candidateCount),
                workerUrl: String(args[0] || ''),
              });
            }
            return post(message, ...postArgs);
          };
          worker.addEventListener('message', (event) => {
            const value = event && event.data;
            if (value && value.__m9RuntimeAudit) {
              emit(value.__m9RuntimeAudit.name || 'worker.runtime.event', {
                workerId, ...(value.__m9RuntimeAudit.details || {}),
                workerBrowserTimeMs: value.__m9RuntimeAudit.browserTimeMs,
              });
            }
          });
        } catch (error) {
          emit('instrumentation.error', { area: 'worker-boundary', workerId, message: String(error && error.message || error) });
        }
        return worker;
      };
      AuditedWorker.prototype = NativeWorker.prototype;
      Object.setPrototypeOf(AuditedWorker, NativeWorker);
      window.Worker = AuditedWorker;
    }
  } catch (error) {
    emit('instrumentation.error', { area: 'worker', message: String(error && error.message || error) });
  }

  function wrapMethod(target, key, label, onResult) {
    if (!target || typeof target[key] !== 'function') return false;
    const original = target[key];
    if (original.__m9Wrapped) return true;
    const wrapped = function(...args) {
      emit(label + '.start', { argumentCount: args.length });
      let result;
      try { result = original.apply(this, args); }
      catch (error) {
        emit(label + '.end', { ok: false, error: String(error && error.message || error) });
        throw error;
      }
      if (!result || typeof result.then !== 'function') {
        if (onResult) onResult(result);
        emit(label + '.end', { ok: true });
        return result;
      }
      return Promise.resolve(result).then((value) => {
        if (onResult) onResult(value);
        emit(label + '.end', { ok: true });
        return value;
      }, (error) => {
        emit(label + '.end', { ok: false, error: String(error && error.message || error) });
        throw error;
      });
    };
    wrapped.__m9Wrapped = true;
    try {
      Object.defineProperty(target, key, { value: wrapped, configurable: true, writable: true });
      return true;
    } catch (_) {
      try { target[key] = wrapped; return target[key] === wrapped; } catch (__) { return false; }
    }
  }

  function instrumentDevice(device) {
    if (!device || device.__m9Instrumented) return;
    try { Object.defineProperty(device, '__m9Instrumented', { value: true }); } catch (_) { /* best effort */ }
    for (const key of ['createShaderModule', 'createComputePipeline', 'createComputePipelineAsync', 'createCommandEncoder']) {
      wrapMethod(device, key, 'webgpu.device.' + key);
    }
    try { wrapMethod(device.queue, 'submit', 'webgpu.queue.submit'); } catch (_) { /* best effort */ }
    try {
      if (device.lost && typeof device.lost.then === 'function') {
        device.lost.then((info) => emit('webgpu.device.lost', { reason: info && info.reason, message: info && info.message }));
      }
    } catch (_) { /* best effort */ }
  }

  function instrumentAdapter(adapter) {
    if (!adapter) return;
    wrapMethod(adapter, 'requestDevice', 'webgpu.adapter.requestDevice', instrumentDevice);
  }

  try {
    const gpu = navigator.gpu;
    if (gpu && typeof gpu.requestAdapter === 'function') {
      wrapMethod(gpu, 'requestAdapter', 'webgpu.requestAdapter', instrumentAdapter);
      emit('webgpu.api.available', { secureContext: Boolean(window.isSecureContext), scope: 'page' });
    } else {
      emit('webgpu.api.unavailable', { secureContext: Boolean(window.isSecureContext), scope: 'page' });
    }
  } catch (error) {
    emit('instrumentation.error', { area: 'webgpu-page', message: String(error && error.message || error) });
  }

  try {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        emit('network.resource-timing', {
          name: String(entry.name || ''), resourceClass: root.classifyResource(entry.name),
          initiatorType: entry.initiatorType || null, startTime: entry.startTime,
          responseEnd: entry.responseEnd, duration: entry.duration,
          transferSize: entry.transferSize, encodedBodySize: entry.encodedBodySize,
          decodedBodySize: entry.decodedBodySize,
        });
      }
    });
    observer.observe({ type: 'resource', buffered: true });
  } catch (error) {
    emit('instrumentation.error', { area: 'performance-observer', message: String(error && error.message || error) });
  }

  emit('runtime.instrumentation.ready', {
    href: location.href, readyState: document.readyState,
    secureContext: Boolean(window.isSecureContext), userAgent: navigator.userAgent,
  });
})();`;
}

function makeWorkerInitScript() {
  return `(() => {
  if (self.__m9WorkerRuntimeAuditInstalled) return;
  self.__m9WorkerRuntimeAuditInstalled = true;
  const nativePost = self.postMessage.bind(self);
  const emit = (name, details = {}) => {
    try {
      nativePost({ __m9RuntimeAudit: {
        name, browserTimeMs: typeof performance !== 'undefined' && performance.now ? performance.now() : null,
        details: details && typeof details === 'object' ? details : { value: details },
      }});
    } catch (_) { /* observer must not break the worker */ }
  };
  function wrap(target, key, label, onResult) {
    if (!target || typeof target[key] !== 'function') return false;
    const original = target[key];
    if (original.__m9Wrapped) return true;
    const wrapped = function(...args) {
      emit(label + '.start', { argumentCount: args.length });
      let result;
      try { result = original.apply(this, args); }
      catch (error) { emit(label + '.end', { ok: false, error: String(error && error.message || error) }); throw error; }
      if (!result || typeof result.then !== 'function') { if (onResult) onResult(result); emit(label + '.end', { ok: true }); return result; }
      return Promise.resolve(result).then((value) => { if (onResult) onResult(value); emit(label + '.end', { ok: true }); return value; }, (error) => { emit(label + '.end', { ok: false, error: String(error && error.message || error) }); throw error; });
    };
    wrapped.__m9Wrapped = true;
    try { Object.defineProperty(target, key, { value: wrapped, configurable: true, writable: true }); return true; }
    catch (_) { try { target[key] = wrapped; return target[key] === wrapped; } catch (__) { return false; } }
  }
  function instrumentDevice(device) {
    if (!device) return;
    for (const key of ['createShaderModule', 'createComputePipeline', 'createComputePipelineAsync', 'createCommandEncoder']) wrap(device, key, 'webgpu.worker.device.' + key);
    try { wrap(device.queue, 'submit', 'webgpu.worker.queue.submit'); } catch (_) { /* best effort */ }
    try { if (device.lost && typeof device.lost.then === 'function') device.lost.then((info) => emit('webgpu.worker.device.lost', { reason: info && info.reason, message: info && info.message })); } catch (_) { /* best effort */ }
  }
  try {
    const gpu = self.navigator && self.navigator.gpu;
    if (gpu && typeof gpu.requestAdapter === 'function') {
      wrap(gpu, 'requestAdapter', 'webgpu.worker.requestAdapter', (adapter) => {
        if (adapter) wrap(adapter, 'requestDevice', 'webgpu.worker.adapter.requestDevice', instrumentDevice);
      });
      emit('webgpu.worker.api.available', { secureContext: Boolean(self.isSecureContext) });
    } else emit('webgpu.worker.api.unavailable', { secureContext: Boolean(self.isSecureContext) });
  } catch (error) { emit('instrumentation.error', { area: 'webgpu-worker', message: String(error && error.message || error) }); }
  emit('runtime.worker.instrumentation.ready', {});
})();`;
}

function createReport(options, staticAudit, url) {
  return {
    schemaVersion: 'm9-browser-runtime-report-v1',
    scriptVersion: SCRIPT_VERSION,
    run: {
      startedAt: isoNow(),
      host: os.hostname(),
      platform: process.platform,
      node: process.version,
      scene: options.scene,
      url,
      viewport: options.viewport,
      durationMs: options.durationMs,
      pollMs: options.pollMs,
      settleMs: options.settleMs,
      waitFor: options.waitFor,
      serverStartedByTool: false,
      browser: null,
    },
    staticAudit,
    status: 'not-started',
    blockedReason: null,
    events: [],
    networkRequests: [],
    console: [],
    pageErrors: [],
    requestFailures: [],
    runtimeSnapshots: [],
    constraintProbes: [],
    summary: null,
  };
}

function pushEvent(report, event) {
  if (report.events.length >= report._maxEvents) {
    report._droppedEvents = Number(report._droppedEvents || 0) + 1;
    return;
  }
  report.events.push(event);
}

function hostEvent(report, name, details = {}) {
  pushEvent(report, {
    seq: report.events.length + 1,
    name,
    hostTimeMs: nowMs(),
    wallTime: isoNow(),
    source: 'runner',
    details,
  });
}

function appendBrowserEvents(report, browserEvents) {
  for (const event of browserEvents || []) {
    pushEvent(report, {
      ...event,
      source: event.name && event.name.startsWith('webgpu.') ? 'page-instrumentation' : 'page-instrumentation',
    });
  }
}

function compactRuntimeStats(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const visibility = raw.visibility || {};
  const neural = raw.neural || {};
  const prediction = neural.predictTimings || {};
  const scheduler = neural.priorityScheduler || {};
  const renderVisibility = neural.renderVisibility || {};
  const renderRefresh = neural.renderRefresh || {};
  const actualRender = neural.actualRender || {};
  const startup = raw.startup || {};
  const load = raw.load || {};
  const gate = neural.predictionGate || {};
  return {
    browserTimeMs: Number.isFinite(Number(raw.browserTimeMs)) ? Number(raw.browserTimeMs) : null,
    startup: {
      firstPredictionMs: Number(startup.firstPredictionMs || 0),
      firstPredictionRawMs: Number(startup.firstPredictionRawMs || 0),
      firstPredictionApplyMs: Number(startup.firstPredictionApplyMs || 0),
      sceneWebMs: Number(startup.sceneWebMs || 0),
      glbIndexMs: Number(startup.glbIndexMs || 0),
      runtimeVisibilityMetaMs: Number(startup.runtimeVisibilityMetaMs || 0),
    },
    visibility: {
      mode: visibility.mode || null,
      backend: visibility.backend || null,
      serial: visibility.serial == null ? null : Number(visibility.serial),
      latencyMs: visibility.latencyMs == null ? null : Number(visibility.latencyMs),
      rawInstanceCount: visibility.rawInstanceCount == null ? null : Number(visibility.rawInstanceCount),
      rawGlbCount: visibility.rawGlbCount == null ? null : Number(visibility.rawGlbCount),
      timestamp: visibility.timestamp == null ? null : Number(visibility.timestamp),
      notes: visibility.notes || null,
    },
    neural: {
      enabled: Boolean(neural.enabled),
      ready: Boolean(neural.ready),
      backend: neural.backend || null,
      cullingMode: neural.cullingMode || null,
      currentEpoch: neural.currentEpoch == null ? null : Number(neural.currentEpoch),
      prediction: {
        serial: prediction.serial == null ? null : Number(prediction.serial),
        totalMs: prediction.totalMs == null ? null : Number(prediction.totalMs),
        inferenceMs: prediction.inferenceMs == null ? null : Number(prediction.inferenceMs),
        postMs: prediction.postMs == null ? null : Number(prediction.postMs),
        candidateCount: prediction.candidateCount == null
          ? (scheduler.candidateCount == null ? null : Number(scheduler.candidateCount))
          : Number(prediction.candidateCount),
        candidateMs: prediction.candidateMs == null ? null : Number(prediction.candidateMs),
        rawInstanceCount: prediction.rawInstanceCount == null ? null : Number(prediction.rawInstanceCount),
        rawGlbCount: prediction.rawGlbCount == null ? null : Number(prediction.rawGlbCount),
        renderInstanceCount: prediction.renderInstanceCount == null ? null : Number(prediction.renderInstanceCount),
        renderGlbCount: prediction.renderGlbCount == null ? null : Number(prediction.renderGlbCount),
        backend: prediction.backend || null,
        stale: Boolean(prediction.stale),
      },
      scheduler: {
        candidateCount: scheduler.candidateCount == null ? null : Number(scheduler.candidateCount),
        rawInstanceCount: scheduler.rawInstanceCount == null ? null : Number(scheduler.rawInstanceCount),
        rawGlbCount: scheduler.rawGlbCount == null ? null : Number(scheduler.rawGlbCount),
        scheduledComponentCount: scheduler.scheduledComponentCount == null ? null : Number(scheduler.scheduledComponentCount),
        scheduledGlbCount: scheduler.scheduledGlbCount == null ? null : Number(scheduler.scheduledGlbCount),
        renderComponentCount: scheduler.renderComponentCount == null ? null : Number(scheduler.renderComponentCount),
        renderGlbCount: scheduler.renderGlbCount == null ? null : Number(scheduler.renderGlbCount),
        backend: scheduler.backend || null,
      },
      renderVisibility: {
        epoch: renderVisibility.epoch == null ? null : Number(renderVisibility.epoch),
        workingSetSize: renderVisibility.workingSetSize == null ? null : Number(renderVisibility.workingSetSize),
        activeEvaluationSize: renderVisibility.activeEvaluationSize == null ? null : Number(renderVisibility.activeEvaluationSize),
        evaluatedCount: renderVisibility.evaluatedCount == null ? null : Number(renderVisibility.evaluatedCount),
        visibleCount: renderVisibility.visibleCount == null ? null : Number(renderVisibility.visibleCount),
        frustumRejected: renderVisibility.frustumRejected == null ? null : Number(renderVisibility.frustumRejected),
        areaRejected: renderVisibility.areaRejected == null ? null : Number(renderVisibility.areaRejected),
        hiddenOutsideWorkingSet: renderVisibility.hiddenOutsideWorkingSet == null ? null : Number(renderVisibility.hiddenOutsideWorkingSet),
        keptVisibleOutsideWorkingSet: renderVisibility.keptVisibleOutsideWorkingSet == null ? null : Number(renderVisibility.keptVisibleOutsideWorkingSet),
        retainedByDelay: renderVisibility.retainedByDelay == null ? null : Number(renderVisibility.retainedByDelay),
        detachedCount: renderVisibility.detachedCount == null ? null : Number(renderVisibility.detachedCount),
        skippedByGate: renderVisibility.skippedByGate == null ? null : Boolean(renderVisibility.skippedByGate),
        skippedByPolicy: renderVisibility.skippedByPolicy == null ? null : Boolean(renderVisibility.skippedByPolicy),
        durationMs: renderVisibility.durationMs == null ? null : Number(renderVisibility.durationMs),
      },
      instancedBindings: neural.instancedBindings ? {
        expectedCount: neural.instancedBindings.expectedCount == null ? null : Number(neural.instancedBindings.expectedCount),
        mappedCount: neural.instancedBindings.mappedCount == null ? null : Number(neural.instancedBindings.mappedCount),
        loadedCount: neural.instancedBindings.loadedCount == null ? null : Number(neural.instancedBindings.loadedCount),
        invalidCount: neural.instancedBindings.invalidCount == null ? null : Number(neural.instancedBindings.invalidCount),
      } : null,
      renderRefresh: {
        desiredCount: renderRefresh.desiredCount == null ? null : Number(renderRefresh.desiredCount),
        fallbackVisibleCount: renderRefresh.fallbackVisibleCount == null ? null : Number(renderRefresh.fallbackVisibleCount),
        visibleCount: renderRefresh.visibleCount == null ? null : Number(renderRefresh.visibleCount),
        hiddenCount: renderRefresh.hiddenCount == null ? null : Number(renderRefresh.hiddenCount),
        residentCount: renderRefresh.residentCount == null ? null : Number(renderRefresh.residentCount),
        detachedCount: renderRefresh.detachedCount == null ? null : Number(renderRefresh.detachedCount),
      },
      actualRender: {
        meshCount: actualRender.meshCount == null ? null : Number(actualRender.meshCount),
        visibleMeshCount: actualRender.visibleMeshCount == null ? null : Number(actualRender.visibleMeshCount),
        instancedMeshCount: actualRender.instancedMeshCount == null ? null : Number(actualRender.instancedMeshCount),
        visibleInstancedMeshCount: actualRender.visibleInstancedMeshCount == null ? null : Number(actualRender.visibleInstancedMeshCount),
        drawnInstanceCount: actualRender.drawnInstanceCount == null ? null : Number(actualRender.drawnInstanceCount),
      },
      gate: {
        mode: gate.mode || null,
        positionDelta: gate.positionDelta == null ? null : Number(gate.positionDelta),
        yawDeltaDeg: gate.yawDeltaDeg == null ? null : Number(gate.yawDeltaDeg),
        pitchDeltaDeg: gate.pitchDeltaDeg == null ? null : Number(gate.pitchDeltaDeg),
        minIntervalMs: gate.minIntervalMs == null ? null : Number(gate.minIntervalMs),
        hasCommittedPrediction: Boolean(gate.hasCommittedPrediction),
        forceNext: Boolean(gate.forceNext),
      },
    },
    load: {
      queueLength: Number(load.queueLength || 0),
      inflightCount: Number(load.inflightCount || 0),
      pendingSceneInsertions: Number(load.pendingSceneInsertions || 0),
      activeDirectLoadCount: Number(load.activeDirectLoadCount || 0),
      totalIntegrated: Number(load.totalIntegrated || 0),
    },
  };
}

async function drainBrowserAudit(page) {
  return page.evaluate(() => {
    const root = window.__m9RuntimeAudit;
    if (!root) return null;
    return {
      events: Array.isArray(root.events) ? root.events.splice(0) : [],
      droppedEvents: Number(root.droppedEvents || 0),
    };
  }).catch(() => null);
}

function addRuntimeSnapshot(report, snapshot) {
  if (!snapshot) return;
  report.runtimeSnapshots.push(snapshot);
  const prediction = snapshot.neural?.prediction || {};
  const signature = JSON.stringify([
    prediction.serial, prediction.totalMs, prediction.candidateCount,
    snapshot.neural?.backend, snapshot.neural?.ready,
    snapshot.neural?.currentEpoch,
  ]);
  if (signature !== report._lastRuntimeSignature) {
    report._lastRuntimeSignature = signature;
    hostEvent(report, 'visibility.runtime.snapshot', {
      browserTimeMs: snapshot.browserTimeMs,
      prediction: snapshot.neural?.prediction,
      scheduler: snapshot.neural?.scheduler,
      backend: snapshot.neural?.backend,
      ready: snapshot.neural?.ready,
      currentEpoch: snapshot.neural?.currentEpoch,
    });
  }
}

async function probePageState(page, report, label) {
  const probe = await page.evaluate(() => {
    const digestIds = (values) => {
      const ids = Array.isArray(values) ? values.map(Number).filter(Number.isFinite) : [];
      let checksum = 2166136261;
      for (const id of ids) {
        checksum ^= (id >>> 0);
        checksum = Math.imul(checksum, 16777619) >>> 0;
      }
      return { count: ids.length, first: ids.slice(0, 8), last: ids.slice(-8), checksum };
    };
    const viewer = window.__slmApp?.viewer;
    const loader = viewer?.slm2Loader;
    if (!loader) return { available: false, reason: 'viewer-loader-not-available' };
    const ids = typeof loader.getBenchmarkVisibilityIds === 'function'
      ? loader.getBenchmarkVisibilityIds()
      : null;
    const stats = typeof loader.getRuntimeStats === 'function' ? loader.getRuntimeStats() : null;
    const raw = Array.isArray(ids?.rawComponentIds) ? ids.rawComponentIds : [];
    const render = Array.isArray(ids?.renderComponentIds) ? ids.renderComponentIds : [];
    const rawSet = new Set(raw);
    const renderOutsideRaw = render.filter((id) => !rawSet.has(id)).length;
    const records = Array.isArray(loader.componentVisibilityRecords)
      ? loader.componentVisibilityRecords : [];
    const unknownRender = render.filter((id) => !records[Number(id)]).length;
    const glbSet = new Set(render.map((id) => records[Number(id)]?.globalGlbId).filter((id) => id != null));
    const instancedStates = loader.loadedInstancedVisibilityStatesByHash || {};
    const stateValues = Object.values(instancedStates).filter(Boolean);
    const disabledStateCount = stateValues.filter((state) => state.disabled).length;
    const meshCounts = [];
    for (const state of stateValues) {
      for (const meshState of state.meshStates || []) {
        meshCounts.push({ hash: state.hash, count: Number(meshState.mesh?.count || 0), originalCount: Number(meshState.originalCount || 0) });
      }
    }
    const componentBindings = loader.instancedVisibilityBindingByComponentId || {};
    const boundRenderCount = render.filter((id) => componentBindings[Number(id)]).length;
    const renderVisibilitySystem = loader.renderVisibilitySystem;
    const renderStatsBefore = renderVisibilitySystem?.lastStats || null;
    const renderUpdateBefore = renderVisibilitySystem && typeof renderVisibilitySystem.update === 'function'
      ? renderVisibilitySystem.update.bind(renderVisibilitySystem)
      : null;
    const frustumDigest = (camera) => {
      if (!camera || typeof loader._getComponentIdsInFrustum !== 'function') return null;
      return loader._getComponentIdsInFrustum(camera);
    };
    const renderComponentIds = () => {
      const visibility = typeof loader.getBenchmarkVisibilityIds === 'function'
        ? loader.getBenchmarkVisibilityIds()
        : null;
      return Array.isArray(visibility?.renderComponentIds)
        ? visibility.renderComponentIds.map(Number).filter(Number.isFinite)
        : [];
    };
    const renderSetBefore = renderComponentIds();
    const activeBefore = frustumDigest(loader.activeCamera);
    const backBefore = frustumDigest(loader.backCamera);
    let activeAfter = null;
    let cameraRestored = true;
    let renderRefreshAfterActiveMove = null;
    let renderSetAfterActiveMove = null;
    let renderRefreshRestored = null;
    if (loader.activeCamera && activeBefore) {
      const camera = loader.activeCamera;
      const original = {
        x: camera.position.x, y: camera.position.y, z: camera.position.z,
        qx: camera.quaternion.x, qy: camera.quaternion.y,
        qz: camera.quaternion.z, qw: camera.quaternion.w,
      };
      camera.position.x += 2.0;
      camera.updateMatrixWorld(true);
      activeAfter = frustumDigest(camera);
      if (renderUpdateBefore) {
        try {
          renderRefreshAfterActiveMove = renderUpdateBefore({ force: true });
          renderSetAfterActiveMove = renderComponentIds();
        } catch (error) {
          renderRefreshAfterActiveMove = { error: String(error && error.message || error) };
        }
      }
      camera.position.set(original.x, original.y, original.z);
      camera.quaternion.set(original.qx, original.qy, original.qz, original.qw);
      camera.updateMatrixWorld(true);
      if (renderUpdateBefore) {
        try {
          renderRefreshRestored = renderUpdateBefore({ force: true });
        } catch (error) {
          renderRefreshRestored = { error: String(error && error.message || error) };
        }
      }
      cameraRestored = Math.abs(camera.position.x - original.x) < 1e-6 &&
        Math.abs(camera.position.y - original.y) < 1e-6 && Math.abs(camera.position.z - original.z) < 1e-6;
    }
    const actualRoot = loader.rootScene;
    let meshCount = 0;
    let visibleMeshCount = 0;
    let instancedMeshCount = 0;
    let visibleInstancedMeshCount = 0;
    let drawnInstanceCount = 0;
    if (actualRoot && typeof actualRoot.traverse === 'function') {
      actualRoot.traverse((node) => {
        if (!node?.isMesh) return;
        meshCount += 1;
        if (node.visible !== false) visibleMeshCount += 1;
        if (node.isInstancedMesh) {
          instancedMeshCount += 1;
          if (node.visible !== false) visibleInstancedMeshCount += 1;
          drawnInstanceCount += Number(node.count || 0);
        }
      });
    }
    return {
      available: true,
      mode: ids?.mode || null,
      idMode: stats?.neural?.idMode || null,
      visibilitySerial: ids?.serial == null ? null : Number(ids.serial),
      componentSets: {
        raw: { count: raw.length },
        scheduled: { count: Array.isArray(ids?.scheduledComponentIds) ? ids.scheduledComponentIds.length : 0 },
        render: { count: render.length },
        renderOutsideRaw,
        unknownRender,
        uniqueRenderGlbs: glbSet.size,
        componentListObserved: render.length > 0 || raw.length > 0,
      },
      frustum: {
        filterEnabled: Boolean(loader.runtimeFrustumFilterEnabled),
        filterSource: loader.runtimeFrustumFilterSource || null,
        activeBefore: activeBefore ? digestIds(activeBefore) : null,
        activeAfterSmallMove: activeAfter ? digestIds(activeAfter) : null,
        activeChangedForSmallMove: Boolean(activeBefore && activeAfter && (
          activeBefore.length !== activeAfter.length || activeBefore.some((id, index) => id !== activeAfter[index])
        )),
        renderComponentIdsBefore: digestIds(renderSetBefore),
        renderComponentIdsAfterActiveMove: renderSetAfterActiveMove ? digestIds(renderSetAfterActiveMove) : null,
        renderComponentSetChangedForActiveMove: Boolean(renderSetAfterActiveMove && (
          renderSetBefore.length !== renderSetAfterActiveMove.length ||
          renderSetBefore.some((id, index) => id !== renderSetAfterActiveMove[index])
        )),
        renderRefreshInvokedForActiveMove: Boolean(renderRefreshAfterActiveMove),
        renderRefreshAfterActiveMove,
        renderRefreshRestored,
        independentFrustumRefreshObserved: Boolean(renderRefreshAfterActiveMove &&
          !renderRefreshAfterActiveMove.error && renderRefreshAfterActiveMove.skippedByGate === false),
        renderStatsBefore,
        backBefore: backBefore ? digestIds(backBefore) : null,
        cameraRestored,
        renderRefreshStats: loader.lastRenderRefreshStats || null,
        renderVisibilityStats: loader.renderVisibilitySystem?.lastStats || null,
        renderVisibilityLastUpdateAt: loader.renderVisibilitySystem?.lastUpdateAt || null,
        renderVisibilityHasLastCamera: Boolean(loader.renderVisibilitySystem?.hasLastCamera),
      },
      instanceDisplay: {
        bindingCount: Object.keys(componentBindings).length,
        loadedStateCount: stateValues.length,
        disabledStateCount,
        boundRenderCount,
        meshCounts: meshCounts.slice(0, 32),
        actualMeshCount: meshCount,
        actualVisibleMeshCount: visibleMeshCount,
        actualInstancedMeshCount: instancedMeshCount,
        actualVisibleInstancedMeshCount: visibleInstancedMeshCount,
        actualDrawnInstanceCount: drawnInstanceCount,
        evidence: render.length > 0 && stateValues.length > 0 && disabledStateCount === 0
          ? 'instance-state-observed'
          : (render.length > 0 ? 'component-set-observed-but-no-loaded-instance-state' : 'no-component-render-set-observed'),
      },
    };
  }).catch((error) => ({ available: false, reason: String(error && error.message || error) }));
  report.constraintProbes = report.constraintProbes || [];
  report.constraintProbes.push({ label, hostTimeMs: nowMs(), wallTime: isoNow(), probe });
  hostEvent(report, 'visibility.constraints.probe', { label, probe });
  return probe;
}

function buildRuntimeAssetErrorSummary(report) {
  const loadErrorConsole = report.console.filter((item) =>
    /\[LoadError\]|Unexpected token.*<!DOCTYPE|GLTFLoader.*error|failed to load/i.test(String(item.text || '')));
  const consoleErrors = report.console.filter((item) => item.type === 'error');
  const samples = loadErrorConsole.slice(0, 8).map((item) => ({
    type: item.type,
    text: String(item.text || '').slice(0, 500),
    hostTimeMs: item.hostTimeMs ?? null,
  }));
  return {
    consoleErrorCount: consoleErrors.length,
    loadErrorCount: loadErrorConsole.length,
    pageErrorCount: report.pageErrors.length,
    requestFailureCount: report.requestFailures.length,
    requestFailures: report.requestFailures.slice(0, 8),
    samples,
  };
}

function buildSummary(report) {
  const events = report.events;
  const firstPrediction = events.find((event) => event.name === 'visibility.worker.postMessage' && event.details?.type === 'predict');
  const firstPredictionCompleted = report.runtimeSnapshots.find((snapshot) =>
    snapshot.neural?.prediction?.serial != null);
  const predictionCompletionTime = firstPredictionCompleted?.visibility?.timestamp ??
    firstPredictionCompleted?.browserTimeMs ?? firstPredictionCompleted?.hostTimeMs ?? null;
  const firstWebGpu = events.find((event) => /webgpu\.(?:worker\.)?(?:requestAdapter|api\.available|adapter\.requestDevice)\.end|webgpu\.(?:worker\.)?api\.available|webgpu\.console/i.test(event.name));
  const targetStarts = events.filter((event) => event.name === 'network.request.start' && event.details?.resourceClass === 'target-glb');
  const targetEnds = events.filter((event) => event.name === 'network.request.end' && event.details?.resourceClass === 'target-glb');
  const firstTarget = targetStarts[0] || null;
  const predictionTime = firstPrediction?.browserTimeMs ?? firstPrediction?.hostTimeMs ?? null;
  const targetTime = firstTarget?.browserTimeMs ?? firstTarget?.hostTimeMs ?? null;
  let cold0Status = 'blocked-no-first-prediction-observed';
  if (firstPrediction) {
    if (!firstTarget) cold0Status = 'pass-observed-no-target-glb-before-first-prediction';
    else if (targetTime < predictionTime) cold0Status = 'fail-target-glb-request-before-first-prediction';
    else cold0Status = 'pass-no-target-glb-request-before-first-prediction';
  }
  const webgpuEvents = events.filter((event) => event.name.startsWith('webgpu.'));
  const webgpuStatus = webgpuEvents.some((event) =>
    event.name.endsWith('.end') && event.details?.ok === true &&
    /requestAdapter|requestDevice/.test(event.name))
    ? 'observed-success'
    : webgpuEvents.some((event) => /unavailable|failed|error|lost/.test(event.name) || event.details?.ok === false)
      ? 'observed-failure-or-unavailable'
      : 'not-observed';
  const staticMissing = report.staticAudit?.checks?.filter((check) => !check.exists) || [];
  const firstPredictionMs = report.runtimeSnapshots.find((snapshot) => snapshot.startup?.firstPredictionMs > 0)?.startup.firstPredictionMs || null;
  return {
    firstPrediction: firstPrediction ? {
      browserTimeMs: firstPrediction.browserTimeMs ?? null,
      hostTimeMs: firstPrediction.hostTimeMs ?? null,
      serial: firstPrediction.details?.serial ?? null,
    } : null,
    targetGlb: {
      requestsObserved: targetStarts.length,
      completedObserved: targetEnds.length,
      firstRequestBrowserTimeMs: firstTarget?.browserTimeMs ?? null,
      firstRequestHostTimeMs: firstTarget?.hostTimeMs ?? null,
      requestsBeforeFirstPrediction: firstPrediction
        ? targetStarts.filter((event) => (event.browserTimeMs ?? event.hostTimeMs) < predictionTime).length
        : null,
      requestsBeforeFirstPredictionCompletion: firstPredictionCompleted
        ? targetStarts.filter((event) => (event.browserTimeMs ?? event.hostTimeMs) < predictionCompletionTime).length
        : null,
      responseSummary: report.networkRequests
        .filter((item) => item.phase === 'response' && item.resourceClass === 'target-glb')
        .reduce((summary, item) => {
          summary.statuses[String(item.status)] = (summary.statuses[String(item.status)] || 0) + 1;
          if (/text\/html/i.test(String(item.contentType || ''))) summary.htmlFallbackResponses += 1;
          else if (item.contentType) summary.nonHtmlResponses += 1;
          if (item.contentValidation === 'glb-magic') summary.validatedGlbResponses += 1;
          else if (item.contentValidation === 'invalid-glb-magic' || item.contentValidation === 'invalid-html-content') {
            summary.invalidContentResponses += 1;
          } else if (item.contentValidation) {
            summary.unvalidatedContentResponses += 1;
          }
          return summary;
        }, {
          statuses: {},
          htmlFallbackResponses: 0,
          nonHtmlResponses: 0,
          validatedGlbResponses: 0,
          invalidContentResponses: 0,
          unvalidatedContentResponses: 0,
        }),
    },
    cold0: {
      status: cold0Status,
      definition: 'Target scene GLB request start must not precede the first visibility predict scheduling event.',
      completionObserved: Boolean(firstPredictionCompleted),
      firstPredictionCompletionBrowserTimeMs: predictionCompletionTime,
      completionTimestampSource: firstPredictionCompleted?.visibility?.timestamp != null
        ? 'frontend-visibility.timestamp'
        : 'runtime-snapshot-observation-time',
      evidence: firstPrediction ? 'worker.postMessage(type=predict) and instrumented XHR/fetch events' : 'no first prediction event',
    },
    webgpu: {
      status: webgpuStatus,
      eventCount: webgpuEvents.length,
      firstEvent: webgpuEvents[0] ? {
        name: webgpuEvents[0].name,
        browserTimeMs: webgpuEvents[0].browserTimeMs ?? null,
        hostTimeMs: webgpuEvents[0].hostTimeMs ?? null,
      } : null,
    },
    visibility: {
      scheduledPredictions: events.filter((event) => event.name === 'visibility.worker.postMessage' && event.details?.type === 'predict').length,
      runtimeSnapshots: report.runtimeSnapshots.length,
      firstPredictionMetricMs: firstPredictionMs,
      finalBackend: report.runtimeSnapshots.at(-1)?.neural?.backend || null,
      finalRuntimePrediction: report.runtimeSnapshots.at(-1)?.neural?.prediction || null,
      backendTransitions: report.runtimeSnapshots
        .map((snapshot) => snapshot.neural?.backend)
        .filter((backend, index, values) => backend && values.indexOf(backend) === index),
    },
    constraints: {
      latestProbe: report.constraintProbes?.at(-1) || null,
      instanceLevelEvidence: report.constraintProbes?.some((item) => item.probe?.instanceDisplay?.evidence === 'instance-state-observed')
        ? 'observed'
        : 'not-observed',
      aabbFrustumQueryEvidence: report.constraintProbes?.some((item) =>
        item.probe?.frustum?.activeChangedForSmallMove === true)
        ? 'observed'
        : 'not-observed',
      independentFrustumRefreshEvidence: report.constraintProbes?.some((item) =>
        item.probe?.frustum?.renderVisibilityStats?.skippedByGate === false ||
        item.probe?.frustum?.renderVisibilityLastUpdateAt != null)
        ? 'observed'
        : 'not-observed',
    },
    runtimeAssetErrors: buildRuntimeAssetErrorSummary(report),
    staticChecks: {
      allRequiredLocalFilesPresent: Boolean(report.staticAudit?.allRequiredLocalFilesPresent),
      missing: staticMissing.map((check) => check.label),
      defaultLocalTargetGlbRootPresent: Boolean(report.staticAudit?.targetGlbDeploymentPresentForDefaultLocalServer),
    },
    limitations: [
      'This run observes scheduling and network boundaries; it does not alter or instrument production source code.',
      'Only a bounded sample of target GLB responses is checked for the glTF binary magic; unprobed responses are not treated as geometry proof.',
      'No browser performance claim is valid unless a real Chromium run reaches the requested visibility event and records timings.',
    ],
  };
}

async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch (firstError) {
    try {
      return await import('playwright-core');
    } catch (secondError) {
      const error = new Error('Playwright is not installed. Run `cd slm2viewer && npm install --no-save playwright` or provide a CDP-capable environment.');
      error.cause = { firstError: String(firstError), secondError: String(secondError) };
      throw error;
    }
  }
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
  const started = nowMs();
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const client = url.startsWith('https:') ? https : http;
      const request = client.get(url, { timeout: 1000 }, (response) => {
        response.resume();
        if (response.statusCode && response.statusCode < 500) {
          resolve(response.statusCode);
        } else retry();
      });
      request.on('error', retry);
      request.on('timeout', () => request.destroy());
    };
    const retry = () => {
      if (nowMs() - started >= timeoutMs) {
        reject(new Error(`Timed out waiting for local server at ${url}`));
      } else setTimeout(attempt, 250);
    };
    attempt();
  });
}

function startLocalServer(options, report) {
  const command = options.serverCommand || 'local Parcel';
  const parcelBinary = path.join(options.serverCwd, 'node_modules', '.bin', 'parcel');
  const shellCommand = options.serverCommand
    ? `${command} -- --port ${options.port}`
    : `${JSON.stringify(parcelBinary)} index.html --port ${options.port}`;
  hostEvent(report, 'server.start', { command: shellCommand, cwd: options.serverCwd, port: options.port });
  const child = spawn(shellCommand, {
    cwd: options.serverCwd,
    shell: true,
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PORT: String(options.port) },
  });
  const output = [];
  const collect = (stream, channel) => stream.on('data', (chunk) => {
    output.push({ channel, text: String(chunk).slice(0, 4000), hostTimeMs: nowMs(), wallTime: isoNow() });
    if (output.length > 100) output.shift();
  });
  collect(child.stdout, 'stdout');
  collect(child.stderr, 'stderr');
  report.serverOutput = output;
  report.run.serverStartedByTool = true;
  return child;
}

async function pollPageRuntime(page, report, durationMs, pollMs, waitFor, settleMs) {
  const started = nowMs();
  let firstPredictionSeen = false;
  let firstWebGpuSeen = false;
  let settleStarted = null;
  while (nowMs() - started < durationMs) {
    let payload = null;
    try {
      payload = await page.evaluate(() => {
        const loader = window.__slmApp?.viewer?.slm2Loader;
        if (!loader || typeof loader.getRuntimeStats !== 'function') {
          return { browserTimeMs: performance.now(), available: false };
        }
        const stats = loader.getRuntimeStats();
        const visibility = stats.visibility || {};
        const neural = stats.neural || {};
        const prediction = neural.predictTimings || {};
        const scheduler = neural.priorityScheduler || {};
        const startup = stats.startup || {};
        const load = stats.load || {};
        const gate = neural.predictionGate || {};
        return {
          browserTimeMs: performance.now(), available: true, stats: {
            startup: {
              firstPredictionMs: startup.firstPredictionMs,
              firstPredictionRawMs: startup.firstPredictionRawMs,
              firstPredictionApplyMs: startup.firstPredictionApplyMs,
              sceneWebMs: startup.sceneWebMs,
              glbIndexMs: startup.glbIndexMs,
              runtimeVisibilityMetaMs: startup.runtimeVisibilityMetaMs,
            },
            visibility: {
              mode: visibility.mode, backend: visibility.backend, serial: visibility.serial,
              latencyMs: visibility.latencyMs, rawInstanceCount: visibility.rawInstanceCount,
              rawGlbCount: visibility.rawGlbCount, timestamp: visibility.timestamp, notes: visibility.notes,
            },
            neural: {
              enabled: neural.enabled, ready: neural.ready, backend: neural.backend,
              cullingMode: neural.cullingMode, currentEpoch: neural.currentEpoch,
              predictTimings: {
                serial: prediction.serial, totalMs: prediction.totalMs,
                inferenceMs: prediction.inferenceMs, postMs: prediction.postMs,
                candidateCount: prediction.candidateCount, candidateMs: prediction.candidateMs,
                rawInstanceCount: prediction.rawInstanceCount,
                rawGlbCount: prediction.rawGlbCount, renderInstanceCount: prediction.renderInstanceCount,
                renderGlbCount: prediction.renderGlbCount, backend: prediction.backend, stale: prediction.stale,
              },
              priorityScheduler: {
                candidateCount: scheduler.candidateCount, rawInstanceCount: scheduler.rawInstanceCount,
                rawGlbCount: scheduler.rawGlbCount, scheduledComponentCount: scheduler.scheduledComponentCount,
                scheduledGlbCount: scheduler.scheduledGlbCount, renderComponentCount: scheduler.renderComponentCount,
                renderGlbCount: scheduler.renderGlbCount, backend: scheduler.backend,
              },
              renderVisibility: neural.renderVisibility || null,
              renderRefresh: neural.renderRefresh || null,
              instancedBindings: neural.instancedBindings || null,
              actualRender: neural.actualRender || null,
              predictionGate: {
                mode: gate.mode, positionDelta: gate.positionDelta, yawDeltaDeg: gate.yawDeltaDeg,
                pitchDeltaDeg: gate.pitchDeltaDeg, minIntervalMs: gate.minIntervalMs,
                hasCommittedPrediction: gate.hasCommittedPrediction, forceNext: gate.forceNext,
              },
            },
            load: {
              queueLength: load.queueLength, inflightCount: load.inflightCount,
              pendingSceneInsertions: load.pendingSceneInsertions,
              activeDirectLoadCount: load.activeDirectLoadCount, totalIntegrated: load.totalIntegrated,
            },
          },
        };
      });
    } catch (error) {
      hostEvent(report, 'runtime.stats.error', { message: String(error && error.message || error) });
      break;
    }
    if (payload?.available) {
      const snapshot = compactRuntimeStats({ ...payload.stats, browserTimeMs: payload.browserTimeMs });
      addRuntimeSnapshot(report, snapshot);
      if (snapshot.neural?.prediction?.serial != null) firstPredictionSeen = true;
      if (snapshot.neural?.backend === 'worker-webgpu' || snapshot.neural?.prediction?.backend === 'worker-webgpu') firstWebGpuSeen = true;
    }
    const browserAudit = await drainBrowserAudit(page);
    if (browserAudit?.events) appendBrowserEvents(report, browserAudit.events);
    if (browserAudit?.droppedEvents) report.browserInstrumentationDroppedEvents = browserAudit.droppedEvents;
    const enough = waitFor === 'none' || (waitFor === 'prediction' && firstPredictionSeen) || (waitFor === 'webgpu' && firstWebGpuSeen);
    if (enough && waitFor !== 'none') {
      if (settleStarted == null) settleStarted = nowMs();
      if (nowMs() - settleStarted >= settleMs) break;
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
  }
  const finalAudit = await drainBrowserAudit(page);
  if (finalAudit?.events) appendBrowserEvents(report, finalAudit.events);
}

async function runBrowserAudit(options) {
  const url = buildAuditUrl(options);
  const staticAudit = createStaticAudit(options.scene);
  const report = createReport(options, staticAudit, url);
  report._maxEvents = options.maxEvents;
  hostEvent(report, 'audit.start', { url, staticAudit });
  let server = null;
  let playwright = null;
  let browser = null;
  let context = null;
  let page = null;
  const pendingResponseChecks = new Set();
  let targetBodyProbeCount = 0;
  try {
    if (options.startServer) {
      server = startLocalServer(options, report);
      await waitForHttp(`http://127.0.0.1:${options.port}/`, options.navigationTimeoutMs);
      hostEvent(report, 'server.ready', { url: `http://127.0.0.1:${options.port}/` });
    }
    playwright = await loadPlaywright();
    const chromium = playwright.chromium;
    if (!chromium || typeof chromium.launch !== 'function') throw new Error('The installed Playwright package has no Chromium launcher.');
    const executablePath = findChromiumExecutable(options.executablePath);
    if (!executablePath && options.executablePath) throw new Error(`Chromium executable does not exist: ${options.executablePath}`);
    const launchOptions = {
      headless: !options.headed,
      ...(executablePath ? { executablePath } : {}),
      args: [
        '--enable-unsafe-webgpu',
        '--ignore-gpu-blocklist',
        '--disable-gpu-sandbox',
        '--no-sandbox',
      ],
    };
    hostEvent(report, 'browser.launch.start', { launchOptions: { ...launchOptions, args: launchOptions.args }, executablePath });
    browser = await chromium.launch(launchOptions);
    context = await browser.newContext({ viewport: options.viewport });
    page = await context.newPage();
    report.run.browser = {
      name: 'chromium',
      version: browser.version(),
      executablePath: executablePath || 'Playwright-managed Chromium',
      headless: !options.headed,
      args: launchOptions.args,
    };
    const initScript = makeBrowserInitScript(options.maxEvents);
    await page.addInitScript({ content: initScript });
    page.on('request', (request) => {
      const requestUrl = request.url();
      const resourceClass = classifyResourceUrl(requestUrl, options.glbRegex);
      const record = {
        transport: 'playwright', method: request.method(), url: requestUrl,
        resourceClass, hostTimeMs: nowMs(), wallTime: isoNow(),
        isNavigation: request.isNavigationRequest(),
      };
      report.networkRequests.push({ phase: 'start', ...record });
      hostEvent(report, 'network.playwright.request', record);
    });
    page.on('response', (response) => {
      const request = response.request();
      const headers = response.headers();
      const record = {
        transport: 'playwright', method: request.method(), url: response.url(),
        resourceClass: classifyResourceUrl(response.url(), options.glbRegex),
        status: response.status(),
        contentType: headers['content-type'] || null,
        contentLength: headers['content-length'] ? Number(headers['content-length']) : null,
        contentRange: headers['content-range'] || null,
        hostTimeMs: nowMs(), wallTime: isoNow(),
      };
      const responseRecord = { phase: 'response', ...record };
      report.networkRequests.push(responseRecord);
      hostEvent(report, 'network.playwright.response', record);

      if (record.resourceClass !== 'target-glb') return;
      if (/text\/html/i.test(String(record.contentType || ''))) {
        responseRecord.contentValidation = 'invalid-html-content';
        return;
      }
      const contentLength = Number(record.contentLength || 0);
      if (targetBodyProbeCount >= MAX_TARGET_GLB_BODY_PROBES ||
          (contentLength > MAX_TARGET_GLB_BODY_PROBE_BYTES && response.status() !== 206)) {
        responseRecord.contentValidation = 'unprobed-size-or-limit';
        return;
      }

      targetBodyProbeCount += 1;
      const check = response.body().then((body) => {
        responseRecord.bodyBytesObserved = body.length;
        const magic = body.length >= 4 ? body.subarray(0, 4).toString('ascii') : '';
        responseRecord.glbMagic = magic || null;
        if (magic === 'glTF') {
          responseRecord.contentValidation = 'glb-magic';
          return;
        }
        const rangeStartsAtZero = /^bytes\s+0-/i.test(String(record.contentRange || ''));
        responseRecord.contentValidation = record.status === 206 && !rangeStartsAtZero
          ? 'unvalidated-range-fragment'
          : 'invalid-glb-magic';
      }).catch((error) => {
        responseRecord.contentValidation = 'probe-error';
        responseRecord.contentProbeError = String(error && error.message || error);
      });
      pendingResponseChecks.add(check);
      check.finally(() => pendingResponseChecks.delete(check));
    });
    page.on('requestfailed', (request) => {
      const failure = { url: request.url(), method: request.method(), error: request.failure()?.errorText || null, hostTimeMs: nowMs(), wallTime: isoNow() };
      report.requestFailures.push(failure);
      hostEvent(report, 'network.playwright.failed', failure);
    });
    page.on('pageerror', (error) => {
      const value = { message: error.message, stack: error.stack || null, hostTimeMs: nowMs(), wallTime: isoNow() };
      report.pageErrors.push(value);
      hostEvent(report, 'page.error', value);
    });
    page.on('console', (message) => {
      const value = { type: message.type(), text: message.text(), hostTimeMs: nowMs(), wallTime: isoNow() };
      report.console.push(value);
      if (report.console.length > options.maxEvents) report.console.shift();
      hostEvent(report, 'page.console', value);
      if (/webgpu|requestadapter|requestdevice|compute backend|neuralpvs|visibility|predict/i.test(value.text)) {
        hostEvent(report, 'runtime.console.marker', value);
      }
    });
    page.on('worker', (worker) => {
      hostEvent(report, 'worker.created', { url: worker.url() });
      if (options.workerEvaluate) {
        worker.evaluate((source) => {
          try { (0, eval)(source); return true; } catch (error) { return { error: String(error && error.message || error) }; }
        }, makeWorkerInitScript()).then((result) => {
          hostEvent(report, 'worker.instrumentation.injected', { url: worker.url(), result });
        }).catch((error) => {
          hostEvent(report, 'worker.instrumentation.blocked', { url: worker.url(), reason: String(error && error.message || error) });
        });
      }
    });

    hostEvent(report, 'navigation.start', { url });
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: options.navigationTimeoutMs });
    hostEvent(report, 'navigation.domcontentloaded', { url });
    await pollPageRuntime(page, report, options.durationMs, options.pollMs, options.waitFor, options.settleMs);
    await probePageState(page, report, 'observation-end');
    report.status = 'completed-observation';
  } catch (error) {
    const message = String(error && error.message || error);
    report.status = 'blocked';
    report.blockedReason = classifyBlockedReason(message, report, options);
    hostEvent(report, 'audit.blocked', { reason: report.blockedReason, message, stack: error?.stack || null });
  } finally {
    if (pendingResponseChecks.size > 0) {
      await Promise.allSettled(Array.from(pendingResponseChecks));
    }
    if (page) {
      try {
        const audit = await drainBrowserAudit(page);
        if (audit?.events) appendBrowserEvents(report, audit.events);
        if (audit?.droppedEvents) report.browserInstrumentationDroppedEvents = audit.droppedEvents;
      } catch (_) { /* page may already be closed */ }
    }
    if (report.status === 'not-started') report.status = 'blocked';
    report.summary = buildSummary(report);
    if (report.status === 'completed-observation') {
      const invalidTarget = Number(report.summary?.targetGlb?.responseSummary?.htmlFallbackResponses || 0) > 0 ||
        Number(report.summary?.targetGlb?.responseSummary?.invalidContentResponses || 0) > 0;
      const missingPrediction = options.waitFor !== 'none' && !report.summary?.firstPrediction;
      if (invalidTarget) report.status = 'completed-observation-with-invalid-target-glb';
      else if (missingPrediction) report.status = 'completed-observation-incomplete';
    }
    report.run.finishedAt = isoNow();
    report.run.elapsedHostMs = null;
    report.droppedEvents = Number(report._droppedEvents || 0);
    delete report._maxEvents;
    delete report._droppedEvents;
    delete report._lastRuntimeSignature;
    if (context) await context.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
    if (server) {
      hostEvent(report, 'server.stop', { pid: server.pid });
      try {
        if (server.pid) process.kill(-server.pid, 'SIGTERM');
        else server.kill('SIGTERM');
      } catch (_) {
        try { server.kill('SIGTERM'); } catch (__) { /* already exited */ }
      }
      setTimeout(() => {
        try {
          if (server.pid) process.kill(-server.pid, 'SIGKILL');
          else if (!server.killed) server.kill('SIGKILL');
        } catch (_) { /* already exited */ }
      }, 2000).unref();
    }
  }
  return report;
}

function classifyBlockedReason(message, report, options) {
  const lower = String(message).toLowerCase();
  if (/playwright is not installed|no chromium launcher/.test(lower)) return 'playwright-or-chromium-unavailable';
  if (/executable does not exist|browserType\.launch|executable|cannot find/.test(lower) && !report.run.browser) return 'chromium-launch-failed';
  if (/err_connection_refused|econnrefused|timed out waiting for local server|net::err_failed/.test(lower)) return 'viewer-server-unreachable';
  if (/timeout.*goto|navigation timeout|exceeded/.test(lower)) return 'viewer-navigation-timeout';
  if (report.pageErrors.length > 0) return 'frontend-runtime-error';
  if (options.waitFor !== 'none') return `required-${options.waitFor}-event-not-observed`;
  return 'browser-audit-error';
}

function runSelfTest() {
  const assertions = [];
  const assert = (condition, message) => {
    assertions.push({ message, passed: Boolean(condition) });
    if (!condition) throw new Error(message);
  };
  assert(classifyResourceUrl('https://host/scene_glbs/hkust/task-0/glb/LOD0/sub_3.glb') === 'target-glb', 'target scene GLB is classified');
  assert(classifyResourceUrl('https://host/assets/scenes/hkust/task-0/proxy/proxy.glb') === 'scene-proxy-or-other-glb', 'proxy GLB is not target geometry');
  assert(classifyResourceUrl('https://host/assets/neural_instance_culling/a/instance_pvs_assets.bin') === 'neural-weight', 'neural weight is classified');
  const base = createReport({ scene: DEFAULT_SCENE, url: 'http://127.0.0.1:3000/', durationMs: 1, pollMs: 1, waitFor: 'prediction', viewport: { width: 1, height: 1 } }, { checks: [] }, 'http://127.0.0.1:3000/');
  base._maxEvents = 100;
  pushEvent(base, { seq: 1, name: 'visibility.worker.postMessage', browserTimeMs: 20, details: { type: 'predict', serial: 1 } });
  pushEvent(base, { seq: 2, name: 'network.request.start', browserTimeMs: 21, details: { resourceClass: 'target-glb' } });
  assert(buildSummary(base).cold0.status === 'pass-no-target-glb-request-before-first-prediction', 'Cold-0 ordering pass is computed');
  const failing = createReport({ scene: DEFAULT_SCENE, url: 'http://127.0.0.1:3000/', durationMs: 1, pollMs: 1, waitFor: 'prediction', viewport: { width: 1, height: 1 } }, { checks: [] }, 'http://127.0.0.1:3000/');
  failing._maxEvents = 100;
  pushEvent(failing, { seq: 1, name: 'network.request.start', browserTimeMs: 19, details: { resourceClass: 'target-glb' } });
  pushEvent(failing, { seq: 2, name: 'visibility.worker.postMessage', browserTimeMs: 20, details: { type: 'predict', serial: 1 } });
  assert(buildSummary(failing).cold0.status === 'fail-target-glb-request-before-first-prediction', 'Cold-0 ordering failure is computed');
  const blocked = createReport({ scene: DEFAULT_SCENE, url: 'http://127.0.0.1:3000/', durationMs: 1, pollMs: 1, waitFor: 'prediction', viewport: { width: 1, height: 1 } }, { checks: [] }, 'http://127.0.0.1:3000/');
  blocked._maxEvents = 100;
  assert(buildSummary(blocked).cold0.status === 'blocked-no-first-prediction-observed', 'missing first prediction is not treated as pass');
  const invalidContent = createReport({ scene: DEFAULT_SCENE, url: 'http://127.0.0.1:3000/', durationMs: 1, pollMs: 1, waitFor: 'prediction', viewport: { width: 1, height: 1 } }, { checks: [] }, 'http://127.0.0.1:3000/');
  invalidContent._maxEvents = 100;
  invalidContent.networkRequests.push({
    phase: 'response', resourceClass: 'target-glb', status: 200,
    contentType: 'text/html', contentValidation: 'invalid-html-content',
  });
  assert(buildSummary(invalidContent).targetGlb.responseSummary.invalidContentResponses === 1, 'invalid target content is counted separately from request ordering');
  const output = { schemaVersion: 'm9-self-test-v1', scriptVersion: SCRIPT_VERSION, passed: true, assertions };
  console.log(JSON.stringify(output, null, 2));
  return output;
}

function writeReport(report, options) {
  const serialized = JSON.stringify(report, null, 2) + '\n';
  if (options.out) {
    fs.mkdirSync(path.dirname(options.out), { recursive: true });
    fs.writeFileSync(options.out, serialized);
  }
  if (options.jsonlOut) {
    fs.mkdirSync(path.dirname(options.jsonlOut), { recursive: true });
    const lines = report.events.map((event) => JSON.stringify(event)).join('\n') + '\n';
    fs.writeFileSync(options.jsonlOut, lines);
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }
  if (options.selfTest) {
    runSelfTest();
    return;
  }
  const report = await runBrowserAudit(options);
  writeReport(report, options);
  console.log(JSON.stringify({
    schemaVersion: report.schemaVersion,
    status: report.status,
    blockedReason: report.blockedReason,
    cold0: report.summary?.cold0,
    webgpu: report.summary?.webgpu,
    visibility: report.summary?.visibility,
    targetGlb: report.summary?.targetGlb,
    report: options.out || null,
    jsonl: options.jsonlOut || null,
  }, null, 2));
  if (report.status === 'blocked') process.exitCode = 2;
}

main().catch((error) => {
  console.error(`[${SCRIPT_VERSION}] ${error.stack || error.message || error}`);
  process.exitCode = 1;
});
