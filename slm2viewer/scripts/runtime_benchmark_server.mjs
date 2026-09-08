#!/usr/bin/env node
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, resolve, sep } from 'node:path';
import { randomBytes, randomUUID } from 'node:crypto';

const root = resolve(process.env.PVS_BENCHMARK_STATIC_ROOT || 'public');
const outputDir = resolve(
  process.env.PVS_BENCHMARK_OUTPUT_DIR
    || '../neural_instance_culling/benchmark/out/pvs_v4_frontend_inference_latency_v1/browser_uploads',
);
const host = process.env.PVS_BENCHMARK_HOST || '127.0.0.1';
const port = Number(process.env.PVS_BENCHMARK_PORT || 8321);
const maxPayloadBytes = 8 * 1024 * 1024;
const sessions = new Map();

const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.ico': 'image/x-icon',
  '.png': 'image/png',
};

function sendJson(response, status, value) {
  const body = `${JSON.stringify(value)}\n`;
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store',
  });
  response.end(body);
}

function sameOrigin(request) {
  const origin = request.headers.origin;
  if (!origin) return true;
  try {
    const url = new URL(origin);
    const forwardedHost = String(request.headers['x-forwarded-host'] || request.headers.host || '');
    return url.host === forwardedHost;
  } catch (_) {
    return false;
  }
}

async function readJsonBody(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxPayloadBytes) throw new Error('payload-too-large');
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
  } catch (_) {
    throw new Error('invalid-json');
  }
}

function finiteNonnegative(value) {
  return Number.isFinite(Number(value)) && Number(value) >= 0;
}

function validateResult(payload) {
  if (!payload || payload.schema !== 'pvs-v4-browser-runtime-result-v1') {
    throw new Error('unsupported result schema');
  }
  const label = String(payload.device?.label || '').trim();
  if (!label || label.length > 120) throw new Error('device label is missing or too long');
  if (payload.workload?.poseCount !== 684 || payload.workload?.split !== 'test') {
    throw new Error('result does not use the frozen 684-pose test workload');
  }
  if (!Array.isArray(payload.sessions) || payload.sessions.length < 1 || payload.sessions.length > 5) {
    throw new Error('result must contain one to five sessions');
  }
  for (const session of payload.sessions) {
    if (!Array.isArray(session.samples) || session.samples.length !== 684) {
      throw new Error('every completed session must contain 684 samples');
    }
    for (const sample of session.samples) {
      if (!Number.isInteger(sample.poseId)
          || !Number.isInteger(sample.candidateCount)
          || sample.candidateCount < 0
          || sample.candidateCount > 18831
          || !finiteNonnegative(sample.modelInferenceMs)
          || !finiteNonnegative(sample.submitCompletionMs)
          || (sample.gpuKernelMs != null && !finiteNonnegative(sample.gpuKernelMs))) {
        throw new Error('sample contains invalid pose, candidate, or timing values');
      }
    }
  }
  return {
    label,
    formalReady: payload.environment?.secureContext === true
      && payload.environment?.hardwareGate?.hardware === true
      && Number(payload.environment?.visibilityViolations || 0) === 0,
  };
}

function cleanExpiredSessions() {
  const now = Date.now();
  for (const [token, expiresAt] of sessions) {
    if (expiresAt <= now) sessions.delete(token);
  }
}

async function createUploadSession(request, response) {
  if (!sameOrigin(request)) return sendJson(response, 403, { error: 'cross-origin request rejected' });
  cleanExpiredSessions();
  const token = randomBytes(24).toString('base64url');
  sessions.set(token, Date.now() + 2 * 60 * 60 * 1000);
  return sendJson(response, 201, { token, expiresInSeconds: 7200 });
}

async function storeResult(request, response) {
  if (!sameOrigin(request)) return sendJson(response, 403, { error: 'cross-origin request rejected' });
  cleanExpiredSessions();
  const token = String(request.headers['x-pvs-session'] || '');
  if (!sessions.has(token)) return sendJson(response, 403, { error: 'invalid or expired upload session' });
  sessions.delete(token);
  let payload;
  try {
    payload = await readJsonBody(request);
  } catch (error) {
    return sendJson(response, error.message === 'payload-too-large' ? 413 : 400, { error: error.message });
  }
  let validation;
  try {
    validation = validateResult(payload);
  } catch (error) {
    return sendJson(response, 422, { error: error.message });
  }

  await mkdir(outputDir, { recursive: true });
  const receiptId = randomUUID();
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const safeLabel = validation.label.replace(/[^a-zA-Z0-9_-]+/g, '_').slice(0, 48) || 'device';
  const stored = {
    ...payload,
    serverReceipt: {
      receiptId,
      receivedAt: new Date().toISOString(),
      formalReady: validation.formalReady,
    },
  };
  const filename = `${timestamp}_${safeLabel}_${receiptId}.json`;
  await writeFile(resolve(outputDir, filename), `${JSON.stringify(stored, null, 2)}\n`, { flag: 'wx' });
  await appendFile(
    resolve(outputDir, 'receipts.jsonl'),
    `${JSON.stringify({ receiptId, filename, label: validation.label, formalReady: validation.formalReady })}\n`,
  );
  return sendJson(response, 201, { receiptId, formalReady: validation.formalReady });
}

async function serveStatic(request, response, pathname) {
  let relative = decodeURIComponent(pathname).replace(/^\/+/, '');
  if (!relative) relative = 'runtime-benchmark.html';
  const path = resolve(root, relative);
  if (path !== root && !path.startsWith(`${root}${sep}`)) {
    return sendJson(response, 403, { error: 'invalid path' });
  }
  try {
    const body = await readFile(path);
    response.writeHead(200, {
      'Content-Type': mimeTypes[extname(path).toLowerCase()] || 'application/octet-stream',
      'Content-Length': body.length,
      'Cache-Control': path.endsWith('.html') ? 'no-cache' : 'public, max-age=3600',
    });
    response.end(body);
  } catch (error) {
    sendJson(response, error.code === 'ENOENT' ? 404 : 500, { error: 'resource unavailable' });
  }
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url || '/', `http://${request.headers.host || 'localhost'}`);
  try {
    if (request.method === 'POST' && url.pathname === '/api/pvs-runtime-session') {
      await createUploadSession(request, response);
      return;
    }
    if (request.method === 'POST' && url.pathname === '/api/pvs-runtime-results') {
      await storeResult(request, response);
      return;
    }
    if (request.method === 'GET' && url.pathname === '/api/pvs-runtime-health') {
      sendJson(response, 200, { status: 'ok', schema: 'pvs-v4-runtime-receiver-v1' });
      return;
    }
    if (request.method === 'GET' || request.method === 'HEAD') {
      await serveStatic(request, response, url.pathname);
      return;
    }
    sendJson(response, 405, { error: 'method not allowed' });
  } catch (error) {
    console.error(error);
    if (!response.headersSent) sendJson(response, 500, { error: 'internal server error' });
    else response.end();
  }
});

server.listen(port, host, () => {
  console.log(`PVS runtime benchmark server listening on http://${host}:${port}`);
  console.log(`Static root: ${root}`);
  console.log(`Result output: ${outputDir}`);
});
