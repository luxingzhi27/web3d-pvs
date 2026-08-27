#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PerspectiveCamera, Quaternion, Vector3 } from 'three';
import { InstancePVSWasm } from '../src/InstancePVSWasm.js';

const captureArg = process.argv.find((value) => value.startsWith('--capture='));
const capturePath = captureArg ? captureArg.slice('--capture='.length) : null;
if (!capturePath || !fs.existsSync(capturePath)) {
  throw new Error('Pass an existing WebGPU capture as --capture=<path>.');
}
const capture = JSON.parse(fs.readFileSync(capturePath, 'utf8'));
if (!String(capture.backend || '').includes('webgpu-v4')) {
  throw new Error(`Capture backend is not V4 WebGPU: ${capture.backend || 'missing'}.`);
}

const viewerRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modelRoot = path.join(viewerRoot, 'assets/neural_instance_culling/pvs_mainline_v4');
const wasmPath = path.join(viewerRoot, 'assets/wasm/instance_pvs_v4.wasm');
const server = http.createServer((request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    const relative = pathname.startsWith('/model/') ? pathname.slice('/model/'.length) : '';
    const file = pathname === '/assets/wasm/instance_pvs_v4.wasm'
      ? wasmPath
      : path.resolve(modelRoot, relative);
    if (file !== wasmPath && !file.startsWith(`${modelRoot}${path.sep}`)) throw new Error('Forbidden');
    const data = fs.readFileSync(file);
    response.writeHead(200, {
      'Content-Type': file.endsWith('.json')
        ? 'application/json'
        : (file.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream'),
      'Content-Length': data.byteLength,
    });
    response.end(data);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});

let runtime;
try {
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  runtime = new InstancePVSWasm(`${baseUrl}/model`, {
    debugLogging: true,
    wasmUrl: `${baseUrl}/assets/wasm/instance_pvs_v4.wasm`,
  });
  await runtime.init();
  const source = capture.camera;
  const camera = new PerspectiveCamera(source.fov, source.aspect, source.near, source.far);
  camera.position.fromArray(source.position);
  camera.quaternion.copy(new Quaternion().fromArray(source.quaternion));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  const forward = camera.getWorldDirection(new Vector3()).normalize();
  const candidateCamera = new PerspectiveCamera(66, source.aspect, source.near, source.far);
  candidateCamera.position.copy(camera.position).addScaledVector(
    forward,
    -Number(runtime.meta.query.candidateCameraBackOffsetM),
  );
  candidateCamera.quaternion.copy(camera.quaternion);
  candidateCamera.updateProjectionMatrix();
  candidateCamera.updateMatrixWorld(true);
  const wasm = await runtime.predict(camera, {
    candidateCamera,
    renderCamera: camera,
    prefetchThreshold: 0.04,
  });

  const gpuCandidateIds = capture.candidateInstanceIds.map(Number);
  const gpuScores = capture.candidateScores.map(Number);
  const wasmCandidateIds = Array.from(wasm.rawCandidateIds);
  const wasmScores = Array.from(wasm.scores);
  const gpuScoreById = new Map(gpuCandidateIds.map((id, index) => [id, gpuScores[index]]));
  assert.deepEqual(wasmCandidateIds, gpuCandidateIds.slice().sort((a, b) => a - b));
  assert.deepEqual(Array.from(wasm.componentModelList), capture.visibleInstanceIds.map(Number));
  assert.deepEqual(
    Array.from(wasm.renderComponentModelList),
    capture.renderVisibleInstanceIds.map(Number),
  );
  assert.equal(wasmScores.length, gpuScores.length);
  let maximumAbsoluteError = 0;
  let meanAbsoluteError = 0;
  for (let index = 0; index < wasmScores.length; index += 1) {
    const difference = Math.abs(wasmScores[index] - gpuScoreById.get(wasmCandidateIds[index]));
    maximumAbsoluteError = Math.max(maximumAbsoluteError, difference);
    meanAbsoluteError += difference;
  }
  meanAbsoluteError /= Math.max(1, wasmScores.length);
  assert.ok(maximumAbsoluteError < 0.001, `WASM/WebGPU max score error is ${maximumAbsoluteError}.`);
  console.log(JSON.stringify({
    candidateCount: wasmCandidateIds.length,
    visibleInstanceCount: wasm.componentModelList.length,
    renderVisibleInstanceCount: wasm.renderComponentModelList.length,
    meanAbsoluteError,
    maximumAbsoluteError,
    inferenceMs: wasm.timings.inferenceMs,
  }, null, 2));
  console.log('V4 WASM SIMD/WebGPU capture parity passed.');
} finally {
  runtime?.dispose();
  await new Promise((resolve) => server.close(resolve));
}
