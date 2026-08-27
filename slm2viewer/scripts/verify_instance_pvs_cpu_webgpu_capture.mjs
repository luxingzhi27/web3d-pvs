#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PerspectiveCamera, Quaternion, Vector3 } from 'three';
import { InstancePVSCPU } from '../src/InstancePVSCPU.js';

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
const assetRoot = path.join(
  viewerRoot,
  'public/assets/neural_instance_culling/pvs_mainline_v4',
);
const server = http.createServer((request, response) => {
  try {
    const relative = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname)
      .replace(/^\/+/, '');
    const file = path.resolve(assetRoot, relative);
    if (!file.startsWith(`${assetRoot}${path.sep}`)) throw new Error('Forbidden');
    const data = fs.readFileSync(file);
    response.writeHead(200, {
      'Content-Type': relative.endsWith('.json') ? 'application/json' : 'application/octet-stream',
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
  runtime = new InstancePVSCPU(baseUrl, { debugLogging: true });
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
  const cpu = await runtime.predict(camera, {
    candidateCamera,
    renderCamera: camera,
    prefetchThreshold: 0.04,
  });

  const gpuCandidateIds = capture.candidateInstanceIds.map(Number);
  const gpuScores = capture.candidateScores.map(Number);
  const cpuCandidateIds = Array.from(cpu.rawCandidateIds);
  const cpuScores = Array.from(cpu.scores);
  const gpuScoreById = new Map(gpuCandidateIds.map((id, index) => [id, gpuScores[index]]));
  assert.deepEqual(cpuCandidateIds, gpuCandidateIds.slice().sort((a, b) => a - b));
  assert.deepEqual(Array.from(cpu.componentModelList), capture.visibleInstanceIds.map(Number));
  assert.deepEqual(
    Array.from(cpu.renderComponentModelList),
    capture.renderVisibleInstanceIds.map(Number),
  );
  assert.equal(cpuScores.length, gpuScores.length);
  let maximumAbsoluteError = 0;
  let meanAbsoluteError = 0;
  for (let index = 0; index < cpuScores.length; index += 1) {
    const difference = Math.abs(cpuScores[index] - gpuScoreById.get(cpuCandidateIds[index]));
    maximumAbsoluteError = Math.max(maximumAbsoluteError, difference);
    meanAbsoluteError += difference;
  }
  meanAbsoluteError /= Math.max(1, cpuScores.length);
  assert.ok(maximumAbsoluteError < 0.001, `CPU/WebGPU max score error is ${maximumAbsoluteError}.`);
  console.log(JSON.stringify({
    candidateCount: cpuCandidateIds.length,
    visibleInstanceCount: cpu.componentModelList.length,
    renderVisibleInstanceCount: cpu.renderComponentModelList.length,
    meanAbsoluteError,
    maximumAbsoluteError,
  }, null, 2));
  console.log('V4 CPU/WebGPU capture parity passed.');
} finally {
  runtime?.dispose();
  await new Promise((resolve) => server.close(resolve));
}
