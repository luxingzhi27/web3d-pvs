#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PerspectiveCamera, Vector3 } from 'three';
import { InstancePVSCPU } from '../src/InstancePVSCPU.js';

const viewerDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const assetRoot = path.join(
  viewerDir,
  'public/assets/neural_instance_culling/pvs_mainline_v4',
);
const config = JSON.parse(fs.readFileSync(path.join(viewerDir, 'assets/config.json'), 'utf8'));
const scene = config.scenes['hkust-v3'];

const server = http.createServer((request, response) => {
  const relative = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname).replace(/^\/+/, '');
  const absolute = path.resolve(assetRoot, relative);
  if (!absolute.startsWith(`${assetRoot}${path.sep}`)) {
    response.writeHead(403).end();
    return;
  }
  fs.readFile(absolute, (error, data) => {
    if (error) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      'Content-Type': relative.endsWith('.json') ? 'application/json' : 'application/octet-stream',
      'Content-Length': data.byteLength,
    });
    response.end(data);
  });
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const address = server.address();
const baseUrl = `http://127.0.0.1:${address.port}`;
let runtime;
try {
  runtime = new InstancePVSCPU(baseUrl);
  await runtime.init();
  assert.equal(runtime.backend, 'cpu-js-v4');
  const meta = runtime.meta;
  const camera = new PerspectiveCamera(60, 694 / 552, 0.1, 20000);
  camera.position.fromArray(scene.cameraPostion);
  camera.lookAt(new Vector3().fromArray(scene.cameraTarget));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  const forward = camera.getWorldDirection(new Vector3()).normalize();
  const candidateCamera = new PerspectiveCamera(66, camera.aspect, camera.near, camera.far);
  candidateCamera.position.copy(camera.position).addScaledVector(
    forward,
    -Number(meta.query.candidateCameraBackOffsetM),
  );
  candidateCamera.quaternion.copy(camera.quaternion);
  candidateCamera.updateProjectionMatrix();
  candidateCamera.updateMatrixWorld(true);

  const prediction = await runtime.predict(camera, {
    candidateCamera,
    renderCamera: camera,
    prefetchThreshold: 0.04,
  });
  assert.ok(prediction.candidateCount > 0);
  assert.ok(prediction.componentModelList.length > 0);
  assert.ok(prediction.componentModelList.length < prediction.candidateCount);
  assert.ok(prediction.modelList.length > 0);
  assert.ok(prediction.glbQueue.length >= prediction.modelList.length);
  assert.equal(prediction.backend, 'cpu-js-v4');
  assert.equal(prediction.timings.workerCpuNeuralInference, true);

  const filtered = await runtime.refilter(camera);
  assert.equal(filtered.renderInstanceCount, prediction.renderComponentModelList.length);
  assert.equal(filtered.renderGlbCount, prediction.renderModelList.length);
  console.log(JSON.stringify({
    backend: prediction.backend,
    candidateCount: prediction.candidateCount,
    visibleInstanceCount: prediction.componentModelList.length,
    visibleGlbCount: prediction.modelList.length,
    inferenceMs: prediction.timings.inferenceMs,
  }));
  console.log('Instance PVS full CPU runtime test passed.');
} finally {
  runtime?.dispose();
  await new Promise((resolve) => server.close(resolve));
}
