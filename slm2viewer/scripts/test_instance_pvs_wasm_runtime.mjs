#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PerspectiveCamera, Vector3 } from 'three';
import { InstancePVSWasm } from '../src/InstancePVSWasm.js';

const viewerDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modelRoot = path.join(viewerDir, 'assets/neural_instance_culling/pvs_mainline_v4');
const wasmPath = path.join(viewerDir, 'assets/wasm/instance_pvs_v4.wasm');
const config = JSON.parse(fs.readFileSync(path.join(viewerDir, 'assets/config.json'), 'utf8'));
const scene = config.scenes['hkust-v3'];

const server = http.createServer((request, response) => {
  const url = new URL(request.url, 'http://127.0.0.1');
  const pathname = decodeURIComponent(url.pathname);
  const relative = pathname.startsWith('/model/') ? pathname.slice('/model/'.length) : '';
  const absolute = pathname === '/assets/wasm/instance_pvs_v4.wasm'
    ? wasmPath
    : path.resolve(modelRoot, relative);
  if (absolute !== wasmPath && !absolute.startsWith(`${modelRoot}${path.sep}`)) {
    response.writeHead(403).end();
    return;
  }
  fs.readFile(absolute, (error, data) => {
    if (error) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      'Content-Type': absolute.endsWith('.json')
        ? 'application/json'
        : (absolute.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream'),
      'Content-Length': data.byteLength,
    });
    response.end(data);
  });
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const baseUrl = `http://127.0.0.1:${server.address().port}`;
let runtime;
try {
  runtime = new InstancePVSWasm(`${baseUrl}/model`, {
    wasmUrl: `${baseUrl}/assets/wasm/instance_pvs_v4.wasm`,
  });
  await runtime.init();
  assert.equal(runtime.backend, 'wasm-simd-v4');
  assert.equal(runtime.wasm.wasm_simd_enabled(), 1);
  const camera = new PerspectiveCamera(60, 694 / 552, 0.1, 20000);
  camera.position.fromArray(scene.cameraPostion);
  camera.lookAt(new Vector3().fromArray(scene.cameraTarget));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  const forward = camera.getWorldDirection(new Vector3()).normalize();
  const candidateCamera = new PerspectiveCamera(66, camera.aspect, camera.near, camera.far);
  candidateCamera.position.copy(camera.position).addScaledVector(
    forward,
    -Number(runtime.meta.query.candidateCameraBackOffsetM),
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
  assert.equal(prediction.backend, 'wasm-simd-v4');
  assert.equal(prediction.timings.workerWasmNeuralInference, true);

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
  console.log('Instance PVS full WASM SIMD runtime test passed.');
} finally {
  runtime?.dispose();
  await new Promise((resolve) => server.close(resolve));
}
