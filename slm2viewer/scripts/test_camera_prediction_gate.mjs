#!/usr/bin/env node
import assert from 'node:assert/strict';
import { PerspectiveCamera } from 'three';
import { CameraPredictionGate } from '../src/CameraPredictionGate.js';

function makeCamera({ x = 0, y = 0, z = 0, yaw = 0, pitch = 0, roll = 0, fov = 60 } = {}) {
  const camera = new PerspectiveCamera(fov, 16 / 9, 0.1, 1000);
  camera.position.set(x, y, z);
  camera.rotation.set(pitch, yaw, roll, 'YXZ');
  camera.updateMatrixWorld(true);
  return camera;
}

function committedGate() {
  const gate = new CameraPredictionGate({
    mode: 'viewcell',
    positionDelta: 2.5,
    yawDeltaDeg: 6,
    pitchDeltaDeg: 5,
    fovDeltaDeg: 2,
    minIntervalMs: 350,
  });
  gate.commit(makeCamera(), 0);
  return gate;
}

const gate = committedGate();
const state = gate.getState();
assert.equal(state.positionDelta, 2);
assert.equal(state.positionShape, 'horizontal_disk');
assert.equal(state.orientationMode, 'fixed_quaternion');
assert.equal(state.renderFovYDeg, 60);
assert.equal(state.modelInputFovYDeg, 66);

assert.equal(gate.shouldPredict(makeCamera({ x: 1.99 }), 100), false);
assert.equal(gate.shouldPredict(makeCamera({ x: 2 }), 100), true);

const verticalGate = committedGate();
assert.equal(verticalGate.shouldPredict(makeCamera({ y: 0.01 }), 100), true);

const orientationGate = committedGate();
assert.equal(orientationGate.shouldPredict(makeCamera({ yaw: 0.5 * Math.PI / 180 }), 100), true);

const fovGate = committedGate();
assert.equal(fovGate.shouldPredict(makeCamera({ fov: 61 }), 100), true);

// A boundary crossing must not be delayed by the prediction minimum interval.
const boundaryGate = committedGate();
assert.equal(boundaryGate.shouldPredict(makeCamera({ x: 3 }), 1), true);

console.log('CameraPredictionGate HKUST view-cell contract tests passed.');
