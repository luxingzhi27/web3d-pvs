import assert from 'node:assert/strict';
import test from 'node:test';

import {
  POINT60_GT_FOV_Y_DEG,
  parseArgs,
  validateModel66Plan,
  validatePoint60Plan,
} from '../run_sampler.mjs';

const SCRIPT = new URL('../run_sampler.mjs', import.meta.url).pathname;

function args(...values) {
  return parseArgs(['node', SCRIPT, ...values]);
}

function point60Pose(overrides = {}) {
  return {
    pose_index: 7,
    viewcell_id: 7,
    subpose_id: 0,
    camera_pos: [1, 2, 3],
    camera_forward: [0, 0, -1],
    fov_y: POINT60_GT_FOV_Y_DEG,
    render_fov_y: POINT60_GT_FOV_Y_DEG,
    aspect: 16 / 9,
    width: 512,
    height: 288,
    ...overrides,
  };
}

test('default sampler remains the strict 66 degree hardware path', () => {
  const parsed = args();
  assert.equal(parsed.point60Gt, false);
  assert.equal(parsed.fovYDeg, 66);
  assert.equal(parsed.requireHardwareGpu, true);
});

test('point60 mode selects 60 degrees and keeps the hardware gate', () => {
  const parsed = args('--point60-gt');
  assert.equal(parsed.point60Gt, true);
  assert.equal(parsed.fovYDeg, POINT60_GT_FOV_Y_DEG);
  assert.equal(parsed.requireHardwareGpu, true);
});

test('point60 mode rejects another FOV in either argument order', () => {
  assert.throws(() => args('--point60-gt', '--fov-y-deg', '66'), /only supports/);
  assert.throws(() => args('--fov-y-deg', '66', '--point60-gt'), /only supports/);
  assert.throws(() => args('--fov-y-deg', '60'), /model FOV/);
});

test('point60 mode rejects software GPU flags', () => {
  assert.throws(() => args('--point60-gt', '--allow-software-gpu'), /hardware GPU gate/);
  assert.throws(() => args('--allow-software-gpu', '--point60-gt'), /hardware GPU gate/);
});

test('point60 plan validation requires canonical IDs and one exact viewport group', () => {
  const viewport = validatePoint60Plan([point60Pose({ pose_index: 0 })]);
  assert.deepEqual(viewport, { width: 512, height: 288, aspect: 16 / 9 });
  assert.throws(() => validatePoint60Plan([point60Pose({ subpose_id: 1 })]), /subpose_id=0/);
  assert.throws(() => validatePoint60Plan([point60Pose({ viewcell_id: -1 })]), /viewcell_id/);
  assert.throws(() => validatePoint60Plan([
    point60Pose(),
    point60Pose({ pose_index: 8, viewcell_id: 8, aspect: 1, width: 288 }),
  ]), /one plan per exact/);
});

test('default plan rows cannot override the strict 66 degree sampler FOV', () => {
  assert.doesNotThrow(() => validateModel66Plan([{ fov_y: 66 }]));
  assert.doesNotThrow(() => validateModel66Plan([{ render_fov_y: 60 }]));
  assert.throws(() => validateModel66Plan([{ fov_y: 60 }]), /fov_y=66/);
});
