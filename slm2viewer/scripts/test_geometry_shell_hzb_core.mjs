#!/usr/bin/env node
import assert from 'node:assert/strict';
import {
  buildMaxMipChain,
  projectAabbConservatively,
  queryAabbAgainstMaxMip,
  queryAabbsAgainstMaxMip,
  queryPoint60,
  queryRegion66,
} from '../src/GeometryShellHZBCore.js';
import {
  GEOMETRY_SHELL_DEPTH_FRAGMENT_SHADER,
  GEOMETRY_SHELL_DEPTH_VERTEX_SHADER,
  GEOMETRY_SHELL_MIP_SHADER,
  GEOMETRY_SHELL_QUERY_SHADER,
} from '../src/GeometryShellHZBShaders.js';
import { GEOMETRY_SHELL_HZB_SAMPLE_COUNT } from '../src/GeometryShellHZB.js';

assert.equal(GEOMETRY_SHELL_HZB_SAMPLE_COUNT, 1);

const levelZero = new Float32Array([
  1, 2, 3,
  4, 5, 6,
  7, 8, 9,
]);
const levels = buildMaxMipChain(levelZero, 3, 3);
assert.deepEqual(Array.from(levels[1].data), [5, 6, 8, 9]);
assert.deepEqual(Array.from(levels[2].data), [9]);

const camera = {
  position: [0, 0, 0],
  forward: [0, 0, -1],
  up: [0, 1, 0],
  fovYDeg: 60,
  aspect: 1,
  near: 0.1,
  far: 100,
};
const plane = buildMaxMipChain(new Float32Array(16).fill(5), 4, 4);
const occludedAabb = [-0.2, -0.2, -10, 0.2, 0.2, -9];
const visibleAabb = [-0.2, -0.2, -4, 0.2, 0.2, -3];
const occludedProjection = projectAabbConservatively(occludedAabb, camera);
assert.equal(occludedProjection.valid, true);
assert.equal(queryAabbAgainstMaxMip(plane, occludedProjection.rect, occludedProjection.candidateNear).occluded, true);
assert.equal(queryAabbAgainstMaxMip(plane, projectAabbConservatively(visibleAabb, camera).rect, 3).visible, true);

const verticallyAsymmetricDepth = buildMaxMipChain(new Float32Array([
  20, 20, 20, 20,
  20, 20, 20, 20,
  2, 2, 2, 2,
  2, 2, 2, 2,
]), 4, 4);
const upperAabb = [-0.2, 1.8, -5, 0.2, 2.2, -4.8];
const upperResult = queryAabbsAgainstMaxMip(
  verticallyAsymmetricDepth,
  camera,
  new Float32Array(upperAabb),
  new Uint32Array([6]),
);
assert.deepEqual(Array.from(upperResult.visibleIds), [6]);
assert.equal(upperResult.records[0].hzbMax, 20);

const inside = queryAabbsAgainstMaxMip(
  plane,
  camera,
  new Float32Array([-1, -1, -1, 1, 1, 1, ...occludedAabb]),
  new Uint32Array([7, 8]),
);
assert.deepEqual(Array.from(inside.visibleIds), [7]);
assert.equal(inside.uncertainCount, 1);
assert.equal(inside.records[0].reason, 'camera-inside-aabb');
const regularCandidate = queryAabbsAgainstMaxMip(
  plane,
  camera,
  new Float32Array(occludedAabb),
  new Uint32Array([8]),
);
assert.deepEqual(Array.from(regularCandidate.visibleIds), []);
assert.equal(regularCandidate.uncertainCount, 0);
assert.equal(regularCandidate.records[0].occluded, true);

const nearCrossing = projectAabbConservatively([0.5, -0.2, -2, 0.7, 0.2, 0], camera);
assert.equal(nearCrossing.uncertain, true);
assert.equal(nearCrossing.reason, 'near-plane-crossing');
const nearResult = queryAabbsAgainstMaxMip(
  plane,
  camera,
  new Float32Array([0.5, -0.2, -2, 0.7, 0.2, 0]),
  new Uint32Array([9]),
);
assert.deepEqual(Array.from(nearResult.visibleIds), [9]);
assert.equal(nearResult.records[0].reason, 'near-plane-crossing');
const point = queryPoint60(plane, camera, new Float32Array(occludedAabb), new Uint32Array([42]));
assert.equal(point.mode, 'Point60');
assert.deepEqual(Array.from(point.visibleIds), []);
const region = queryRegion66(
  [plane, buildMaxMipChain(new Float32Array(16).fill(20), 4, 4)],
  [camera, camera],
  new Float32Array(occludedAabb),
  new Uint32Array([42]),
);
assert.equal(region.mode, 'Region66');
assert.equal(region.passCount, 2);
assert.deepEqual(Array.from(region.perPose[0].visibleIds), []);
assert.deepEqual(Array.from(region.perPose[1].visibleIds), [42]);
assert.deepEqual(Array.from(region.visibleIds), [42]);

assert.match(GEOMETRY_SHELL_DEPTH_VERTEX_SHADER, /matrix_column_0/);
assert.match(GEOMETRY_SHELL_MIP_SHADER, /maximum = max/);
assert.match(GEOMETRY_SHELL_MIP_SHADER, /source_width/);
assert.doesNotMatch(GEOMETRY_SHELL_MIP_SHADER, /textureDimensions/);
assert.doesNotMatch(GEOMETRY_SHELL_QUERY_SHADER, /textureDimensions/);
assert.doesNotMatch(GEOMETRY_SHELL_DEPTH_FRAGMENT_SHADER, /instance_id/);
assert.match(GEOMETRY_SHELL_QUERY_SHADER, /depth <= near_plane/);
assert.match(GEOMETRY_SHELL_QUERY_SHADER, /hzb_max \+ uniforms\.projection\.z < minimum_depth/);
assert.doesNotMatch(GEOMETRY_SHELL_QUERY_SHADER, /instance_occluder|visible_occluder_words/);

console.log('Geometry-shell HZB core tests passed.');
