#!/usr/bin/env node
import assert from 'node:assert/strict';
import {
  createFloat16Lookup,
  linearInto,
  sigmoid,
  unpackFloat16,
} from '../src/InstancePVSCPUMath.js';

const halves = new Uint16Array([0x0000, 0x3c00, 0xc000, 0x7bff]);
const unpacked = unpackFloat16(new Uint32Array(halves.buffer), halves.length, createFloat16Lookup());
assert.deepEqual(Array.from(unpacked), [0, 1, -2, 65504]);
assert.equal(sigmoid(0), 0.5);
assert.ok(Math.abs(sigmoid(2) + sigmoid(-2) - 1) < 1e-12);

const weights = new Float32Array([
  1, 2,
  -1, 0.5,
  0.25, -0.5,
]);
const output = new Float32Array(2);
linearInto(new Float32Array([2, 3]), output, weights, 0, 4, 'none');
assert.ok(Math.abs(output[0] - 8.25) < 1e-6);
assert.ok(Math.abs(output[1] - -1) < 1e-6);

console.log('Instance PVS CPU math tests passed.');
