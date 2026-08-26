#!/usr/bin/env node
import assert from 'node:assert/strict';
import { applyDenseInstancedDelta, initializeDenseInstancedState } from '../src/DenseInstancedSlots.js';
import { IdBitsetState } from '../src/IdBitsetState.js';

const bitset = new IdBitsetState(70);
let delta = bitset.replace([1, 3, 33, 69]);
assert.deepEqual(Array.from(delta.added), [1, 3, 33, 69]);
assert.equal(bitset.count, 4);
delta = bitset.applyDelta([2, 33], [1, 68]);
assert.deepEqual(Array.from(delta.added), [2]);
assert.deepEqual(Array.from(delta.removed), [1]);
assert.deepEqual(bitset.toIds(), [2, 3, 33, 69]);

function makeMeshState(instanceCount) {
  const source = new Float32Array(instanceCount * 16);
  for (let index = 0; index < instanceCount; index += 1) source[index * 16] = index + 1;
  const ranges = [];
  return {
    originalCount: instanceCount,
    originalMatrixArray: source,
    mesh: {
      count: instanceCount,
      visible: true,
      frustumCulled: true,
      instanceMatrix: {
        array: new Float32Array(source),
        needsUpdate: false,
        clearUpdateRanges() { ranges.length = 0; },
        addUpdateRange(start, count) { ranges.push([start, count]); },
      },
      ranges,
    },
  };
}

const meshState = makeMeshState(6);
const state = initializeDenseInstancedState({
  originalCount: 6,
  meshStates: [meshState],
});
applyDenseInstancedDelta(state, [1, 3, 5], []);
assert.deepEqual(state.activeSourceIndices, [1, 3, 5]);
assert.deepEqual(Array.from(state.slotBySourceIndex), [-1, 0, -1, 1, -1, 2]);
assert.equal(meshState.mesh.instanceMatrix.array[0], 2);
assert.equal(meshState.mesh.instanceMatrix.array[16], 4);
assert.equal(meshState.mesh.instanceMatrix.array[32], 6);

applyDenseInstancedDelta(state, [2], [3]);
assert.deepEqual(state.activeSourceIndices, [1, 5, 2]);
assert.equal(state.slotBySourceIndex[5], 1);
assert.equal(state.slotBySourceIndex[2], 2);
assert.equal(meshState.mesh.instanceMatrix.array[16], 6);
assert.equal(meshState.mesh.instanceMatrix.array[32], 3);
assert.deepEqual(meshState.mesh.ranges, [[16, 16], [32, 16]]);

console.log('Dense instance slots and visibility bitset tests passed.');
