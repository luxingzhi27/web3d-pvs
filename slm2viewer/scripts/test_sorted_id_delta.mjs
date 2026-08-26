#!/usr/bin/env node
import assert from 'node:assert/strict';
import {
  applySortedIdDelta,
  bitsetFromIds,
  countBitsetIds,
  diffIdBitsets,
  diffSortedIds,
} from '../src/sortedIdDelta.js';

const previous = [1, 2, 31, 32, 99];
const next = [2, 3, 32, 63, 99];
const expectedAdded = [3, 63];
const expectedRemoved = [1, 31];

const sortedDelta = diffSortedIds(previous, next);
assert.deepEqual(Array.from(sortedDelta.added), expectedAdded);
assert.deepEqual(Array.from(sortedDelta.removed), expectedRemoved);
assert.deepEqual(applySortedIdDelta(previous, sortedDelta.added, sortedDelta.removed), next);

const previousBits = bitsetFromIds(previous, 100);
const nextBits = bitsetFromIds(next, 100);
const bitsetDelta = diffIdBitsets(previousBits, nextBits, 100);
assert.deepEqual(Array.from(bitsetDelta.added), expectedAdded);
assert.deepEqual(Array.from(bitsetDelta.removed), expectedRemoved);
assert.equal(countBitsetIds(previousBits, 100), previous.length);
assert.equal(countBitsetIds(nextBits, 100), next.length);
assert.equal(countBitsetIds(new Uint32Array([0, 0xffffffff]), 35), 3);

console.log('Sorted ID and bitset delta tests passed.');
