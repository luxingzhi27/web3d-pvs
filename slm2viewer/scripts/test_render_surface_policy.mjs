#!/usr/bin/env node
import assert from 'node:assert/strict';
import {
  resolveRenderSurface,
  sameRenderSurface,
} from '../src/RenderSurfacePolicy.js';

const normal = resolveRenderSurface({ width: 1000, height: 700, devicePixelRatio: 1 });
assert.deepEqual(normal, {
  cssWidth: 1000,
  cssHeight: 700,
  pixelRatio: 1,
  drawingBufferWidth: 1000,
  drawingBufferHeight: 700,
});

const highDpi = resolveRenderSurface({ width: 1600, height: 900, devicePixelRatio: 2 });
assert.equal(highDpi.pixelRatio, 2);
assert.equal(highDpi.drawingBufferWidth, 3200);
assert.equal(highDpi.drawingBufferHeight, 1800);
assert.ok(sameRenderSurface(highDpi, { ...highDpi }));
assert.ok(!sameRenderSurface(highDpi, { ...highDpi, cssWidth: 1599 }));

const fractionalDpi = resolveRenderSurface({
  width: 800,
  height: 600,
  devicePixelRatio: 1.25,
});
assert.equal(fractionalDpi.pixelRatio, 1.25);
assert.equal(fractionalDpi.drawingBufferWidth, 1000);
assert.equal(fractionalDpi.drawingBufferHeight, 750);

const bounded = resolveRenderSurface({ width: 3840, height: 2160, devicePixelRatio: 4 });
assert.equal(bounded.pixelRatio, 4);
assert.equal(bounded.drawingBufferWidth, 15360);
assert.equal(bounded.drawingBufferHeight, 8640);

console.log('Fixed native-DPR render surface tests passed.');
