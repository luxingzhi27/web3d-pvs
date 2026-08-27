const TWO_PI = Math.PI * 2;
const FLOAT32_EPSILON = 1.1920929e-7;

export function createFloat16Lookup() {
  const output = new Float32Array(65536);
  for (let value = 0; value < output.length; value += 1) {
    const sign = (value & 0x8000) ? -1 : 1;
    const exponent = (value >>> 10) & 0x1f;
    const mantissa = value & 0x03ff;
    if (exponent === 0) output[value] = sign * mantissa * 2 ** -24;
    else if (exponent === 0x1f) output[value] = mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
    else output[value] = sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }
  return output;
}

export function unpackFloat16(words, elementCount, lookup = createFloat16Lookup()) {
  const halves = new Uint16Array(words.buffer, words.byteOffset, words.byteLength / 2);
  const output = new Float32Array(elementCount);
  for (let index = 0; index < elementCount; index += 1) output[index] = lookup[halves[index]];
  return output;
}

export function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

export function silu(value) {
  return value * sigmoid(value);
}

export function softplus(value) {
  return Math.log1p(Math.exp(-Math.abs(value))) + Math.max(value, 0);
}

function activate(value, activation) {
  if (activation === 'silu') return silu(value);
  if (activation === 'relu') return Math.max(value, 0);
  if (activation === 'tanh') return Math.tanh(value);
  return value;
}

export function linearInto(input, output, weights, weightOffset, biasOffset, activation = 'none') {
  const inputDim = input.length;
  for (let row = 0; row < output.length; row += 1) {
    let value = weights[biasOffset + row];
    const rowOffset = weightOffset + row * inputDim;
    for (let column = 0; column < inputDim; column += 1) {
      value += weights[rowOffset + column] * input[column];
    }
    output[row] = activate(value, activation);
  }
  return output;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalize3(x, y, z, fallback) {
  const length = Math.hypot(x, y, z);
  if (length <= 1e-6) return fallback.slice();
  return [x / length, y / length, z / length];
}

function fillRayDifferential(target, targetOffset, axis, values) {
  const {
    ray, distance, radius, forward, right, up, tanX, tanY,
    dotForward, dotRight, dotUp, denominatorXRaw, denominatorYRaw,
    denominatorX, denominatorY, screenURaw, screenVRaw, horizontalRaw, verticalRaw,
  } = values;
  const rayAxis = ray[axis];
  const dDistance = -rayAxis;
  const dRay = [
    -(Number(axis === 0) - ray[0] * rayAxis) / distance,
    -(Number(axis === 1) - ray[1] * rayAxis) / distance,
    -(Number(axis === 2) - ray[2] * rayAxis) / distance,
  ];
  const dLog = dDistance / (4 * (100 + distance));
  const dForward = dRay[0] * forward[0] + dRay[1] * forward[1] + dRay[2] * forward[2];
  const dRight = dRay[0] * right[0] + dRay[1] * right[1] + dRay[2] * right[2];
  const dUp = dRay[0] * up[0] + dRay[1] * up[1] + dRay[2] * up[2];
  const dDenominatorXRaw = Math.sign(dotForward) * dForward * tanX;
  const dDenominatorYRaw = Math.sign(dotForward) * dForward * tanY;
  const dDenominatorX = denominatorXRaw > 1e-4 ? dDenominatorXRaw : 0;
  const dDenominatorY = denominatorYRaw > 1e-4 ? dDenominatorYRaw : 0;
  const dURaw = (dRight * denominatorX - dotRight * dDenominatorX) / (denominatorX * denominatorX);
  const dVRaw = (dUp * denominatorY - dotUp * dDenominatorY) / (denominatorY * denominatorY);
  const dAngular = distance > 1 ? -radius * dDistance / (distance * distance) : 0;
  const differential = [
    dRay[0], dRay[1], dRay[2], dLog, dForward,
    Math.abs(screenURaw) < 4 ? dURaw / 4 : 0,
    Math.abs(screenVRaw) < 4 ? dVRaw / 4 : 0,
    horizontalRaw > 0 && horizontalRaw < 4 ? dAngular / tanX / 2 : 0,
    verticalRaw > 0 && verticalRaw < 4 ? dAngular / tanY / 2 : 0,
  ];
  for (let index = 0; index < differential.length; index += 1) {
    target[targetOffset + index * 2] = differential[index];
  }
}

export function makeRayQuery(aabbs, instanceId, cameraQuery, viewcellRadius, output) {
  const offset = instanceId * 6;
  const centerWorld = [
    0.5 * (aabbs[offset] + aabbs[offset + 3]),
    0.5 * (aabbs[offset + 1] + aabbs[offset + 4]),
    0.5 * (aabbs[offset + 2] + aabbs[offset + 5]),
  ];
  const size = [
    Math.max(aabbs[offset + 3] - aabbs[offset], 1e-4),
    Math.max(aabbs[offset + 4] - aabbs[offset + 1], 1e-4),
    Math.max(aabbs[offset + 5] - aabbs[offset + 2], 1e-4),
  ];
  const delta = centerWorld.map((value, index) => value - cameraQuery.center[index]);
  const distance = Math.max(Math.hypot(delta[0], delta[1], delta[2]), 1e-4);
  const ray = delta.map((value) => value / distance);
  const radius = 0.5 * Math.hypot(size[0], size[1], size[2]);
  const forward = normalize3(...cameraQuery.forward, [0, 0, -1]);
  const upSeed = Math.abs(forward[1]) > 0.98 ? [0, 0, 1] : [0, 1, 0];
  const right = normalize3(
    forward[1] * upSeed[2] - forward[2] * upSeed[1],
    forward[2] * upSeed[0] - forward[0] * upSeed[2],
    forward[0] * upSeed[1] - forward[1] * upSeed[0],
    [1, 0, 0],
  );
  const up = normalize3(
    right[1] * forward[2] - right[2] * forward[1],
    right[2] * forward[0] - right[0] * forward[2],
    right[0] * forward[1] - right[1] * forward[0],
    [0, 1, 0],
  );
  const tanX = Math.max(cameraQuery.tanX, 1e-4);
  const tanY = Math.max(cameraQuery.tanY, 1e-4);
  const dotForward = ray[0] * forward[0] + ray[1] * forward[1] + ray[2] * forward[2];
  const dotRight = ray[0] * right[0] + ray[1] * right[1] + ray[2] * right[2];
  const dotUp = ray[0] * up[0] + ray[1] * up[1] + ray[2] * up[2];
  const denominatorXRaw = Math.abs(dotForward) * tanX;
  const denominatorYRaw = Math.abs(dotForward) * tanY;
  const denominatorX = Math.max(denominatorXRaw, 1e-4);
  const denominatorY = Math.max(denominatorYRaw, 1e-4);
  const screenURaw = dotRight / denominatorX;
  const screenVRaw = dotUp / denominatorY;
  const angular = radius / Math.max(distance, 1);
  const horizontalRaw = angular / tanX;
  const verticalRaw = angular / tanY;
  const center = output.center;
  center[0] = ray[0];
  center[1] = ray[1];
  center[2] = ray[2];
  center[3] = clamp(Math.log1p(distance / 100) / 4, 0, 2) - 1;
  center[4] = clamp(dotForward, -1, 1);
  center[5] = clamp(screenURaw, -4, 4) / 4;
  center[6] = clamp(screenVRaw, -4, 4) / 4;
  center[7] = clamp(horizontalRaw, 0, 4) / 2 - 1;
  center[8] = clamp(verticalRaw, 0, 4) / 2 - 1;
  const values = {
    ray, distance, radius, forward, right, up, tanX, tanY,
    dotForward, dotRight, dotUp, denominatorXRaw, denominatorYRaw,
    denominatorX, denominatorY, screenURaw, screenVRaw, horizontalRaw, verticalRaw,
  };
  output.axes.fill(0);
  fillRayDifferential(output.axes, 0, 0, values);
  fillRayDifferential(output.axes, 1, 2, values);
  for (let feature = 0; feature < 9; feature += 1) {
    const xIndex = feature * 2;
    const rawX = output.axes[xIndex] * viewcellRadius;
    const rawZ = output.axes[xIndex + 1] * viewcellRadius;
    const norm = Math.hypot(rawX, rawZ);
    const budget = Math.max(1 - Math.abs(center[feature]), 0);
    const scale = norm > budget ? budget / Math.max(norm, 1e-12) : 1;
    output.axes[xIndex] = rawX * scale;
    output.axes[xIndex + 1] = rawZ * scale;
  }
  output.distance = distance;
  output.radius = radius;
  return output;
}

function lookupChi(chi, argument) {
  const position = clamp(argument, 0, 320) * (8191 / 320);
  const lower = Math.min(Math.floor(position), 8190);
  const fraction = position - lower;
  return chi[lower] + fraction * (chi[lower + 1] - chi[lower]);
}

export function spectralInto(query, frequencies, chi, output) {
  for (let frequency = 0; frequency < 16; frequency += 1) {
    let phaseCycles = 0;
    let projectedX = 0;
    let projectedZ = 0;
    for (let feature = 0; feature < 9; feature += 1) {
      const cycle = frequencies[frequency * 9 + feature];
      phaseCycles += query.center[feature] * cycle;
      projectedX += query.axes[feature * 2] * cycle;
      projectedZ += query.axes[feature * 2 + 1] * cycle;
    }
    const phase = TWO_PI * phaseCycles;
    const radial = TWO_PI * Math.hypot(projectedX, projectedZ);
    const chiS = lookupChi(chi, radial);
    const chi2S = lookupChi(chi, 2 * radial);
    const chiSquared = chiS * chiS;
    const sharedVariance = 0.5 * (1 - chiSquared);
    const directedVariance = 0.5 * Math.cos(2 * phase) * (chiSquared - chi2S);
    const varianceSin = Math.max(sharedVariance + directedVariance, 0);
    const varianceCos = Math.max(sharedVariance - directedVariance, 0);
    const base = frequency * 4;
    output[base] = Math.sin(phase) * chiS;
    output[base + 1] = Math.cos(phase) * chiS;
    output[base + 2] = radial === 0 || varianceSin <= FLOAT32_EPSILON ? 0 : Math.sqrt(varianceSin);
    output[base + 3] = radial === 0 || varianceCos <= FLOAT32_EPSILON ? 0 : Math.sqrt(varianceCos);
  }
  return output;
}

export function lowRankInto(query, output) {
  let xSquared = 0;
  let zSquared = 0;
  let maximum = 0;
  for (let feature = 0; feature < 9; feature += 1) {
    const x = query.axes[feature * 2];
    const z = query.axes[feature * 2 + 1];
    xSquared += x * x;
    zSquared += z * z;
    maximum = Math.max(maximum, Math.abs(x), Math.abs(z));
  }
  output[0] = Math.sqrt(xSquared);
  output[1] = Math.sqrt(zSquared);
  output[2] = Math.sqrt(xSquared + zSquared);
  output[3] = maximum;
  return output;
}

export function survivalInto(basis, coefficients, normalizedDepth, output) {
  const parameters = new Float32Array(7);
  for (let parameter = 0; parameter < 7; parameter += 1) {
    let value = 0;
    for (let rank = 0; rank < 4; rank += 1) value += basis[rank] * coefficients[rank * 7 + parameter];
    parameters[parameter] = value;
  }
  const noBlock = sigmoid(parameters[0]);
  const weightMaximum = Math.max(parameters[1], parameters[2]);
  const weightExp0 = Math.exp(parameters[1] - weightMaximum);
  const weightExp1 = Math.exp(parameters[2] - weightMaximum);
  let weight0 = weightExp0 / (weightExp0 + weightExp1);
  let weight1 = weightExp1 / (weightExp0 + weightExp1);
  let depth0 = 0.05 + 0.9 * sigmoid(parameters[3]);
  let depth1 = 0.05 + 0.9 * sigmoid(parameters[4]);
  let scale0 = 0.03 + softplus(parameters[5]);
  let scale1 = 0.03 + softplus(parameters[6]);
  if (depth0 > depth1) {
    [depth0, depth1] = [depth1, depth0];
    [weight0, weight1] = [weight1, weight0];
    [scale0, scale1] = [scale1, scale0];
  }
  const cdf0 = sigmoid((normalizedDepth - depth0) / Math.max(scale0, 1e-3));
  const cdf1 = sigmoid((normalizedDepth - depth1) / Math.max(scale1, 1e-3));
  const survival = clamp(1 - (1 - noBlock) * (weight0 * cdf0 + weight1 * cdf1), 1e-5, 1);
  const expectedDepth = weight0 * depth0 + weight1 * depth1;
  const variance = weight0 * (depth0 - expectedDepth) ** 2 + weight1 * (depth1 - expectedDepth) ** 2;
  output[0] = survival;
  output[1] = 1 - survival;
  output[2] = noBlock;
  output[3] = expectedDepth;
  output[4] = Math.sqrt(Math.max(variance, 1e-8));
  output[5] = weight0;
  output[6] = weight1;
  output[7] = (1 - noBlock) * (
    weight0 / Math.max(scale0, 1e-3) * cdf0 * (1 - cdf0)
    + weight1 / Math.max(scale1, 1e-3) * cdf1 * (1 - cdf1)
  );
  return output;
}

export function intersectsAabbFrustum(aabbs, instanceId, planes) {
  const offset = instanceId * 6;
  for (let planeIndex = 0; planeIndex < 6; planeIndex += 1) {
    const planeOffset = planeIndex * 4;
    const nx = planes[planeOffset];
    const ny = planes[planeOffset + 1];
    const nz = planes[planeOffset + 2];
    const x = nx >= 0 ? aabbs[offset + 3] : aabbs[offset];
    const y = ny >= 0 ? aabbs[offset + 4] : aabbs[offset + 1];
    const z = nz >= 0 ? aabbs[offset + 5] : aabbs[offset + 2];
    if (nx * x + ny * y + nz * z + planes[planeOffset + 3] < 0) return false;
  }
  return true;
}
