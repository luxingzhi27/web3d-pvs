import crypto from 'node:crypto';

function updateInt32Chunked(hash, producer, count) {
  const chunkSize = 4096;
  const buffer = Buffer.allocUnsafe(chunkSize * 4);
  let offset = 0;
  while (offset < count) {
    const take = Math.min(chunkSize, count - offset);
    for (let i = 0; i < take; i += 1) buffer.writeInt32LE(producer(offset + i), i * 4);
    hash.update(buffer.subarray(0, take * 4));
    offset += take;
  }
}

function regionVertexId(connectivity, region, localVertex) {
  return connectivity.verticesByRegion[region.vertexOffset + localVertex];
}

function buildLocalIndexMap(mesh, connectivity, region, localIndexByVertex) {
  for (let local = 0; local < region.vertexCount; local += 1) {
    localIndexByVertex[regionVertexId(connectivity, region, local)] = local;
  }
}

function clearLocalIndexMap(connectivity, region, localIndexByVertex) {
  for (let local = 0; local < region.vertexCount; local += 1) {
    localIndexByVertex[regionVertexId(connectivity, region, local)] = -1;
  }
}

function computeCenterBounds(mesh, connectivity, region) {
  const center = [0, 0, 0];
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let local = 0; local < region.vertexCount; local += 1) {
    const vertex = regionVertexId(connectivity, region, local);
    const offset = vertex * 3;
    for (let axis = 0; axis < 3; axis += 1) {
      const value = mesh.positions[offset + axis];
      center[axis] += value;
      min[axis] = Math.min(min[axis], value);
      max[axis] = Math.max(max[axis], value);
    }
  }
  for (let axis = 0; axis < 3; axis += 1) center[axis] /= region.vertexCount;
  return { center, min, max };
}

export function describeRegions(mesh, connectivity, options = {}) {
  const tolerance = Number(options.fingerprintTolerance || 0.001);
  const progressEvery = Number(options.progressEvery || 10000);
  const localIndexByVertex = new Int32Array(mesh.vertexCount);
  localIndexByVertex.fill(-1);
  const descriptors = [];

  for (let regionIndex = 0; regionIndex < connectivity.regions.length; regionIndex += 1) {
    const region = connectivity.regions[regionIndex];
    const { center, min, max } = computeCenterBounds(mesh, connectivity, region);
    const hash = crypto.createHash('sha256');
    hash.update(`${region.vertexCount}:${region.triangleCount}:`);

    updateInt32Chunked(hash, (local) => {
      const vertex = regionVertexId(connectivity, region, local);
      const offset = vertex * 3;
      const dx = mesh.positions[offset] - center[0];
      const dy = mesh.positions[offset + 1] - center[1];
      const dz = mesh.positions[offset + 2] - center[2];
      return Math.round(Math.hypot(dx, dy, dz) / tolerance);
    }, region.vertexCount);

    const colorChunk = Buffer.allocUnsafe(Math.min(4096, region.vertexCount) * 4);
    let colorOffset = 0;
    while (colorOffset < region.vertexCount) {
      const take = Math.min(4096, region.vertexCount - colorOffset);
      for (let i = 0; i < take; i += 1) {
        const vertex = regionVertexId(connectivity, region, colorOffset + i);
        colorChunk[i * 4] = mesh.colors[vertex * 4];
        colorChunk[i * 4 + 1] = mesh.colors[vertex * 4 + 1];
        colorChunk[i * 4 + 2] = mesh.colors[vertex * 4 + 2];
        colorChunk[i * 4 + 3] = mesh.colors[vertex * 4 + 3];
      }
      hash.update(colorChunk.subarray(0, take * 4));
      colorOffset += take;
    }

    buildLocalIndexMap(mesh, connectivity, region, localIndexByVertex);
    updateInt32Chunked(hash, (localIndexOffset) => {
      const triangleLocal = Math.floor(localIndexOffset / 3);
      const corner = localIndexOffset % 3;
      const triangle = connectivity.trianglesByRegion[region.triangleOffset + triangleLocal];
      return localIndexByVertex[mesh.indices[triangle * 3 + corner]];
    }, region.triangleCount * 3);
    clearLocalIndexMap(connectivity, region, localIndexByVertex);

    descriptors.push({
      ...region,
      center,
      bounds: { min, max },
      fingerprint: `${region.vertexCount}:${region.triangleCount}:${hash.digest('hex')}`,
    });
    if (progressEvery > 0 && (regionIndex + 1) % progressEvery === 0) {
      console.log(`[glb-instancer] described=${regionIndex + 1}/${connectivity.regions.length}`);
    }
  }
  return descriptors;
}

function largestEigenvectorSymmetric4(matrix) {
  const a = matrix.slice();
  const vectors = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 64; sweep += 1) {
    let p = 0;
    let q = 1;
    let maxValue = 0;
    for (let row = 0; row < 4; row += 1) {
      for (let column = row + 1; column < 4; column += 1) {
        const value = Math.abs(a[row * 4 + column]);
        if (value > maxValue) {
          maxValue = value;
          p = row;
          q = column;
        }
      }
    }
    if (maxValue < 1e-12) break;
    const app = a[p * 4 + p];
    const aqq = a[q * 4 + q];
    const apq = a[p * 4 + q];
    const angle = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(angle);
    const s = Math.sin(angle);

    for (let k = 0; k < 4; k += 1) {
      if (k === p || k === q) continue;
      const akp = a[k * 4 + p];
      const akq = a[k * 4 + q];
      const newKp = c * akp - s * akq;
      const newKq = s * akp + c * akq;
      a[k * 4 + p] = newKp;
      a[p * 4 + k] = newKp;
      a[k * 4 + q] = newKq;
      a[q * 4 + k] = newKq;
    }
    a[p * 4 + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[q * 4 + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[p * 4 + q] = 0;
    a[q * 4 + p] = 0;
    for (let k = 0; k < 4; k += 1) {
      const vkp = vectors[k * 4 + p];
      const vkq = vectors[k * 4 + q];
      vectors[k * 4 + p] = c * vkp - s * vkq;
      vectors[k * 4 + q] = s * vkp + c * vkq;
    }
  }
  let largest = 0;
  for (let i = 1; i < 4; i += 1) if (a[i * 4 + i] > a[largest * 4 + largest]) largest = i;
  const out = [vectors[largest], vectors[4 + largest], vectors[8 + largest], vectors[12 + largest]];
  const length = Math.hypot(...out) || 1;
  return out.map((value) => value / length);
}

function quaternionToMatrix(quaternion) {
  const [w, x, y, z] = quaternion;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

function applyRotation(matrix, x, y, z) {
  return [
    matrix[0] * x + matrix[1] * y + matrix[2] * z,
    matrix[3] * x + matrix[4] * y + matrix[5] * z,
    matrix[6] * x + matrix[7] * y + matrix[8] * z,
  ];
}

export function alignRegions(mesh, connectivity, prototype, instance, options = {}) {
  if (prototype.vertexCount !== instance.vertexCount || prototype.triangleCount !== instance.triangleCount) return null;
  const s = new Array(9).fill(0);
  for (let local = 0; local < prototype.vertexCount; local += 1) {
    const prototypeVertex = regionVertexId(connectivity, prototype, local);
    const instanceVertex = regionVertexId(connectivity, instance, local);
    const po = prototypeVertex * 3;
    const io = instanceVertex * 3;
    const p = [
      mesh.positions[po] - prototype.center[0],
      mesh.positions[po + 1] - prototype.center[1],
      mesh.positions[po + 2] - prototype.center[2],
    ];
    const q = [
      mesh.positions[io] - instance.center[0],
      mesh.positions[io + 1] - instance.center[1],
      mesh.positions[io + 2] - instance.center[2],
    ];
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) s[row * 3 + column] += p[row] * q[column];
    }
  }
  const [sxx, sxy, sxz, syx, syy, syz, szx, szy, szz] = s;
  const n = [
    sxx + syy + szz, syz - szy, szx - sxz, sxy - syx,
    syz - szy, sxx - syy - szz, sxy + syx, szx + sxz,
    szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy,
    sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz,
  ];
  let quaternionWxyz = largestEigenvectorSymmetric4(n);
  if (quaternionWxyz[0] < 0) quaternionWxyz = quaternionWxyz.map((value) => -value);
  const rotationMatrix = quaternionToMatrix(quaternionWxyz);
  const rotatedCenter = applyRotation(rotationMatrix, ...prototype.center);
  const translation = [
    instance.center[0] - rotatedCenter[0],
    instance.center[1] - rotatedCenter[1],
    instance.center[2] - rotatedCenter[2],
  ];

  let squaredError = 0;
  let maxError = 0;
  for (let local = 0; local < prototype.vertexCount; local += 1) {
    const prototypeVertex = regionVertexId(connectivity, prototype, local);
    const instanceVertex = regionVertexId(connectivity, instance, local);
    const po = prototypeVertex * 3;
    const io = instanceVertex * 3;
    const rotated = applyRotation(
      rotationMatrix,
      mesh.positions[po],
      mesh.positions[po + 1],
      mesh.positions[po + 2],
    );
    const error = Math.hypot(
      rotated[0] + translation[0] - mesh.positions[io],
      rotated[1] + translation[1] - mesh.positions[io + 1],
      rotated[2] + translation[2] - mesh.positions[io + 2],
    );
    squaredError += error * error;
    maxError = Math.max(maxError, error);
  }
  const rmsError = Math.sqrt(squaredError / prototype.vertexCount);
  const rmsTolerance = Number(options.rmsTolerance || 0.001);
  const maxTolerance = Number(options.maxTolerance || 0.005);
  if (rmsError > rmsTolerance || maxError > maxTolerance) return null;
  return {
    translation,
    rotation: [quaternionWxyz[1], quaternionWxyz[2], quaternionWxyz[3], quaternionWxyz[0]],
    rmsError,
    maxError,
  };
}

export function groupSimilarRegions(mesh, connectivity, descriptors, options = {}) {
  const buckets = new Map();
  const groups = [];
  for (let index = 0; index < descriptors.length; index += 1) {
    const descriptor = descriptors[index];
    let bucketGroups = buckets.get(descriptor.fingerprint);
    if (!bucketGroups) {
      bucketGroups = [];
      buckets.set(descriptor.fingerprint, bucketGroups);
    }
    let match = null;
    for (const group of bucketGroups) {
      const alignment = alignRegions(mesh, connectivity, group.prototype, descriptor, options);
      if (alignment) {
        match = { group, alignment };
        break;
      }
    }
    if (match) {
      match.group.members.push({ descriptor, alignment: match.alignment });
    } else {
      const group = {
        prototype: descriptor,
        members: [{
          descriptor,
          alignment: { translation: [0, 0, 0], rotation: [0, 0, 0, 1], rmsError: 0, maxError: 0 },
        }],
      };
      bucketGroups.push(group);
      groups.push(group);
    }
    const progressEvery = Number(options.progressEvery || 10000);
    if (progressEvery > 0 && (index + 1) % progressEvery === 0) {
      console.log(`[glb-instancer] matched=${index + 1}/${descriptors.length} prototypes=${groups.length}`);
    }
  }
  return groups;
}

export function extractRegionGeometry(mesh, connectivity, region) {
  const positions = new Float32Array(region.vertexCount * 3);
  const colors = new Uint8Array(region.vertexCount * 4);
  const indices = new Uint32Array(region.triangleCount * 3);
  const localIndexByVertex = new Int32Array(mesh.vertexCount);
  localIndexByVertex.fill(-1);
  for (let local = 0; local < region.vertexCount; local += 1) {
    const vertex = regionVertexId(connectivity, region, local);
    localIndexByVertex[vertex] = local;
    positions.set(mesh.positions.subarray(vertex * 3, vertex * 3 + 3), local * 3);
    colors.set(mesh.colors.subarray(vertex * 4, vertex * 4 + 4), local * 4);
  }
  for (let localTriangle = 0; localTriangle < region.triangleCount; localTriangle += 1) {
    const triangle = connectivity.trianglesByRegion[region.triangleOffset + localTriangle];
    for (let corner = 0; corner < 3; corner += 1) {
      indices[localTriangle * 3 + corner] = localIndexByVertex[mesh.indices[triangle * 3 + corner]];
    }
  }
  return { positions, colors, indices };
}

