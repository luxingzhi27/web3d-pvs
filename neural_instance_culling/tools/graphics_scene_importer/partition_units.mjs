import { MeshoptEncoder } from 'meshoptimizer/encoder';

export const TARGET_UNIT_BYTES = 128 * 1024;

function positiveInteger(value, label) {
  const number = Number(value);
  if (!Number.isInteger(number) || number <= 0) throw new Error(`${label} must be a positive integer`);
  return number;
}

function compareText(left, right) {
  const a = String(left);
  const b = String(right);
  return a < b ? -1 : a > b ? 1 : 0;
}

function bytesOf(values) {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function boundsForPositions(values) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < values.length; index += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], values[index + axis]);
      max[axis] = Math.max(max[axis], values[index + axis]);
    }
  }
  return {
    min,
    max,
    center: min.map((value, axis) => (value + max[axis]) * 0.5),
    size: min.map((value, axis) => Math.max(0, max[axis] - value)),
  };
}

function includeBounds(target, source) {
  for (let axis = 0; axis < 3; axis += 1) {
    target.min[axis] = Math.min(target.min[axis], source.min[axis]);
    target.max[axis] = Math.max(target.max[axis], source.max[axis]);
  }
}

function unionFind(count) {
  const parent = Uint32Array.from({ length: count }, (_value, index) => index);
  const rank = new Uint8Array(count);
  function find(value) {
    let root = value;
    while (parent[root] !== root) root = parent[root];
    while (parent[value] !== value) {
      const next = parent[value];
      parent[value] = root;
      value = next;
    }
    return root;
  }
  function join(left, right) {
    let leftRoot = find(left);
    let rightRoot = find(right);
    if (leftRoot === rightRoot) return;
    if (rank[leftRoot] < rank[rightRoot]) [leftRoot, rightRoot] = [rightRoot, leftRoot];
    parent[rightRoot] = leftRoot;
    if (rank[leftRoot] === rank[rightRoot]) rank[leftRoot] += 1;
  }
  return { find, join };
}

function connectedTriangleGroups(renderable) {
  const triangleCount = renderable.indices.length / 3;
  const groups = new Map();
  const { find, join } = unionFind(triangleCount);
  const firstTriangleByVertex = new Map();
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = renderable.indices[triangle * 3 + corner];
      const previous = firstTriangleByVertex.get(vertex);
      if (previous == null) firstTriangleByVertex.set(vertex, triangle);
      else join(previous, triangle);
    }
  }
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const root = find(triangle);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(triangle);
  }
  return [...groups.values()];
}

function triangleCentroid(renderable, triangle) {
  const first = triangle * 3;
  const points = [0, 1, 2].map((corner) => {
    const vertex = renderable.indices[first + corner] * 3;
    return [
      renderable.positions[vertex],
      renderable.positions[vertex + 1],
      renderable.positions[vertex + 2],
    ];
  });
  return [
    (points[0][0] + points[1][0] + points[2][0]) / 3,
    (points[0][1] + points[1][1] + points[2][1]) / 3,
    (points[0][2] + points[1][2] + points[2][2]) / 3,
  ];
}

function morton3(x, y, z) {
  let result = 0;
  for (let bit = 0; bit < 10; bit += 1) {
    result |= ((x >> bit) & 1) << (bit * 3);
    result |= ((y >> bit) & 1) << (bit * 3 + 1);
    result |= ((z >> bit) & 1) << (bit * 3 + 2);
  }
  return result >>> 0;
}

function mortonForPoint(point, bounds) {
  const quantize = (value, axis) => {
    const extent = bounds.max[axis] - bounds.min[axis];
    const normalized = extent > 0 ? (value - bounds.min[axis]) / extent : 0;
    return Math.max(0, Math.min(1023, Math.floor(normalized * 1023)));
  };
  return morton3(quantize(point[0], 0), quantize(point[1], 1), quantize(point[2], 2));
}

function componentDescriptor(renderable, triangles, bounds) {
  const componentBounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  const centroid = [0, 0, 0];
  for (const triangle of triangles) {
    const point = triangleCentroid(renderable, triangle);
    for (let axis = 0; axis < 3; axis += 1) centroid[axis] += point[axis];
    for (const corner of [0, 1, 2]) {
      const vertex = renderable.indices[triangle * 3 + corner] * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        componentBounds.min[axis] = Math.min(componentBounds.min[axis], renderable.positions[vertex + axis]);
        componentBounds.max[axis] = Math.max(componentBounds.max[axis], renderable.positions[vertex + axis]);
      }
    }
  }
  for (let axis = 0; axis < 3; axis += 1) centroid[axis] /= Math.max(1, triangles.length);
  return {
    triangles,
    bounds: {
      ...componentBounds,
      center: componentBounds.min.map((value, axis) => (value + componentBounds.max[axis]) * 0.5),
      size: componentBounds.min.map((value, axis) => Math.max(0, componentBounds.max[axis] - value)),
    },
    centroid,
    morton: mortonForPoint(centroid, bounds),
    firstTriangle: triangles.reduce((minimum, triangle) => Math.min(minimum, triangle), Infinity),
  };
}

function sortedTriangles(renderable, triangles, bounds) {
  return triangles.slice().sort((left, right) => {
    const leftCode = mortonForPoint(triangleCentroid(renderable, left), bounds);
    const rightCode = mortonForPoint(triangleCentroid(renderable, right), bounds);
    return leftCode - rightCode || left - right;
  });
}

function geometryForTriangles(renderable, triangles) {
  const vertexMap = new Map();
  const originalVertices = [];
  const outputIndices = [];
  for (const triangle of triangles) {
    for (let corner = 0; corner < 3; corner += 1) {
      const original = Number(renderable.indices[triangle * 3 + corner]);
      let mapped = vertexMap.get(original);
      if (mapped == null) {
        mapped = originalVertices.length;
        vertexMap.set(original, mapped);
        originalVertices.push(original);
      }
      outputIndices.push(mapped);
    }
  }
  const positions = new Float32Array(originalVertices.length * 3);
  for (let index = 0; index < originalVertices.length; index += 1) {
    positions.set(renderable.positions.slice(originalVertices[index] * 3, originalVertices[index] * 3 + 3), index * 3);
  }
  const attributes = {};
  for (const [name, attribute] of Object.entries(renderable.attributes || {})) {
    const values = new Float32Array(originalVertices.length * attribute.components);
    for (let index = 0; index < originalVertices.length; index += 1) {
      const start = originalVertices[index] * attribute.components;
      values.set(attribute.values.slice(start, start + attribute.components), index * attribute.components);
    }
    attributes[name] = { values, components: attribute.components, type: attribute.type };
  }
  return {
    positions,
    indices: Uint32Array.from(outputIndices),
    attributes,
    vertexCount: positions.length / 3,
    triangleCount: triangles.length,
    bounds: boundsForPositions(positions),
  };
}

async function ensureEncoder() {
  if (!MeshoptEncoder.supported) throw new Error('meshopt encoder is unavailable');
  await MeshoptEncoder.ready;
}

function indexStream(geometry) {
  if (geometry.vertexCount <= 65535) return new Uint16Array(geometry.indices);
  return geometry.indices;
}

export async function estimateEncodedGeometryBytes(geometry) {
  await ensureEncoder();
  const attributes = {};
  let encodedBytes = 0;
  let rawBytes = geometry.indices.byteLength + geometry.positions.byteLength;
  const positionEncoded = MeshoptEncoder.encodeGltfBuffer(
    bytesOf(geometry.positions),
    geometry.vertexCount,
    12,
    'ATTRIBUTES',
  );
  encodedBytes += positionEncoded.length;
  for (const [name, attribute] of Object.entries(geometry.attributes || {})) {
    const stride = attribute.components * 4;
    const encoded = MeshoptEncoder.encodeGltfBuffer(
      bytesOf(attribute.values),
      geometry.vertexCount,
      stride,
      'ATTRIBUTES',
    );
    attributes[name] = { encodedBytes: encoded.length, rawBytes: attribute.values.byteLength, stride };
    encodedBytes += encoded.length;
    rawBytes += attribute.values.byteLength;
  }
  const indices = indexStream(geometry);
  const encodedIndices = MeshoptEncoder.encodeGltfBuffer(
    bytesOf(indices),
    indices.length,
    indices.BYTES_PER_ELEMENT,
    'TRIANGLES',
  );
  encodedBytes += encodedIndices.length;
  return {
    encodedGeometryBytes: encodedBytes,
    rawGeometryBytes: rawBytes,
    encodedPositionBytes: positionEncoded.length,
    encodedIndexBytes: encodedIndices.length,
    encodedAttributes: attributes,
    indexComponentType: indices.BYTES_PER_ELEMENT === 2 ? 5123 : 5125,
  };
}

async function makePart(renderable, triangles, componentOrdinals, partitionReason) {
  const geometry = geometryForTriangles(renderable, triangles);
  const compression = await estimateEncodedGeometryBytes(geometry);
  return {
    sourceNodeIndex: renderable.sourceNodeIndex,
    sourceNodePath: renderable.sourceNodePath,
    sourceMeshIndex: renderable.sourceMeshIndex,
    sourcePrimitiveIndex: renderable.sourcePrimitiveIndex,
    sourceMaterialIndex: renderable.sourceMaterialIndex,
    material: renderable.material,
    materialTexture: renderable.materialTexture || null,
    materialOutput: renderable.materialOutput,
    nodeName: renderable.nodeName,
    worldMatrix: renderable.worldMatrix,
    sourceTriangleIndices: triangles.slice(),
    sourceComponentOrdinals: componentOrdinals.slice(),
    partitionReason,
    geometry,
    bounds: geometry.bounds,
    compression,
    vertexCount: geometry.vertexCount,
    triangleCount: geometry.triangleCount,
  };
}

async function splitComponent(renderable, descriptor, targetBytes, componentOrdinal) {
  const triangles = sortedTriangles(renderable, descriptor.triangles, {
    min: renderable.bounds.min,
    max: renderable.bounds.max,
  });
  const parts = [];
  let offset = 0;
  while (offset < triangles.length) {
    let low = 1;
    let high = triangles.length - offset;
    let best = 1;
    while (low <= high) {
      const middle = Math.floor((low + high) * 0.5);
      const candidate = triangles.slice(offset, offset + middle);
      const estimate = await estimateEncodedGeometryBytes(geometryForTriangles(renderable, candidate));
      if (estimate.encodedGeometryBytes <= targetBytes) {
        best = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    const selected = triangles.slice(offset, offset + best);
    parts.push(await makePart(renderable, selected, [componentOrdinal], 'connected-component-morton'));
    offset += best;
  }
  return parts;
}

export async function partitionRenderable(renderable, options = {}) {
  const targetBytes = positiveInteger(options.targetBytes ?? TARGET_UNIT_BYTES, 'targetBytes');
  if (!renderable || renderable.indices.length % 3 !== 0) throw new Error('renderable must contain triangle indices');
  const primitiveBounds = renderable.bounds || boundsForPositions(renderable.positions);
  const componentDescriptors = connectedTriangleGroups(renderable)
    .map((triangles, ordinal) => componentDescriptor(renderable, triangles, primitiveBounds))
    .sort((left, right) => left.morton - right.morton || left.firstTriangle - right.firstTriangle);
  const parts = [];
  for (let ordinal = 0; ordinal < componentDescriptors.length; ordinal += 1) {
    const descriptor = componentDescriptors[ordinal];
    const full = await makePart(renderable, sortedTriangles(renderable, descriptor.triangles, primitiveBounds), [ordinal], 'natural-primitive');
    if (full.compression.encodedGeometryBytes <= targetBytes || descriptor.triangles.length === 1) {
      parts.push(full);
    } else {
      parts.push(...await splitComponent(renderable, descriptor, targetBytes, ordinal));
    }
  }

  const packed = [];
  let currentTriangles = [];
  let currentComponents = [];
  let currentEstimate = 0;
  for (const part of parts) {
    const candidateTriangles = currentTriangles.concat(part.sourceTriangleIndices);
    const candidateComponents = currentComponents.concat(part.sourceComponentOrdinals);
    const candidateEstimate = currentEstimate + part.compression.encodedGeometryBytes;
    if (currentTriangles.length > 0 && candidateEstimate > targetBytes) {
      packed.push(await makePart(
        renderable,
        currentTriangles,
        [...new Set(currentComponents)],
        currentComponents.length > 1 ? 'adjacent-small-components-morton' : parts.length > 1 ? 'connected-component-morton' : 'natural-primitive',
      ));
      currentTriangles = part.sourceTriangleIndices.slice();
      currentComponents = part.sourceComponentOrdinals.slice();
      currentEstimate = part.compression.encodedGeometryBytes;
    } else {
      currentTriangles = candidateTriangles;
      currentComponents = candidateComponents;
      currentEstimate = candidateEstimate;
    }
  }
  if (currentTriangles.length > 0) {
    packed.push(await makePart(
      renderable,
      currentTriangles,
      [...new Set(currentComponents)],
      currentComponents.length > 1 ? 'adjacent-small-components-morton' : parts.length > 1 ? 'connected-component-morton' : 'natural-primitive',
    ));
  }
  return packed;
}

export async function partitionScene(scene, options = {}) {
  const targetBytes = positiveInteger(options.targetBytes ?? TARGET_UNIT_BYTES, 'targetBytes');
  const allRenderables = [...(scene.renderables || [])];
  const blendExcluded = allRenderables
    .filter((item) => item.material?.alphaMode === 'BLEND')
    .map((item) => ({
      ...item,
      reason: 'blend-always-resident',
      alwaysResident: true,
      staticPvsEligible: false,
    }));
  const renderables = allRenderables.filter((item) => item.material?.alphaMode !== 'BLEND').sort((left, right) => (
    compareText(left.sourceNodePath, right.sourceNodePath)
    || Number(left.sourcePrimitiveIndex) - Number(right.sourcePrimitiveIndex)
    || Number(left.sourceMeshIndex) - Number(right.sourceMeshIndex)
  ));
  const units = [];
  for (let renderableIndex = 0; renderableIndex < renderables.length; renderableIndex += 1) {
    const parts = await partitionRenderable(renderables[renderableIndex], { targetBytes });
    for (const part of parts) {
      units.push({
        ...part,
        unitId: units.length,
        sourceRenderableIndex: renderableIndex,
        targetUnitBytes: targetBytes,
      });
    }
  }
  return {
    targetUnitBytes: targetBytes,
    units,
    source: scene.source,
    excluded: [...(scene.excluded || []), ...blendExcluded],
    imageResources: scene.imageResources || [],
    sceneBounds: scene.sceneBounds || null,
  };
}
