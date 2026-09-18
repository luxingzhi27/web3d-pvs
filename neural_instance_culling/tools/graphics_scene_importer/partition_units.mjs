import { MeshoptEncoder } from 'meshoptimizer/encoder';

export const TARGET_UNIT_BYTES = 128 * 1024;
export const PARTITION_SCHEMA = 'connected-sah-pack-v2';
export const SAH_BIN_COUNT = 16;
// Large nodes are split before exact meshopt probing to cap temporary geometry.
export const MAX_GEOMETRY_PROBE_TRIANGLES = 32768;

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

function unionFindComponents(renderable) {
  const triangleCount = renderable.indices.length / 3;
  const vertexCount = renderable.positions.length / 3;
  const parent = new Int32Array(triangleCount);
  const firstTriangleByVertex = new Int32Array(vertexCount);
  parent.fill(-1);
  firstTriangleByVertex.fill(-1);

  function find(value) {
    let root = value;
    while (parent[root] >= 0) root = parent[root];
    while (value !== root) {
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
    if (parent[leftRoot] > parent[rightRoot]) [leftRoot, rightRoot] = [rightRoot, leftRoot];
    parent[leftRoot] += parent[rightRoot];
    parent[rightRoot] = leftRoot;
  }

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const firstIndex = triangle * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = Number(renderable.indices[firstIndex + corner]);
      if (!Number.isInteger(vertex) || vertex < 0 || vertex >= vertexCount) {
        throw new Error(`triangle ${triangle} references invalid vertex ${vertex}`);
      }
      const previous = firstTriangleByVertex[vertex];
      if (previous < 0) firstTriangleByVertex[vertex] = triangle;
      else join(previous, triangle);
    }
  }

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const root = find(triangle);
    if (root !== triangle) parent[triangle] = root;
  }
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    if (parent[triangle] < 0) parent[triangle] = -1;
  }

  let componentCount = 0;
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const root = parent[triangle] >= 0 ? parent[triangle] : triangle;
    if (parent[root] === -1) {
      parent[root] = -(componentCount + 2);
      componentCount += 1;
    }
  }
  // Convert non-roots first so root markers remain available to every link.
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    if (parent[triangle] >= 0) {
      const root = parent[triangle];
      parent[triangle] = -parent[root] - 2;
    }
  }
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    if (parent[triangle] < 0) parent[triangle] = -parent[triangle] - 2;
  }

  const componentCounts = new Uint32Array(componentCount);
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    componentCounts[parent[triangle]] += 1;
  }
  const componentOffsets = new Uint32Array(componentCount + 1);
  for (let component = 0; component < componentCount; component += 1) {
    componentOffsets[component + 1] = componentOffsets[component] + componentCounts[component];
  }
  const writeOffsets = componentOffsets.slice(0, componentCount);
  const componentTriangles = new Uint32Array(triangleCount);
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const component = parent[triangle];
    componentTriangles[writeOffsets[component]] = triangle;
    writeOffsets[component] += 1;
  }
  return { componentCount, componentOffsets, componentTriangles };
}

// Layout per item: centroid xyz, bounds min xyz, bounds max xyz.
function buildItemData(renderable, items, itemOffsets, itemTriangles) {
  const itemCount = items;
  const data = new Float32Array(itemCount * 9);
  const weights = new Uint32Array(itemCount);
  const firstSourceTriangles = new Uint32Array(itemCount);
  for (let item = 0; item < itemCount; item += 1) {
    const start = itemOffsets[item];
    const end = itemOffsets[item + 1];
    const dataIndex = item * 9;
    data[dataIndex + 3] = Infinity;
    data[dataIndex + 4] = Infinity;
    data[dataIndex + 5] = Infinity;
    data[dataIndex + 6] = -Infinity;
    data[dataIndex + 7] = -Infinity;
    data[dataIndex + 8] = -Infinity;
    weights[item] = end - start;
    firstSourceTriangles[item] = itemTriangles[start];
    for (let offset = start; offset < end; offset += 1) {
      const sourceTriangle = itemTriangles[offset];
      const sourceIndex = sourceTriangle * 3;
      for (let corner = 0; corner < 3; corner += 1) {
        const vertex = Number(renderable.indices[sourceIndex + corner]) * 3;
        for (let axis = 0; axis < 3; axis += 1) {
          const value = renderable.positions[vertex + axis];
          data[dataIndex + axis] += value / 3;
          data[dataIndex + 3 + axis] = Math.min(data[dataIndex + 3 + axis], value);
          data[dataIndex + 6 + axis] = Math.max(data[dataIndex + 6 + axis], value);
        }
      }
    }
    for (let axis = 0; axis < 3; axis += 1) data[dataIndex + axis] /= Math.max(1, weights[item]);
  }
  return { data, weights, firstSourceTriangles };
}

function centroidBoundsForItems(items, data) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    const dataIndex = item * 9;
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], data[dataIndex + axis]);
      max[axis] = Math.max(max[axis], data[dataIndex + axis]);
    }
  }
  return { min, max };
}

function itemBin(item, axis, centroidBounds, data) {
  const extent = centroidBounds.max[axis] - centroidBounds.min[axis];
  if (extent <= 0) return 0;
  const dataIndex = item * 9;
  const normalized = (data[dataIndex + axis] - centroidBounds.min[axis]) / extent;
  return Math.max(0, Math.min(SAH_BIN_COUNT - 1, Math.floor(normalized * SAH_BIN_COUNT)));
}

function initializeBinBounds(bounds) {
  for (let bin = 0; bin < SAH_BIN_COUNT; bin += 1) {
    const offset = bin * 6;
    bounds[offset] = Infinity;
    bounds[offset + 1] = Infinity;
    bounds[offset + 2] = Infinity;
    bounds[offset + 3] = -Infinity;
    bounds[offset + 4] = -Infinity;
    bounds[offset + 5] = -Infinity;
  }
}

function includeItemBounds(target, targetOffset, data, dataOffset) {
  for (let axis = 0; axis < 3; axis += 1) {
    target[targetOffset + axis] = Math.min(target[targetOffset + axis], data[dataOffset + 3 + axis]);
    target[targetOffset + 3 + axis] = Math.max(target[targetOffset + 3 + axis], data[dataOffset + 6 + axis]);
  }
}

function includeBoundsArray(target, targetOffset, source, sourceOffset) {
  for (let axis = 0; axis < 3; axis += 1) {
    target[targetOffset + axis] = Math.min(target[targetOffset + axis], source[sourceOffset + axis]);
    target[targetOffset + 3 + axis] = Math.max(target[targetOffset + 3 + axis], source[sourceOffset + 3 + axis]);
  }
}

function surfaceAreaArray(bounds, offset) {
  const x = Math.max(0, bounds[offset + 3] - bounds[offset]);
  const y = Math.max(0, bounds[offset + 4] - bounds[offset + 1]);
  const z = Math.max(0, bounds[offset + 5] - bounds[offset + 2]);
  return 2 * (x * y + x * z + y * z);
}

function compareSahCandidate(cost, axis, bin, sourceTriangle, best) {
  if (best.axis < 0) return true;
  const tolerance = 1e-12 * Math.max(1, Math.abs(cost), Math.abs(best.cost));
  if (cost < best.cost - tolerance) return true;
  if (cost > best.cost + tolerance) return false;
  return axis < best.axis
    || (axis === best.axis && bin < best.bin)
    || (axis === best.axis && bin === best.bin && sourceTriangle < best.sourceTriangle);
}

function partitionByBin(items, data, axis, bin, centroidBounds) {
  let leftCount = 0;
  for (let index = 0; index < items.length; index += 1) {
    if (itemBin(items[index], axis, centroidBounds, data) <= bin) leftCount += 1;
  }
  let left = 0;
  let right = items.length - 1;
  while (left < leftCount && right >= leftCount) {
    while (left < leftCount && itemBin(items[left], axis, centroidBounds, data) <= bin) left += 1;
    while (right >= leftCount && itemBin(items[right], axis, centroidBounds, data) > bin) right -= 1;
    if (left < leftCount && right >= leftCount) {
      const value = items[left];
      items[left] = items[right];
      items[right] = value;
      left += 1;
      right -= 1;
    }
  }
  if (leftCount === 0 || leftCount === items.length) return null;
  return { left: items.subarray(0, leftCount), right: items.subarray(leftCount), axis, bin };
}

function binnedSahSplit(items, stableKeys, data, weights = null) {
  const centroidBounds = centroidBoundsForItems(items, data);
  let best = { axis: -1, bin: -1, cost: Infinity, sourceTriangle: Infinity };
  for (let axis = 0; axis < 3; axis += 1) {
    if (centroidBounds.max[axis] <= centroidBounds.min[axis]) continue;
    const binCounts = new Uint32Array(SAH_BIN_COUNT);
    const binWeights = new Float64Array(SAH_BIN_COUNT);
    const binFirst = new Uint32Array(SAH_BIN_COUNT);
    binFirst.fill(0xffffffff);
    const binBounds = new Float64Array(SAH_BIN_COUNT * 6);
    initializeBinBounds(binBounds);
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index];
      const bin = itemBin(item, axis, centroidBounds, data);
      const dataIndex = item * 9;
      const binOffset = bin * 6;
      binCounts[bin] += 1;
      binWeights[bin] += weights ? weights[item] : 1;
      binFirst[bin] = Math.min(binFirst[bin], stableKeys[item]);
      includeItemBounds(binBounds, binOffset, data, dataIndex);
    }

    const prefixCounts = new Uint32Array(SAH_BIN_COUNT);
    const suffixCounts = new Uint32Array(SAH_BIN_COUNT);
    const prefixWeights = new Float64Array(SAH_BIN_COUNT);
    const suffixWeights = new Float64Array(SAH_BIN_COUNT);
    const prefixFirst = new Uint32Array(SAH_BIN_COUNT);
    const suffixFirst = new Uint32Array(SAH_BIN_COUNT);
    prefixFirst.fill(0xffffffff);
    suffixFirst.fill(0xffffffff);
    const prefixBounds = new Float64Array(SAH_BIN_COUNT * 6);
    const suffixBounds = new Float64Array(SAH_BIN_COUNT * 6);
    initializeBinBounds(prefixBounds);
    initializeBinBounds(suffixBounds);
    let count = 0;
    let weight = 0;
    let first = 0xffffffff;
    for (let bin = 0; bin < SAH_BIN_COUNT; bin += 1) {
      count += binCounts[bin];
      weight += binWeights[bin];
      first = Math.min(first, binFirst[bin]);
      prefixCounts[bin] = count;
      prefixWeights[bin] = weight;
      prefixFirst[bin] = first;
      const offset = bin * 6;
      if (bin > 0) {
        for (let axisIndex = 0; axisIndex < 6; axisIndex += 1) {
          prefixBounds[offset + axisIndex] = prefixBounds[offset - 6 + axisIndex];
        }
      }
      includeBoundsArray(prefixBounds, offset, binBounds, offset);
    }
    count = 0;
    weight = 0;
    first = 0xffffffff;
    for (let bin = SAH_BIN_COUNT - 1; bin >= 0; bin -= 1) {
      count += binCounts[bin];
      weight += binWeights[bin];
      first = Math.min(first, binFirst[bin]);
      suffixCounts[bin] = count;
      suffixWeights[bin] = weight;
      suffixFirst[bin] = first;
      const offset = bin * 6;
      if (bin < SAH_BIN_COUNT - 1) {
        for (let axisIndex = 0; axisIndex < 6; axisIndex += 1) {
          suffixBounds[offset + axisIndex] = suffixBounds[offset + 6 + axisIndex];
        }
      }
      includeBoundsArray(suffixBounds, offset, binBounds, offset);
    }
    for (let bin = 0; bin < SAH_BIN_COUNT - 1; bin += 1) {
      if (prefixCounts[bin] === 0 || suffixCounts[bin + 1] === 0) continue;
      const cost = surfaceAreaArray(prefixBounds, bin * 6) * prefixWeights[bin]
        + surfaceAreaArray(suffixBounds, (bin + 1) * 6) * suffixWeights[bin + 1];
      if (compareSahCandidate(cost, axis, bin, suffixFirst[bin + 1], best)) {
        best = { axis, bin, cost, sourceTriangle: suffixFirst[bin + 1] };
      }
    }
  }
  if (best.axis < 0) return null;
  return partitionByBin(items, data, best.axis, best.bin, centroidBounds);
}

function stableMedianSplit(items, stableKeys) {
  items.sort((left, right) => stableKeys[left] - stableKeys[right]);
  const middle = Math.floor(items.length * 0.5);
  return { left: items.subarray(0, middle), right: items.subarray(middle), axis: null, bin: null };
}

function splitItemsDeterministically(items, stableKeys, data, weights = null) {
  return binnedSahSplit(items, stableKeys, data, weights)
    || stableMedianSplit(items, stableKeys);
}

function buildTriangleCache(renderable, sourceTriangles) {
  const triangleCount = sourceTriangles.length;
  const data = new Float32Array(triangleCount * 9);
  const positions = renderable.positions;
  const indices = renderable.indices;
  for (let localTriangle = 0; localTriangle < triangleCount; localTriangle += 1) {
    const sourceTriangle = sourceTriangles[localTriangle];
    const sourceIndex = sourceTriangle * 3;
    const dataIndex = localTriangle * 9;
    data[dataIndex + 3] = Infinity;
    data[dataIndex + 4] = Infinity;
    data[dataIndex + 5] = Infinity;
    data[dataIndex + 6] = -Infinity;
    data[dataIndex + 7] = -Infinity;
    data[dataIndex + 8] = -Infinity;
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = Number(indices[sourceIndex + corner]) * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        const value = positions[vertex + axis];
        data[dataIndex + axis] += value / 3;
        data[dataIndex + 3 + axis] = Math.min(data[dataIndex + 3 + axis], value);
        data[dataIndex + 6 + axis] = Math.max(data[dataIndex + 6 + axis], value);
      }
    }
  }
  const order = new Uint32Array(triangleCount);
  for (let triangle = 0; triangle < triangleCount; triangle += 1) order[triangle] = triangle;
  return { data, order };
}

function createVertexRemapper(vertexCount) {
  return {
    marks: new Int32Array(vertexCount),
    localIndices: new Uint32Array(vertexCount),
    epoch: 0,
  };
}

function nextRemapEpoch(remapper) {
  remapper.epoch += 1;
  if (remapper.epoch >= 0x7fffffff) {
    remapper.marks.fill(0);
    remapper.epoch = 1;
  }
  return remapper.epoch;
}

function geometryForSourceTriangles(renderable, sourceTriangleIds, remapper) {
  const maxVertexCount = sourceTriangleIds.length * 3;
  const originalVertices = new Uint32Array(maxVertexCount);
  const outputIndices = new Uint32Array(maxVertexCount);
  const epoch = nextRemapEpoch(remapper);
  let vertexCount = 0;
  let outputIndex = 0;
  for (let index = 0; index < sourceTriangleIds.length; index += 1) {
    const sourceTriangle = sourceTriangleIds[index];
    const sourceIndex = sourceTriangle * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const originalVertex = Number(renderable.indices[sourceIndex + corner]);
      if (remapper.marks[originalVertex] !== epoch) {
        remapper.marks[originalVertex] = epoch;
        remapper.localIndices[originalVertex] = vertexCount;
        originalVertices[vertexCount] = originalVertex;
        vertexCount += 1;
      }
      outputIndices[outputIndex] = remapper.localIndices[originalVertex];
      outputIndex += 1;
    }
  }
  const positions = new Float32Array(vertexCount * 3);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const sourceVertex = originalVertices[vertex] * 3;
    positions.set(renderable.positions.subarray(sourceVertex, sourceVertex + 3), vertex * 3);
  }
  const attributes = {};
  for (const [name, attribute] of Object.entries(renderable.attributes || {})) {
    const values = new Float32Array(vertexCount * attribute.components);
    for (let vertex = 0; vertex < vertexCount; vertex += 1) {
      const sourceVertex = originalVertices[vertex] * attribute.components;
      values.set(
        attribute.values.subarray(sourceVertex, sourceVertex + attribute.components),
        vertex * attribute.components,
      );
    }
    attributes[name] = { values, components: attribute.components, type: attribute.type };
  }
  return {
    positions,
    indices: outputIndices,
    attributes,
    vertexCount,
    triangleCount: sourceTriangleIds.length,
    bounds: boundsForPositions(positions),
  };
}

function sourceTrianglesForLocalOrder(localTriangles, sourceTriangles) {
  const result = new Uint32Array(localTriangles.length);
  for (let index = 0; index < localTriangles.length; index += 1) {
    result[index] = sourceTriangles[localTriangles[index]];
  }
  return result;
}

function sourceTrianglesForComponents(componentOrder, componentOffsets, componentTriangles) {
  let triangleCount = 0;
  for (let index = 0; index < componentOrder.length; index += 1) {
    const component = componentOrder[index];
    triangleCount += componentOffsets[component + 1] - componentOffsets[component];
  }
  const result = new Uint32Array(triangleCount);
  let offset = 0;
  for (let index = 0; index < componentOrder.length; index += 1) {
    const component = componentOrder[index];
    const start = componentOffsets[component];
    const end = componentOffsets[component + 1];
    result.set(componentTriangles.subarray(start, end), offset);
    offset += end - start;
  }
  return result;
}

function componentOrdinals(componentOrder) {
  const result = new Array(componentOrder.length);
  for (let index = 0; index < componentOrder.length; index += 1) result[index] = componentOrder[index];
  return result;
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

function makePartRecord(
  renderable,
  sourceTriangleIds,
  sourceComponentOrdinals,
  sourceComponentCount,
  partitionReason,
  geometry,
  compression,
  oversizeReason = null,
  oversizedComponentSplit = false,
) {
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
    sourceTriangleIndices: sourceTriangleIds.slice(),
    sourceComponentOrdinals,
    partitionSchema: PARTITION_SCHEMA,
    componentCount: sourceComponentOrdinals.length,
    sourceComponentCount,
    partitionReason,
    oversizedComponentSplit,
    oversize: oversizeReason != null,
    oversizeReason,
    geometry,
    bounds: geometry.bounds,
    compression,
    vertexCount: geometry.vertexCount,
    triangleCount: geometry.triangleCount,
  };
}

async function evaluateLocalTriangles(renderable, localTriangles, sourceTriangles, remapper) {
  const sourceTriangleIds = sourceTrianglesForLocalOrder(localTriangles, sourceTriangles);
  const geometry = geometryForSourceTriangles(renderable, sourceTriangleIds, remapper);
  return {
    sourceTriangleIds,
    geometry,
    compression: await estimateEncodedGeometryBytes(geometry),
  };
}

async function splitOversizedComponent(
  renderable,
  sourceTriangles,
  cache,
  targetBytes,
  componentOrdinal,
  componentCount,
  remapper,
  emit,
) {
  let rootNode = true;
  async function visit(localTriangles) {
    let evaluation = null;
    if (localTriangles.length <= MAX_GEOMETRY_PROBE_TRIANGLES) {
      evaluation = await evaluateLocalTriangles(renderable, localTriangles, sourceTriangles, remapper);
      if (evaluation.compression.encodedGeometryBytes <= targetBytes) {
        await emit(makePartRecord(
          renderable,
          evaluation.sourceTriangleIds,
          [componentOrdinal],
          componentCount,
          rootNode ? 'connected-component' : 'oversized-connected-component-binned-sah',
          evaluation.geometry,
          evaluation.compression,
          null,
          !rootNode,
        ));
        return;
      }
      if (localTriangles.length === 1) {
        await emit(makePartRecord(
          renderable,
          evaluation.sourceTriangleIds,
          [componentOrdinal],
          componentCount,
          'single-triangle-oversize',
          evaluation.geometry,
          evaluation.compression,
          'single-triangle',
        ));
        return;
      }
    }
    rootNode = false;
    evaluation = null;
    const split = splitItemsDeterministically(localTriangles, sourceTriangles, cache.data);
    if (!split || !split.left.length || !split.right.length) {
      throw new Error('connected-SAH could not split a multi-triangle component');
    }
    await visit(split.left);
    await visit(split.right);
  }
  await visit(cache.order);
}

async function partitionComponentGroups(
  renderable,
  connected,
  componentData,
  componentOrder,
  targetBytes,
  maxComponentsPerUnit,
  remapper,
  emit,
) {
  const componentCount = connected.componentCount;
  async function visit(order) {
    if (order.length === 1) {
      const component = order[0];
      const start = connected.componentOffsets[component];
      const end = connected.componentOffsets[component + 1];
      const sourceTriangles = connected.componentTriangles.subarray(start, end);
      const cache = buildTriangleCache(renderable, sourceTriangles);
      await splitOversizedComponent(
        renderable,
        sourceTriangles,
        cache,
        targetBytes,
        component,
        componentCount,
        remapper,
        emit,
      );
      return;
    }

    let triangleCount = 0;
    for (let index = 0; index < order.length; index += 1) {
      const component = order[index];
      triangleCount += connected.componentOffsets[component + 1] - connected.componentOffsets[component];
    }
    if (order.length <= maxComponentsPerUnit && triangleCount <= MAX_GEOMETRY_PROBE_TRIANGLES) {
      const sourceTriangleIds = sourceTrianglesForComponents(
        order,
        connected.componentOffsets,
        connected.componentTriangles,
      );
      const geometry = geometryForSourceTriangles(renderable, sourceTriangleIds, remapper);
      const compression = await estimateEncodedGeometryBytes(geometry);
      if (compression.encodedGeometryBytes <= targetBytes) {
        await emit(makePartRecord(
          renderable,
          sourceTriangleIds,
          componentOrdinals(order),
          componentCount,
          'connected-component-binned-sah-pack',
          geometry,
          compression,
        ));
        return;
      }
    }
    const split = splitItemsDeterministically(
      order,
      componentData.firstSourceTriangles,
      componentData.data,
      componentData.weights,
    );
    if (!split || !split.left.length || !split.right.length) {
      throw new Error('connected-SAH could not split a multi-component group');
    }
    await visit(split.left);
    await visit(split.right);
  }
  await visit(componentOrder);
}

export async function partitionRenderable(renderable, options = {}) {
  const targetBytes = positiveInteger(options.targetBytes ?? TARGET_UNIT_BYTES, 'targetBytes');
  const maxComponentsPerUnit = options.maxComponentsPerUnit == null
    ? Infinity
    : positiveInteger(options.maxComponentsPerUnit, 'maxComponentsPerUnit');
  if (!renderable || renderable.indices.length % 3 !== 0) throw new Error('renderable must contain triangle indices');
  if (renderable.positions.length % 3 !== 0) throw new Error('renderable positions must contain xyz triples');
  const connected = unionFindComponents(renderable);
  const componentCount = connected.componentCount;
  if (componentCount === 0) return [];
  const componentData = buildItemData(
    renderable,
    componentCount,
    connected.componentOffsets,
    connected.componentTriangles,
  );
  const componentOrder = new Uint32Array(componentCount);
  for (let component = 0; component < componentCount; component += 1) componentOrder[component] = component;
  const remapper = createVertexRemapper(renderable.positions.length / 3);
  const onPart = options.onPart || null;
  const parts = onPart ? null : [];
  const emit = async (part) => {
    if (onPart) await onPart(part);
    else parts.push(part);
  };
  await partitionComponentGroups(
    renderable,
    connected,
    componentData,
    componentOrder,
    targetBytes,
    maxComponentsPerUnit,
    remapper,
    emit,
  );
  return parts || [];
}

export async function partitionScene(scene, options = {}) {
  const targetBytes = positiveInteger(options.targetBytes ?? TARGET_UNIT_BYTES, 'targetBytes');
  const maxComponentsPerUnit = options.maxComponentsPerUnit == null
    ? null
    : positiveInteger(options.maxComponentsPerUnit, 'maxComponentsPerUnit');
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
  const retainUnits = options.retainUnits !== false;
  const units = retainUnits ? [] : null;
  let componentCount = 0;
  let unitCount = 0;
  let oversizeUnitCount = 0;
  let oversizeTriangleCount = 0;
  let packedMultiComponentUnitCount = 0;
  let packedComponentCount = 0;
  let oversizedComponentSplitUnitCount = 0;
  for (let renderableIndex = 0; renderableIndex < renderables.length; renderableIndex += 1) {
    let renderableComponentCount = 0;
    const parts = await partitionRenderable(renderables[renderableIndex], {
      targetBytes,
      maxComponentsPerUnit,
      onPart: options.onUnit
        ? async (part) => {
          renderableComponentCount = part.sourceComponentCount;
          const unit = {
            ...part,
            unitId: unitCount,
            sourceRenderableIndex: renderableIndex,
            targetUnitBytes: targetBytes,
            maxComponentsPerUnit,
          };
          if (unit.oversize) {
            oversizeUnitCount += 1;
            oversizeTriangleCount += unit.triangleCount;
          }
          if (unit.componentCount > 1) {
            packedMultiComponentUnitCount += 1;
            packedComponentCount += unit.componentCount;
          }
          if (unit.oversizedComponentSplit) oversizedComponentSplitUnitCount += 1;
          await options.onUnit(unit);
          if (units) units.push(unit);
          unitCount += 1;
        }
        : null,
    });
    if (!options.onUnit) {
      renderableComponentCount = parts.length
        ? parts[0].sourceComponentCount
        : 0;
      for (const part of parts) {
        const unit = {
          ...part,
          unitId: unitCount,
          sourceRenderableIndex: renderableIndex,
          targetUnitBytes: targetBytes,
          maxComponentsPerUnit,
        };
        if (unit.oversize) {
          oversizeUnitCount += 1;
          oversizeTriangleCount += unit.triangleCount;
        }
        if (unit.componentCount > 1) {
          packedMultiComponentUnitCount += 1;
          packedComponentCount += unit.componentCount;
        }
        if (unit.oversizedComponentSplit) oversizedComponentSplitUnitCount += 1;
        if (units) units.push(unit);
        unitCount += 1;
      }
    }
    componentCount += renderableComponentCount;
  }
  return {
    targetUnitBytes: targetBytes,
    partitionSchema: PARTITION_SCHEMA,
    componentCount,
    unitCount,
    partition: {
      schema: PARTITION_SCHEMA,
      algorithm: 'shared-vertex connected components with deterministic binned SAH component packing and internal triangle splitting',
      binCount: SAH_BIN_COUNT,
      maxGeometryProbeTriangles: MAX_GEOMETRY_PROBE_TRIANGLES,
      targetUnitBytes: targetBytes,
      maxComponentsPerUnit,
      componentCount,
      packedMultiComponentUnitCount,
      packedComponentCount,
      oversizedComponentSplitUnitCount,
      oversizeUnitCount,
      oversizeTriangleCount,
    },
    units: units || [],
    source: scene.source,
    excluded: [...(scene.excluded || []), ...blendExcluded],
    imageResources: scene.imageResources || [],
    sceneBounds: scene.sceneBounds || null,
  };
}
