class UnionFind {
  constructor(size) {
    this.parent = new Int32Array(size);
    this.parent.fill(-1);
  }

  find(value) {
    let root = value;
    while (this.parent[root] >= 0) root = this.parent[root];
    let current = value;
    while (current !== root) {
      const next = this.parent[current];
      this.parent[current] = root;
      current = next;
    }
    return root;
  }

  union(a, b) {
    let rootA = this.find(a);
    let rootB = this.find(b);
    if (rootA === rootB) return;
    if (this.parent[rootA] > this.parent[rootB]) {
      const tmp = rootA;
      rootA = rootB;
      rootB = tmp;
    }
    this.parent[rootA] += this.parent[rootB];
    this.parent[rootB] = rootA;
  }
}

function prefixOffsets(counts) {
  const offsets = new Uint32Array(counts.length + 1);
  for (let i = 0; i < counts.length; i += 1) offsets[i + 1] = offsets[i] + counts[i];
  return offsets;
}

export function extractConnectedRegions(mesh, options = {}) {
  const progressEvery = Number(options.progressEvery || 5_000_000);
  const { vertexCount, triangleCount, indices } = mesh;
  const unionFind = new UnionFind(vertexCount);
  const used = new Uint8Array(vertexCount);

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3;
    const a = indices[offset];
    const b = indices[offset + 1];
    const c = indices[offset + 2];
    if (a >= vertexCount || b >= vertexCount || c >= vertexCount) throw new Error(`triangle ${triangle} references an invalid vertex`);
    used[a] = 1;
    used[b] = 1;
    used[c] = 1;
    unionFind.union(a, b);
    unionFind.union(b, c);
    if (progressEvery > 0 && (triangle + 1) % progressEvery === 0) {
      console.log(`[glb-instancer] connectivity triangles=${triangle + 1}/${triangleCount}`);
    }
  }

  const rootToRegion = new Int32Array(vertexCount);
  rootToRegion.fill(-1);
  const vertexRegion = new Int32Array(vertexCount);
  vertexRegion.fill(-1);
  const regionVertexCounts = [];
  let usedVertexCount = 0;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    if (!used[vertex]) continue;
    const root = unionFind.find(vertex);
    let region = rootToRegion[root];
    if (region < 0) {
      region = regionVertexCounts.length;
      rootToRegion[root] = region;
      regionVertexCounts.push(0);
    }
    vertexRegion[vertex] = region;
    regionVertexCounts[region] += 1;
    usedVertexCount += 1;
  }

  const regionTriangleCounts = new Array(regionVertexCounts.length).fill(0);
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3;
    const region = vertexRegion[indices[offset]];
    if (region < 0 || vertexRegion[indices[offset + 1]] !== region || vertexRegion[indices[offset + 2]] !== region) {
      throw new Error(`triangle ${triangle} crosses connected-region assignments`);
    }
    regionTriangleCounts[region] += 1;
  }

  const vertexOffsets = prefixOffsets(regionVertexCounts);
  const triangleOffsets = prefixOffsets(regionTriangleCounts);
  const verticesByRegion = new Uint32Array(usedVertexCount);
  const trianglesByRegion = new Uint32Array(triangleCount);
  const vertexCursors = vertexOffsets.slice(0, -1);
  const triangleCursors = triangleOffsets.slice(0, -1);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const region = vertexRegion[vertex];
    if (region >= 0) verticesByRegion[vertexCursors[region]++] = vertex;
  }
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const region = vertexRegion[indices[triangle * 3]];
    trianglesByRegion[triangleCursors[region]++] = triangle;
  }

  const regions = regionVertexCounts.map((vertexCountForRegion, id) => ({
    id,
    vertexOffset: vertexOffsets[id],
    vertexCount: vertexCountForRegion,
    triangleOffset: triangleOffsets[id],
    triangleCount: regionTriangleCounts[id],
  }));
  return {
    regions,
    vertexRegion,
    verticesByRegion,
    trianglesByRegion,
    usedVertexCount,
    unusedVertexCount: vertexCount - usedVertexCount,
  };
}

