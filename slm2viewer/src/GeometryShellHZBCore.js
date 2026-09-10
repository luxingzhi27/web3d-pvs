// CPU-reference HZB operations shared by tests and offline diagnostics.
// The browser runtime uses the same projection and comparison contract in WGSL.

export const GEOMETRY_SHELL_HZB_SCHEMA = 'geometry-shell-hzb-v2';
export const LINEAR_DEPTH_ENCODING = 'positive_linear_view_depth_meters';

const EPSILON = 1e-8;
const CORNER_SIGNS = [
  [0, 0, 0],
  [0, 0, 1],
  [0, 1, 0],
  [0, 1, 1],
  [1, 0, 0],
  [1, 0, 1],
  [1, 1, 0],
  [1, 1, 1],
];

function finiteNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${name} must be finite.`);
  return number;
}

function vector3(value, name) {
  if (!value || value.length !== 3) throw new Error(`${name} must contain three values.`);
  return [
    finiteNumber(value[0], `${name}[0]`),
    finiteNumber(value[1], `${name}[1]`),
    finiteNumber(value[2], `${name}[2]`),
  ];
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(value) {
  return Math.hypot(value[0], value[1], value[2]);
}

function normalize(value, fallback, name) {
  const norm = length(value);
  if (!Number.isFinite(norm) || norm < EPSILON) {
    if (fallback) return fallback.slice();
    throw new Error(`${name} must have non-zero length.`);
  }
  return value.map((component) => component / norm);
}

function cameraValue(camera, names, fallback) {
  for (const name of names) {
    if (camera && camera[name] != null) return camera[name];
  }
  return fallback;
}

export function cameraBasis(camera) {
  const forward = normalize(
    vector3(camera?.forward, 'camera.forward'),
    [0, 0, -1],
    'camera.forward',
  );
  const suppliedUp = camera?.up ? vector3(camera.up, 'camera.up') : [0, 1, 0];
  const upReference = Math.abs(dot(forward, suppliedUp)) > 0.98
    ? [0, 0, 1]
    : suppliedUp;
  const right = normalize(cross(forward, upReference), [1, 0, 0], 'camera right');
  const up = normalize(cross(right, forward), [0, 1, 0], 'camera up');
  return { forward, right, up };
}

export function cameraProjectionParameters(camera) {
  const aspect = finiteNumber(camera?.aspect ?? 1, 'camera.aspect');
  if (!(aspect > 0)) throw new Error('camera.aspect must be positive.');
  const fovYDeg = finiteNumber(camera?.fovYDeg ?? camera?.fov ?? 60, 'camera.fovYDeg');
  if (!(fovYDeg > 0 && fovYDeg < 180)) throw new Error('camera.fovYDeg must be in (0, 180).');
  const tanY = Number.isFinite(Number(camera?.tanY))
    ? Number(camera.tanY)
    : Math.tan((fovYDeg * Math.PI) / 360);
  const tanX = Number.isFinite(Number(camera?.tanX))
    ? Number(camera.tanX)
    : tanY * aspect;
  if (!(tanX > 0) || !(tanY > 0)) throw new Error('camera tangents must be positive.');
  const near = finiteNumber(camera?.near ?? 0.05, 'camera.near');
  const far = finiteNumber(camera?.far ?? 20000, 'camera.far');
  if (!(near >= 0) || !(far > near)) throw new Error('camera near/far range is invalid.');
  return { aspect, fovYDeg, tanX, tanY, near, far };
}

function dimensions(width, height) {
  const resolvedWidth = Number(width);
  const resolvedHeight = Number(height);
  if (!Number.isInteger(resolvedWidth) || resolvedWidth < 1
      || !Number.isInteger(resolvedHeight) || resolvedHeight < 1) {
    throw new Error('HZB dimensions must be positive integers.');
  }
  return { width: resolvedWidth, height: resolvedHeight };
}

export function buildMaxMipChain(levelZero, width, height) {
  const size = dimensions(width, height);
  if (!levelZero || levelZero.length !== size.width * size.height) {
    throw new Error('HZB level zero length does not match its dimensions.');
  }
  const first = new Float32Array(levelZero.length);
  for (let index = 0; index < levelZero.length; index += 1) {
    const value = Number(levelZero[index]);
    if (!Number.isFinite(value) || value < 0) {
      throw new Error('HZB level zero must contain finite non-negative depths.');
    }
    first[index] = value;
  }
  const levels = [{ width: size.width, height: size.height, data: first }];
  while (levels.at(-1).width > 1 || levels.at(-1).height > 1) {
    const previous = levels.at(-1);
    const nextWidth = Math.max(1, Math.ceil(previous.width / 2));
    const nextHeight = Math.max(1, Math.ceil(previous.height / 2));
    const next = new Float32Array(nextWidth * nextHeight);
    for (let y = 0; y < nextHeight; y += 1) {
      for (let x = 0; x < nextWidth; x += 1) {
        let maximum = 0;
        for (let offsetY = 0; offsetY < 2; offsetY += 1) {
          const sourceY = y * 2 + offsetY;
          if (sourceY >= previous.height) continue;
          for (let offsetX = 0; offsetX < 2; offsetX += 1) {
            const sourceX = x * 2 + offsetX;
            if (sourceX >= previous.width) continue;
            maximum = Math.max(maximum, previous.data[sourceY * previous.width + sourceX]);
          }
        }
        next[y * nextWidth + x] = maximum;
      }
    }
    levels.push({ width: nextWidth, height: nextHeight, data: next });
  }
  return levels;
}

function aabbCorners(aabb) {
  if (!aabb || aabb.length !== 6) throw new Error('AABB must contain six values.');
  const minimum = [
    finiteNumber(aabb[0], 'aabb.min.x'),
    finiteNumber(aabb[1], 'aabb.min.y'),
    finiteNumber(aabb[2], 'aabb.min.z'),
  ];
  const maximum = [
    finiteNumber(aabb[3], 'aabb.max.x'),
    finiteNumber(aabb[4], 'aabb.max.y'),
    finiteNumber(aabb[5], 'aabb.max.z'),
  ];
  for (let axis = 0; axis < 3; axis += 1) {
    if (minimum[axis] > maximum[axis]) throw new Error('AABB minimum exceeds maximum.');
  }
  return CORNER_SIGNS.map(([x, y, z]) => [
    x ? maximum[0] : minimum[0],
    y ? maximum[1] : minimum[1],
    z ? maximum[2] : minimum[2],
  ]);
}

export function projectAabbConservatively(aabb, camera) {
  const position = vector3(camera?.position, 'camera.position');
  const basis = cameraBasis(camera);
  const projection = cameraProjectionParameters(camera);
  const corners = aabbCorners(aabb);
  const cameraInside = position[0] >= Number(aabb[0]) && position[0] <= Number(aabb[3])
    && position[1] >= Number(aabb[1]) && position[1] <= Number(aabb[4])
    && position[2] >= Number(aabb[2]) && position[2] <= Number(aabb[5]);
  const viewCorners = corners.map((corner) => {
    const relative = subtract(corner, position);
    return [dot(relative, basis.right), dot(relative, basis.up), dot(relative, basis.forward)];
  });
  const candidateNear = Math.min(...viewCorners.map((corner) => corner[2]));
  const candidateFar = Math.max(...viewCorners.map((corner) => corner[2]));
  const nearUncertain = viewCorners.some((corner) => corner[2] <= projection.near);
  const finite = viewCorners.every((corner) => corner.every(Number.isFinite));
  if (!finite || cameraInside || nearUncertain || candidateFar <= projection.near) {
    return {
      valid: false,
      uncertain: true,
      reason: !finite ? 'non-finite-projection'
        : cameraInside ? 'camera-inside-aabb'
          : candidateFar <= projection.near ? 'behind-near-plane' : 'near-plane-crossing',
      candidateNear,
      candidateFar,
      rect: [-1, -1, 1, 1],
    };
  }
  const projected = viewCorners.map(([x, y, z]) => [
    x / (z * projection.tanX),
    y / (z * projection.tanY),
  ]);
  const rawRect = [
    Math.min(...projected.map((corner) => corner[0])),
    Math.min(...projected.map((corner) => corner[1])),
    Math.max(...projected.map((corner) => corner[0])),
    Math.max(...projected.map((corner) => corner[1])),
  ];
  const outsideScreen = rawRect[2] < -1 || rawRect[0] > 1 || rawRect[3] < -1 || rawRect[1] > 1;
  if (outsideScreen || rawRect[2] <= rawRect[0] || rawRect[3] <= rawRect[1]) {
    return {
      valid: false,
      uncertain: true,
      reason: outsideScreen ? 'outside-screen' : 'empty-projection',
      candidateNear,
      candidateFar,
      rect: [
        Math.max(-1, Math.min(1, rawRect[0])),
        Math.max(-1, Math.min(1, rawRect[1])),
        Math.max(-1, Math.min(1, rawRect[2])),
        Math.max(-1, Math.min(1, rawRect[3])),
      ],
    };
  }
  return {
    valid: true,
    uncertain: false,
    reason: null,
    candidateNear,
    candidateFar,
    rect: [
      Math.max(-1, Math.min(1, rawRect[0])),
      Math.max(-1, Math.min(1, rawRect[1])),
      Math.max(-1, Math.min(1, rawRect[2])),
      Math.max(-1, Math.min(1, rawRect[3])),
    ],
  };
}

function queryFootprint(level, rect, baseWidth, baseHeight) {
  const x0 = Math.floor(((rect[0] + 1) * 0.5) * level.width);
  const y0 = Math.floor(((rect[1] + 1) * 0.5) * level.height);
  const x1 = Math.ceil(((rect[2] + 1) * 0.5) * level.width) - 1;
  const y1 = Math.ceil(((rect[3] + 1) * 0.5) * level.height) - 1;
  return {
    x0: Math.max(0, Math.min(level.width - 1, x0)),
    y0: Math.max(0, Math.min(level.height - 1, y0)),
    x1: Math.max(0, Math.min(level.width - 1, x1)),
    y1: Math.max(0, Math.min(level.height - 1, y1)),
    baseWidth,
    baseHeight,
  };
}

export function selectMaxMip(levels, rect) {
  if (!Array.isArray(levels) || levels.length === 0) throw new Error('HZB mip levels are required.');
  const base = levels[0];
  const pixelWidth = Math.max(1, (rect[2] - rect[0]) * 0.5 * base.width);
  const pixelHeight = Math.max(1, (rect[3] - rect[1]) * 0.5 * base.height);
  return Math.max(0, Math.min(levels.length - 1, Math.floor(Math.log2(Math.max(pixelWidth, pixelHeight)))));
}

export function queryAabbAgainstMaxMip(levels, rect, candidateNear, depthBiasM = 0) {
  if (!Number.isFinite(candidateNear) || candidateNear < 0) {
    return { visible: true, occluded: false, uncertain: true, reason: 'non-finite-candidate-depth' };
  }
  if (!Number.isFinite(Number(depthBiasM)) || Number(depthBiasM) < 0) {
    throw new Error('depthBiasM must be a finite non-negative value.');
  }
  const mip = selectMaxMip(levels, rect);
  const level = levels[mip];
  const footprint = queryFootprint(level, rect, levels[0].width, levels[0].height);
  if (footprint.x1 < footprint.x0 || footprint.y1 < footprint.y0) {
    return { visible: true, occluded: false, uncertain: true, reason: 'empty-hzb-footprint', mip };
  }
  let hzbMax = 0;
  for (let y = footprint.y0; y <= footprint.y1; y += 1) {
    for (let x = footprint.x0; x <= footprint.x1; x += 1) {
      const depth = Number(level.data[y * level.width + x]);
      if (!Number.isFinite(depth) || depth < 0) {
        return { visible: true, occluded: false, uncertain: true, reason: 'non-finite-hzb-depth', mip };
      }
      hzbMax = Math.max(hzbMax, depth);
    }
  }
  const occluded = hzbMax + Number(depthBiasM) < candidateNear;
  return {
    visible: !occluded,
    occluded,
    uncertain: false,
    reason: null,
    mip,
    hzbMax,
    candidateNear,
  };
}

function normalizeCandidateIds(candidateIds, count) {
  if (candidateIds == null) return Uint32Array.from({ length: count }, (_, index) => index);
  if (candidateIds.length !== count) throw new Error('candidateIds length does not match AABB count.');
  return Uint32Array.from(candidateIds, (value) => {
    if (!Number.isInteger(Number(value)) || Number(value) < 0) throw new Error('candidate IDs must be non-negative integers.');
    return Number(value);
  });
}

export function queryAabbsAgainstMaxMip(levels, camera, aabbs, candidateIds = null, options = {}) {
  if (!aabbs || aabbs.length % 6 !== 0) throw new Error('AABB buffer length must be divisible by six.');
  const count = aabbs.length / 6;
  const ids = normalizeCandidateIds(candidateIds, count);
  const depthBiasM = Number(options.depthBiasM ?? 0);
  const visibleIds = [];
  const uncertainIds = [];
  const records = [];
  const shellMask = options.shellMask || null;
  const hasShellMask = shellMask !== null;
  const shellVisibleIds = options.shellVisibleIds == null
    ? null
    : new Set(Array.from(options.shellVisibleIds, Number));
  if (hasShellMask && shellVisibleIds === null) {
    throw new Error('shellVisibleIds is required when a shell mask is provided.');
  }
  for (let index = 0; index < count; index += 1) {
    const projection = projectAabbConservatively(aabbs.subarray
      ? aabbs.subarray(index * 6, index * 6 + 6)
      : aabbs.slice(index * 6, index * 6 + 6), camera);
    let result;
    const maskValue = hasShellMask ? Number(shellMask[ids[index]]) : 0;
    if (hasShellMask && maskValue === 1) {
      result = {
        visible: shellVisibleIds.has(Number(ids[index])),
        occluded: !shellVisibleIds.has(Number(ids[index])),
        uncertain: false,
        reason: shellVisibleIds.has(Number(ids[index]))
          ? 'visible-shell-surface' : 'occluded-shell-instance',
      };
    } else if (hasShellMask && maskValue !== 0) {
      result = {
        visible: true,
        occluded: false,
        uncertain: true,
        reason: 'missing-occluder-mask',
      };
    } else if (!projection.valid || projection.uncertain) {
      result = {
        visible: true,
        occluded: false,
        uncertain: true,
        reason: projection.reason,
      };
    } else {
      result = queryAabbAgainstMaxMip(levels, projection.rect, projection.candidateNear, depthBiasM);
    }
    if (result.visible) visibleIds.push(ids[index]);
    if (result.uncertain) uncertainIds.push(ids[index]);
    records.push({ id: ids[index], ...projection, ...result });
  }
  visibleIds.sort((a, b) => a - b);
  uncertainIds.sort((a, b) => a - b);
  return {
    visibleIds: Uint32Array.from(visibleIds),
    uncertainIds: Uint32Array.from(uncertainIds),
    records,
    candidateCount: count,
    visibleCount: visibleIds.length,
    uncertainCount: uncertainIds.length,
    occludedCount: count - visibleIds.length,
  };
}

export function queryPoint60(levels, camera, aabbs, candidateIds = null, options = {}) {
  return {
    mode: 'Point60',
    ...queryAabbsAgainstMaxMip(levels, camera, aabbs, candidateIds, options),
  };
}

export function queryRegion66(hzbLevels, cameras, aabbs, candidateIds = null, options = {}) {
  if (!Array.isArray(cameras) || cameras.length === 0) {
    throw new Error('Region66 requires at least one camera.');
  }
  const perPose = [];
  const union = new Set();
  const uncertain = new Set();
  for (let index = 0; index < cameras.length; index += 1) {
    const levels = Array.isArray(hzbLevels) ? hzbLevels[index] : hzbLevels;
    if (!levels) throw new Error(`Region66 is missing HZB levels for pose ${index}.`);
    const result = queryAabbsAgainstMaxMip(levels, cameras[index], aabbs, candidateIds, options);
    perPose.push(result);
    for (const id of result.visibleIds) union.add(Number(id));
    for (const id of result.uncertainIds) uncertain.add(Number(id));
  }
  const visibleIds = Uint32Array.from([...union].sort((a, b) => a - b));
  return {
    mode: 'Region66',
    visibleIds,
    uncertainIds: Uint32Array.from([...uncertain].sort((a, b) => a - b)),
    perPose,
    passCount: cameras.length,
    candidateCount: candidateIds == null ? aabbs.length / 6 : candidateIds.length,
    visibleCount: visibleIds.length,
  };
}
