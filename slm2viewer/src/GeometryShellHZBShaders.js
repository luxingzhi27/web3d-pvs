// WebGPU shaders for the geometry-shell HZB MVP.

export const GEOMETRY_SHELL_DEPTH_VERTEX_SHADER = /* wgsl */ `
struct RenderUniforms {
  view_proj: mat4x4<f32>,
  view: mat4x4<f32>,
  depth_range: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: RenderUniforms;
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) view_depth: f32,
};

@vertex
fn depth_vertex_main(
  @location(0) position: vec3<f32>,
  @location(1) matrix_column_0: vec4<f32>,
  @location(2) matrix_column_1: vec4<f32>,
  @location(3) matrix_column_2: vec4<f32>,
  @location(4) matrix_column_3: vec4<f32>,
) -> VertexOutput {
  let instance_matrix = mat4x4<f32>(
    matrix_column_0,
    matrix_column_1,
    matrix_column_2,
    matrix_column_3,
  );
  let world_position = instance_matrix * vec4<f32>(position, 1.0);
  let view_position = uniforms.view * world_position;
  var output: VertexOutput;
  output.position = uniforms.view_proj * world_position;
  output.view_depth = max(-view_position.z, uniforms.depth_range.y);
  return output;
}
`;

export const GEOMETRY_SHELL_DEPTH_FRAGMENT_SHADER = /* wgsl */ `
@fragment
fn depth_fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.view_depth, 0.0, 0.0, 1.0);
}
`;

export const GEOMETRY_SHELL_MIP_SHADER = /* wgsl */ `
struct MipUniforms {
  source_width: u32,
  source_height: u32,
  target_width: u32,
  target_height: u32,
};

@group(0) @binding(0) var source_depth: texture_2d<f32>;
@group(0) @binding(1) var target_depth: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> uniforms: MipUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= uniforms.target_width || id.y >= uniforms.target_height) { return; }
  let source_origin = id.xy * 2u;
  var maximum = 0.0;
  for (var offset_y = 0u; offset_y < 2u; offset_y += 1u) {
    for (var offset_x = 0u; offset_x < 2u; offset_x += 1u) {
      let source_x = min(source_origin.x + offset_x, uniforms.source_width - 1u);
      let source_y = min(source_origin.y + offset_y, uniforms.source_height - 1u);
      maximum = max(maximum, textureLoad(source_depth, vec2<i32>(i32(source_x), i32(source_y)), 0).r);
    }
  }
  textureStore(target_depth, vec2<i32>(id.xy), vec4<f32>(maximum, 0.0, 0.0, 1.0));
}
`;

export const GEOMETRY_SHELL_QUERY_SHADER = /* wgsl */ `
struct QueryUniforms {
  view: mat4x4<f32>,
  viewport: vec4<f32>,
  projection: vec4<f32>,
  camera_position: vec4<f32>,
  counts: vec4<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: QueryUniforms;
@group(0) @binding(1) var<storage, read> instance_aabbs: array<f32>;
@group(0) @binding(2) var<storage, read> candidate_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> result_words: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> visible_ids: array<u32>;
@group(0) @binding(5) var hzb: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> instance_to_glb: array<u32>;
@group(0) @binding(7) var<storage, read_write> glb_flags: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> instance_occluder: array<u32>;

fn append_visible(candidate_id: u32) {
  let output_index = atomicAdd(&result_words[0], 1u);
  if (output_index < uniforms.counts.x) {
    visible_ids[output_index] = candidate_id;
  } else {
    atomicAdd(&result_words[2], 1u);
  }
  if (candidate_id < uniforms.counts.y) {
    let global_glb_id = instance_to_glb[candidate_id];
    if (global_glb_id < uniforms.counts.z) {
      atomicOr(&glb_flags[global_glb_id], 1u);
    }
  }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let candidate_index = id.x;
  let candidate_count = uniforms.counts.x;
  if (candidate_index >= candidate_count) { return; }
  let candidate_id = candidate_ids[candidate_index];
  if (candidate_id >= uniforms.counts.y) {
    atomicAdd(&result_words[1], 1u);
    append_visible(candidate_id);
    return;
  }
  // Selected shell occluders already contributed depth. Keep them to avoid
  // self-occlusion; every other candidate must be tested against that shell.
  if (instance_occluder[candidate_id] == 1u) {
    append_visible(candidate_id);
    return;
  }

  let aabb_offset = candidate_id * 6u;
  let minimum = vec3<f32>(
    instance_aabbs[aabb_offset],
    instance_aabbs[aabb_offset + 1u],
    instance_aabbs[aabb_offset + 2u]
  );
  let maximum = vec3<f32>(
    instance_aabbs[aabb_offset + 3u],
    instance_aabbs[aabb_offset + 4u],
    instance_aabbs[aabb_offset + 5u]
  );
  let camera = uniforms.camera_position.xyz;
  let camera_inside = all(camera >= minimum) && all(camera <= maximum);
  let tan_x = max(uniforms.projection.x, 1e-6);
  let tan_y = max(uniforms.projection.y, 1e-6);
  let near_plane = max(uniforms.projection.w, 0.0);
  var minimum_depth = 3.402823e+38;
  var maximum_depth = -3.402823e+38;
  var minimum_x = 3.402823e+38;
  var minimum_y = 3.402823e+38;
  var maximum_x = -3.402823e+38;
  var maximum_y = -3.402823e+38;
  var uncertain = camera_inside;

  for (var corner = 0u; corner < 8u; corner += 1u) {
    let point = vec3<f32>(
      select(minimum.x, maximum.x, (corner & 1u) != 0u),
      select(minimum.y, maximum.y, (corner & 2u) != 0u),
      select(minimum.z, maximum.z, (corner & 4u) != 0u)
    );
    let view_point = uniforms.view * vec4<f32>(point, 1.0);
    let depth = -view_point.z;
    if (!(depth == depth)) { uncertain = true; }
    minimum_depth = min(minimum_depth, depth);
    maximum_depth = max(maximum_depth, depth);
    if (depth <= near_plane) { uncertain = true; }
    let safe_depth = max(depth, 1e-6);
    let projected_x = view_point.x / (safe_depth * tan_x);
    let projected_y = view_point.y / (safe_depth * tan_y);
    if (!(projected_x == projected_x) || !(projected_y == projected_y)) { uncertain = true; }
    minimum_x = min(minimum_x, projected_x);
    minimum_y = min(minimum_y, projected_y);
    maximum_x = max(maximum_x, projected_x);
    maximum_y = max(maximum_y, projected_y);
  }

  if (maximum_depth <= near_plane || maximum_x < -1.0 || minimum_x > 1.0
      || maximum_y < -1.0 || minimum_y > 1.0) {
    uncertain = true;
  }
  if (uncertain) {
    atomicAdd(&result_words[1], 1u);
    append_visible(candidate_id);
    return;
  }

  let base_width = u32(uniforms.viewport.x);
  let base_height = u32(uniforms.viewport.y);
  let span_x = max(1.0, (maximum_x - minimum_x) * 0.5 * uniforms.viewport.x);
  let span_y = max(1.0, (maximum_y - minimum_y) * 0.5 * uniforms.viewport.y);
  let requested_mip = u32(max(0.0, floor(log2(max(span_x, span_y)))));
  let mip_level = min(requested_mip, uniforms.counts.w - 1u);
  let mip_scale = 1u << mip_level;
  let mip_width = max(1u, (base_width + mip_scale - 1u) >> mip_level);
  let mip_height = max(1u, (base_height + mip_scale - 1u) >> mip_level);
  let raw_x0 = floor((clamp(minimum_x, -1.0, 1.0) + 1.0) * 0.5 * f32(mip_width));
  let raw_y0 = floor((clamp(minimum_y, -1.0, 1.0) + 1.0) * 0.5 * f32(mip_height));
  let raw_x1 = ceil((clamp(maximum_x, -1.0, 1.0) + 1.0) * 0.5 * f32(mip_width)) - 1.0;
  let raw_y1 = ceil((clamp(maximum_y, -1.0, 1.0) + 1.0) * 0.5 * f32(mip_height)) - 1.0;
  let x0 = u32(clamp(raw_x0, 0.0, f32(mip_width - 1u)));
  let y0 = u32(clamp(raw_y0, 0.0, f32(mip_height - 1u)));
  let x1 = u32(clamp(raw_x1, 0.0, f32(mip_width - 1u)));
  let y1 = u32(clamp(raw_y1, 0.0, f32(mip_height - 1u)));
  var hzb_max = 0.0;
  var sample_count = 0u;
  for (var y = y0; y <= y1; y += 1u) {
    for (var x = x0; x <= x1; x += 1u) {
      let depth = textureLoad(hzb, vec2<i32>(i32(x), i32(y)), mip_level).r;
      if (!(depth == depth)) { uncertain = true; }
      hzb_max = max(hzb_max, depth);
      sample_count += 1u;
    }
  }
  if (sample_count == 0u || uncertain || !(minimum_depth == minimum_depth)) {
    atomicAdd(&result_words[1], 1u);
    append_visible(candidate_id);
    return;
  }

  // Positive linear depth: only this strict proof permits culling.
  if (hzb_max + uniforms.projection.z < minimum_depth) { return; }
  append_visible(candidate_id);
}
`;
