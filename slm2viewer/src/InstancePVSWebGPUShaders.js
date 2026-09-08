import { DIAGNOSTIC_OUTPUT_FLOATS } from './InstancePVSBase.js';

const WORKGROUP_SIZE = 64;
const GLB_FLAG_MODEL_VISIBLE = 1;
const GLB_FLAG_RENDER_VISIBLE = 2;

function activationExpression(kind, value) {
  if (kind === 'silu') return `silu(${value})`;
  if (kind === 'relu') return `max(${value}, 0.0)`;
  if (kind === 'tanh') return `tanh(${value})`;
  return value;
}

function linearFunction(name, inputDim, outputDim, weightOffset, biasOffset, activation = 'none') {
  return `
fn ${name}(input: array<f32, ${inputDim}>) -> array<f32, ${outputDim}> {
  var output: array<f32, ${outputDim}>;
  for (var row = 0u; row < ${outputDim}u; row += 1u) {
    var value = weight_value(${biasOffset}u + row);
    let row_offset = ${weightOffset}u + row * ${inputDim}u;
    for (var col = 0u; col < ${inputDim}u; col += 1u) {
      value += weight_value(row_offset + col) * input[col];
    }
    output[row] = ${activationExpression(activation, 'value')};
  }
  return output;
}`;
}

export function buildPredictionShader() {
    const offset = (name) => this._weightOffset(name);
    const layers = [
      linearFunction('relation_hidden', 28, 16, offset('relation_condition_head.0.weight'), offset('relation_condition_head.0.bias'), 'silu'),
      linearFunction('relation_output', 16, 8, offset('relation_condition_head.2.weight'), offset('relation_condition_head.2.bias')),
      linearFunction('boundary_hidden', 85, 48, offset('boundary_summary_head.0.weight'), offset('boundary_summary_head.0.bias'), 'silu'),
      linearFunction('boundary_output', 48, 8, offset('boundary_summary_head.2.weight'), offset('boundary_summary_head.2.bias'), 'tanh'),
      linearFunction('direction_hidden', 19, 24, offset('direction_basis_head.0.weight'), offset('direction_basis_head.0.bias'), 'silu'),
      linearFunction('direction_output', 24, 4, offset('direction_basis_head.2.weight'), offset('direction_basis_head.2.bias'), 'tanh'),
      linearFunction('trunk_hidden0', 130, 64, offset('shared_trunk.0.weight'), offset('shared_trunk.0.bias'), 'relu'),
      linearFunction('trunk_hidden1', 64, 64, offset('shared_trunk.2.weight'), offset('shared_trunk.2.bias'), 'relu'),
      linearFunction('visibility_output', 64, 1, offset('visibility_head.weight'), offset('visibility_head.bias')),
    ].join('\n');
    const layout = this.resultLayout;
    const diagnosticWrites = this.debugLogging ? `
  atomicStore(&results[${layout.debugCandidateIds}u + candidate_index], instance_id);
  let diagnostic_offset = ${layout.debugRows}u + candidate_index * ${DIAGNOSTIC_OUTPUT_FLOATS}u;
  atomicStore(&results[diagnostic_offset], bitcast<u32>(probability));
  for (var value_index = 0u; value_index < 9u; value_index += 1u) {
    atomicStore(&results[diagnostic_offset + 1u + value_index], bitcast<u32>(query.center[value_index]));
  }
  for (var value_index = 0u; value_index < 18u; value_index += 1u) {
    atomicStore(&results[diagnostic_offset + 10u + value_index], bitcast<u32>(query.axes[value_index]));
  }
  for (var value_index = 0u; value_index < 64u; value_index += 1u) {
    atomicStore(&results[diagnostic_offset + 28u + value_index], bitcast<u32>(spectral[value_index]));
  }` : '';

    return `
struct Uniforms {
  query_center_threshold: vec4<f32>,
  forward_tan_x: vec4<f32>,
  query_parameters: vec4<f32>,
  depth_count: vec4<f32>,
  candidate_planes: array<vec4<f32>, 6>,
  render_planes: array<vec4<f32>, 6>,
};

struct RayQuery {
  center: array<f32, 9>,
  axes: array<f32, 18>,
  distance: f32,
  radius: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> runtime_words: array<u32>;
@group(0) @binding(2) var<storage, read> instance_aabbs: array<f32>;
@group(0) @binding(3) var<storage, read> weight_words: array<u32>;
@group(0) @binding(4) var<storage, read> frequency_cycles: array<f32>;
@group(0) @binding(5) var<storage, read> chi_table: array<f32>;
@group(0) @binding(6) var<storage, read> instance_to_glb: array<u32>;
@group(0) @binding(7) var<storage, read_write> results: array<atomic<u32>>;

fn runtime_value(index: u32) -> f32 {
  let pair = unpack2x16float(runtime_words[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn weight_value(index: u32) -> f32 {
  let pair = unpack2x16float(weight_words[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn sigmoid(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-value));
}

fn silu(value: f32) -> f32 {
  return value * sigmoid(value);
}

fn softplus(value: f32) -> f32 {
  return log(1.0 + exp(-abs(value))) + max(value, 0.0);
}

fn safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let length_value = length(value);
  return select(fallback, value / length_value, length_value > 1e-6);
}

fn intersects_frustum(instance_id: u32, render_frustum: bool) -> bool {
  let aabb_offset = instance_id * 6u;
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
  for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
    let plane = select(
      uniforms.candidate_planes[plane_index],
      uniforms.render_planes[plane_index],
      render_frustum
    );
    let positive = select(minimum, maximum, plane.xyz >= vec3<f32>(0.0));
    if (dot(plane.xyz, positive) + plane.w < 0.0) { return false; }
  }
  return true;
}

fn ray_differential(
  axis: vec3<f32>,
  ray: vec3<f32>,
  distance: f32,
  radius: f32,
  forward: vec3<f32>,
  right: vec3<f32>,
  up: vec3<f32>,
  tan_x: f32,
  tan_y: f32,
  dot_forward: f32,
  dot_right: f32,
  dot_up: f32,
  denominator_x_raw: f32,
  denominator_y_raw: f32,
  denominator_x: f32,
  denominator_y: f32,
  screen_u_raw: f32,
  screen_v_raw: f32,
  horizontal_raw: f32,
  vertical_raw: f32,
) -> array<f32, 9> {
  var result: array<f32, 9>;
  let ray_axis = dot(ray, axis);
  let d_distance = -ray_axis;
  let d_ray = -(axis - ray * ray_axis) / distance;
  let d_log = d_distance / (4.0 * (100.0 + distance));
  let d_forward = dot(d_ray, forward);
  let d_right = dot(d_ray, right);
  let d_up = dot(d_ray, up);
  let d_denominator_x_raw = sign(dot_forward) * d_forward * tan_x;
  let d_denominator_y_raw = sign(dot_forward) * d_forward * tan_y;
  let d_denominator_x = select(0.0, d_denominator_x_raw, denominator_x_raw > 1e-4);
  let d_denominator_y = select(0.0, d_denominator_y_raw, denominator_y_raw > 1e-4);
  let d_u_raw = (d_right * denominator_x - dot_right * d_denominator_x) / (denominator_x * denominator_x);
  let d_v_raw = (d_up * denominator_y - dot_up * d_denominator_y) / (denominator_y * denominator_y);
  let d_angular = select(0.0, -radius * d_distance / (distance * distance), distance > 1.0);
  let d_horizontal_raw = d_angular / tan_x;
  let d_vertical_raw = d_angular / tan_y;
  result[0] = d_ray.x;
  result[1] = d_ray.y;
  result[2] = d_ray.z;
  result[3] = d_log;
  result[4] = d_forward;
  result[5] = select(0.0, d_u_raw / 4.0, abs(screen_u_raw) < 4.0);
  result[6] = select(0.0, d_v_raw / 4.0, abs(screen_v_raw) < 4.0);
  result[7] = select(0.0, d_horizontal_raw / 2.0, horizontal_raw > 0.0 && horizontal_raw < 4.0);
  result[8] = select(0.0, d_vertical_raw / 2.0, vertical_raw > 0.0 && vertical_raw < 4.0);
  return result;
}

fn make_ray_query(instance_id: u32) -> RayQuery {
  let aabb_offset = instance_id * 6u;
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
  let center_world = 0.5 * (minimum + maximum);
  let size = max(maximum - minimum, vec3<f32>(1e-4));
  let delta = center_world - uniforms.query_center_threshold.xyz;
  let distance = max(length(delta), 1e-4);
  let ray = delta / distance;
  let radius = 0.5 * length(size);
  let forward = safe_normalize(uniforms.forward_tan_x.xyz, vec3<f32>(0.0, 0.0, -1.0));
  let world_y = vec3<f32>(0.0, 1.0, 0.0);
  let up_seed = select(world_y, vec3<f32>(0.0, 0.0, 1.0), abs(dot(forward, world_y)) > 0.98);
  let right = safe_normalize(cross(forward, up_seed), vec3<f32>(1.0, 0.0, 0.0));
  let up = safe_normalize(cross(right, forward), vec3<f32>(0.0, 1.0, 0.0));
  let tan_x = max(uniforms.forward_tan_x.w, 1e-4);
  let tan_y = max(uniforms.query_parameters.x, 1e-4);
  let dot_forward = dot(ray, forward);
  let dot_right = dot(ray, right);
  let dot_up = dot(ray, up);
  let denominator_x_raw = abs(dot_forward) * tan_x;
  let denominator_y_raw = abs(dot_forward) * tan_y;
  let denominator_x = max(denominator_x_raw, 1e-4);
  let denominator_y = max(denominator_y_raw, 1e-4);
  let screen_u_raw = dot_right / denominator_x;
  let screen_v_raw = dot_up / denominator_y;
  let safe_distance = max(distance, 1.0);
  let angular = radius / safe_distance;
  let horizontal_raw = angular / tan_x;
  let vertical_raw = angular / tan_y;
  var result: RayQuery;
  result.center[0] = ray.x;
  result.center[1] = ray.y;
  result.center[2] = ray.z;
  result.center[3] = clamp(log(1.0 + distance / 100.0) / 4.0, 0.0, 2.0) - 1.0;
  result.center[4] = clamp(dot_forward, -1.0, 1.0);
  result.center[5] = clamp(screen_u_raw, -4.0, 4.0) / 4.0;
  result.center[6] = clamp(screen_v_raw, -4.0, 4.0) / 4.0;
  result.center[7] = clamp(horizontal_raw, 0.0, 4.0) / 2.0 - 1.0;
  result.center[8] = clamp(vertical_raw, 0.0, 4.0) / 2.0 - 1.0;
  let differential_x = ray_differential(
    vec3<f32>(1.0, 0.0, 0.0), ray, distance, radius, forward, right, up,
    tan_x, tan_y, dot_forward, dot_right, dot_up, denominator_x_raw,
    denominator_y_raw, denominator_x, denominator_y, screen_u_raw,
    screen_v_raw, horizontal_raw, vertical_raw
  );
  let differential_z = ray_differential(
    vec3<f32>(0.0, 0.0, 1.0), ray, distance, radius, forward, right, up,
    tan_x, tan_y, dot_forward, dot_right, dot_up, denominator_x_raw,
    denominator_y_raw, denominator_x, denominator_y, screen_u_raw,
    screen_v_raw, horizontal_raw, vertical_raw
  );
  let viewcell_radius = uniforms.query_parameters.y;
  for (var feature = 0u; feature < 9u; feature += 1u) {
    let raw_axis = vec2<f32>(
      differential_x[feature] * viewcell_radius,
      differential_z[feature] * viewcell_radius
    );
    let row_norm = length(raw_axis);
    let budget = max(1.0 - abs(result.center[feature]), 0.0);
    let scale = select(1.0, budget / max(row_norm, 1e-12), row_norm > budget);
    result.axes[feature * 2u] = raw_axis.x * scale;
    result.axes[feature * 2u + 1u] = raw_axis.y * scale;
  }
  result.distance = distance;
  result.radius = radius;
  return result;
}

fn lookup_chi(argument: f32) -> f32 {
  let position = clamp(argument, 0.0, 320.0) * (8191.0 / 320.0);
  let lower = min(u32(floor(position)), 8190u);
  let fraction = position - f32(lower);
  return chi_table[lower] + fraction * (chi_table[lower + 1u] - chi_table[lower]);
}

fn spectral_features(query: RayQuery) -> array<f32, 64> {
  var output: array<f32, 64>;
  for (var frequency = 0u; frequency < 16u; frequency += 1u) {
    var phase_cycles = 0.0;
    var projected_x = 0.0;
    var projected_z = 0.0;
    for (var feature = 0u; feature < 9u; feature += 1u) {
      let cycle = frequency_cycles[frequency * 9u + feature];
      phase_cycles += query.center[feature] * cycle;
      projected_x += query.axes[feature * 2u] * cycle;
      projected_z += query.axes[feature * 2u + 1u] * cycle;
    }
    let phase = 6.283185307179586 * phase_cycles;
    let radial = 6.283185307179586 * length(vec2<f32>(projected_x, projected_z));
    let chi_s = lookup_chi(radial);
    let chi_2s = lookup_chi(2.0 * radial);
    let mean_sin = sin(phase) * chi_s;
    let mean_cos = cos(phase) * chi_s;
    let chi_squared = chi_s * chi_s;
    let phase_double_cos = cos(2.0 * phase);
    let shared_variance = 0.5 * (1.0 - chi_squared);
    let directed_variance = 0.5 * phase_double_cos * (chi_squared - chi_2s);
    let variance_sin = max(shared_variance + directed_variance, 0.0);
    let variance_cos = max(shared_variance - directed_variance, 0.0);
    var std_sin = select(0.0, sqrt(max(variance_sin, 1.1920929e-7)), variance_sin > 1.1920929e-7);
    var std_cos = select(0.0, sqrt(max(variance_cos, 1.1920929e-7)), variance_cos > 1.1920929e-7);
    if (radial == 0.0) {
      std_sin = 0.0;
      std_cos = 0.0;
    }
    let base = frequency * 4u;
    output[base] = mean_sin;
    output[base + 1u] = mean_cos;
    output[base + 2u] = std_sin;
    output[base + 3u] = std_cos;
  }
  return output;
}

fn low_rank_summary(query: RayQuery) -> array<f32, 4> {
  var output: array<f32, 4>;
  var axis_x_squared = 0.0;
  var axis_z_squared = 0.0;
  var maximum = 0.0;
  for (var feature = 0u; feature < 9u; feature += 1u) {
    let axis_x = query.axes[feature * 2u];
    let axis_z = query.axes[feature * 2u + 1u];
    axis_x_squared += axis_x * axis_x;
    axis_z_squared += axis_z * axis_z;
    maximum = max(maximum, max(abs(axis_x), abs(axis_z)));
  }
  output[0] = sqrt(axis_x_squared);
  output[1] = sqrt(axis_z_squared);
  output[2] = sqrt(axis_x_squared + axis_z_squared);
  output[3] = maximum;
  return output;
}

${layers}

fn survival_semantic(
  basis: array<f32, 4>,
  coefficients: array<f32, 28>,
  normalized_depth: f32,
) -> array<f32, 8> {
  var parameters: array<f32, 7>;
  for (var parameter = 0u; parameter < 7u; parameter += 1u) {
    var value = 0.0;
    for (var rank = 0u; rank < 4u; rank += 1u) {
      value += basis[rank] * coefficients[rank * 7u + parameter];
    }
    parameters[parameter] = value;
  }
  let no_block = sigmoid(parameters[0]);
  let weight_max = max(parameters[1], parameters[2]);
  let weight_exp0 = exp(parameters[1] - weight_max);
  let weight_exp1 = exp(parameters[2] - weight_max);
  let weight_sum = weight_exp0 + weight_exp1;
  var weight0 = weight_exp0 / weight_sum;
  var weight1 = weight_exp1 / weight_sum;
  var depth0 = 0.05 + 0.90 * sigmoid(parameters[3]);
  var depth1 = 0.05 + 0.90 * sigmoid(parameters[4]);
  var scale0 = 0.03 + softplus(parameters[5]);
  var scale1 = 0.03 + softplus(parameters[6]);
  if (depth0 > depth1) {
    let old_depth = depth0;
    let old_weight = weight0;
    let old_scale = scale0;
    depth0 = depth1;
    weight0 = weight1;
    scale0 = scale1;
    depth1 = old_depth;
    weight1 = old_weight;
    scale1 = old_scale;
  }
  let cdf0 = sigmoid((normalized_depth - depth0) / max(scale0, 1e-3));
  let cdf1 = sigmoid((normalized_depth - depth1) / max(scale1, 1e-3));
  let cdf = (1.0 - no_block) * (weight0 * cdf0 + weight1 * cdf1);
  let survival = clamp(1.0 - cdf, 1e-5, 1.0);
  let expected_depth = weight0 * depth0 + weight1 * depth1;
  let variance = weight0 * (depth0 - expected_depth) * (depth0 - expected_depth)
    + weight1 * (depth1 - expected_depth) * (depth1 - expected_depth);
  let uncertainty = sqrt(max(variance, 1e-8));
  let local_slope = (1.0 - no_block) * (
    weight0 / max(scale0, 1e-3) * cdf0 * (1.0 - cdf0)
    + weight1 / max(scale1, 1e-3) * cdf1 * (1.0 - cdf1)
  );
  var output: array<f32, 8>;
  output[0] = survival;
  output[1] = 1.0 - survival;
  output[2] = no_block;
  output[3] = expected_depth;
  output[4] = uncertainty;
  output[5] = weight0;
  output[6] = weight1;
  output[7] = local_slope;
  return output;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let instance_id = global_id.x;
  let count = u32(uniforms.depth_count.y);
  if (instance_id >= count || !intersects_frustum(instance_id, false)) { return; }
  let candidate_index = atomicAdd(&results[0], 1u);
  let query = make_ray_query(instance_id);
  let runtime_offset = instance_id * 124u;
  var coefficients: array<f32, 28>;
  for (var value_index = 0u; value_index < 28u; value_index += 1u) {
    coefficients[value_index] = runtime_value(runtime_offset + 96u + value_index);
  }
  let relation = relation_output(relation_hidden(coefficients));
  let spectral = spectral_features(query);
  let low_rank = low_rank_summary(query);
  var boundary_input: array<f32, 85>;
  for (var value_index = 0u; value_index < 64u; value_index += 1u) {
    boundary_input[value_index] = spectral[value_index];
  }
  for (var value_index = 0u; value_index < 9u; value_index += 1u) {
    boundary_input[64u + value_index] = query.center[value_index];
  }
  for (var value_index = 0u; value_index < 4u; value_index += 1u) {
    boundary_input[73u + value_index] = low_rank[value_index];
  }
  for (var value_index = 0u; value_index < 8u; value_index += 1u) {
    boundary_input[77u + value_index] = relation[value_index];
  }
  let boundary = boundary_output(boundary_hidden(boundary_input));
  var direction_input: array<f32, 19>;
  for (var value_index = 0u; value_index < 8u; value_index += 1u) {
    direction_input[value_index] = boundary[value_index];
    direction_input[8u + value_index] = relation[value_index];
  }
  direction_input[16] = query.center[0];
  direction_input[17] = query.center[1];
  direction_input[18] = query.center[2];
  let basis = direction_output(direction_hidden(direction_input));
  let depth_raw = log(1.0 + query.distance / (query.radius + uniforms.depth_count.x));
  let normalized_depth = clamp(
    (depth_raw - uniforms.query_parameters.z)
      / (uniforms.query_parameters.w - uniforms.query_parameters.z),
    0.0,
    1.0
  );
  let semantic = survival_semantic(basis, coefficients, normalized_depth);
  var trunk_input: array<f32, 130>;
  for (var value_index = 0u; value_index < 96u; value_index += 1u) {
    trunk_input[value_index] = runtime_value(runtime_offset + value_index);
  }
  for (var value_index = 0u; value_index < 4u; value_index += 1u) {
    trunk_input[96u + value_index] = basis[value_index];
  }
  for (var value_index = 0u; value_index < 8u; value_index += 1u) {
    trunk_input[100u + value_index] = semantic[value_index];
    trunk_input[108u + value_index] = boundary[value_index];
  }
  for (var value_index = 0u; value_index < 9u; value_index += 1u) {
    trunk_input[116u + value_index] = query.center[value_index];
  }
  for (var value_index = 0u; value_index < 4u; value_index += 1u) {
    trunk_input[125u + value_index] = low_rank[value_index];
  }
  trunk_input[129] = normalized_depth;
  let hidden = trunk_hidden1(trunk_hidden0(trunk_input));
  let logit = visibility_output(hidden)[0];
  let probability = sigmoid(logit);
  let global_glb_id = instance_to_glb[instance_id];
  atomicMax(&results[${layout.glbScores}u + global_glb_id], bitcast<u32>(probability));
  atomicOr(&results[${layout.glbFlags}u + global_glb_id], 4u);
  if (probability >= uniforms.query_center_threshold.w) {
    let visible_index = atomicAdd(&results[1], 1u);
    atomicStore(&results[${layout.modelVisibleIds}u + visible_index], instance_id);
    atomicOr(&results[${layout.glbFlags}u + global_glb_id], ${GLB_FLAG_MODEL_VISIBLE}u);
    if (intersects_frustum(instance_id, true)) {
      let render_index = atomicAdd(&results[2], 1u);
      atomicStore(&results[${layout.renderVisibleIds}u + render_index], instance_id);
      atomicOr(&results[${layout.glbFlags}u + global_glb_id], ${GLB_FLAG_RENDER_VISIBLE}u);
    }
  }
${diagnosticWrites}
}`;
  }

export function buildGlbCompactionShader() {
    const layout = this.resultLayout;
    return `
struct Uniforms {
  query_center_threshold: vec4<f32>,
  forward_tan_x: vec4<f32>,
  query_parameters: vec4<f32>,
  depth_count: vec4<f32>,
  candidate_planes: array<vec4<f32>, 6>,
  render_planes: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> results: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let glb_id = global_id.x;
  if (glb_id >= u32(uniforms.depth_count.w)) { return; }
  let flags = atomicLoad(&results[${layout.glbFlags}u + glb_id]);
  if ((flags & 4u) == 0u) { return; }
  let score_bits = atomicLoad(&results[${layout.glbScores}u + glb_id]);
  let score = bitcast<f32>(score_bits);
  if (score < uniforms.depth_count.z) { return; }
  let queue_index = atomicAdd(&results[3], 1u);
  atomicStore(&results[${layout.glbQueueIds}u + queue_index], glb_id);
  atomicStore(&results[${layout.glbQueueScores}u + queue_index], score_bits);
  atomicStore(&results[${layout.glbQueueFlags}u + queue_index], flags);
}`;
  }

export function buildRenderFilterShader() {
    const sourceLayout = this.resultLayout;
    const filterLayout = this.filterResultLayout || this._makeFilterResultLayout();
    return `
struct Uniforms {
  query_center_threshold: vec4<f32>,
  forward_tan_x: vec4<f32>,
  query_parameters: vec4<f32>,
  depth_count: vec4<f32>,
  candidate_planes: array<vec4<f32>, 6>,
  render_planes: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> instance_aabbs: array<f32>;
@group(0) @binding(2) var<storage, read> instance_to_glb: array<u32>;
@group(0) @binding(3) var<storage, read_write> prediction_results: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> filter_results: array<atomic<u32>>;

fn intersects_render_frustum(instance_id: u32) -> bool {
  let aabb_offset = instance_id * 6u;
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
  for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
    let plane = uniforms.render_planes[plane_index];
    let positive = select(minimum, maximum, plane.xyz >= vec3<f32>(0.0));
    if (dot(plane.xyz, positive) + plane.w < 0.0) { return false; }
  }
  return true;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let prediction_index = global_id.x;
  let prediction_count = atomicLoad(&prediction_results[1]);
  if (prediction_index >= prediction_count) { return; }
  let instance_id = atomicLoad(
    &prediction_results[${sourceLayout.modelVisibleIds}u + prediction_index]
  );
  if (!intersects_render_frustum(instance_id)) { return; }
  atomicAdd(&filter_results[0], 1u);
  let instance_word = instance_id >> 5u;
  let instance_mask = 1u << (instance_id & 31u);
  atomicOr(&filter_results[${filterLayout.renderVisibleBits}u + instance_word], instance_mask);
  let glb_id = instance_to_glb[instance_id];
  let glb_word = glb_id >> 5u;
  let glb_mask = 1u << (glb_id & 31u);
  let previous_glb_bits = atomicOr(
    &filter_results[${filterLayout.glbVisibleBits}u + glb_word],
    glb_mask
  );
  if ((previous_glb_bits & glb_mask) == 0u) {
    atomicAdd(&filter_results[1], 1u);
  }
}`;
  }
