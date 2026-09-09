use std::alloc::{alloc, dealloc, Layout};
use std::mem;

const RUNTIME_FEATURE_DIM: usize = 124;
const GEOMETRY_DIM: usize = 96;
const SURVIVAL_DIM: usize = 28;
const RELATION_DIM: usize = 8;
const GLB_FLAG_MODEL_VISIBLE: u32 = 1;
const GLB_FLAG_RENDER_VISIBLE: u32 = 2;
const GLB_FLAG_TOUCHED: u32 = 4;
const TWO_PI: f32 = std::f32::consts::TAU;
const FLOAT32_EPSILON: f32 = 1.1920929e-7;

const RELATION0_WEIGHT: usize = 0;
const RELATION0_BIAS: usize = 448;
const RELATION2_WEIGHT: usize = 464;
const RELATION2_BIAS: usize = 592;
const DIRECTION0_WEIGHT: usize = 600;
const DIRECTION0_BIAS: usize = 1056;
const DIRECTION2_WEIGHT: usize = 1080;
const DIRECTION2_BIAS: usize = 1176;
const BOUNDARY0_WEIGHT: usize = 1180;
const BOUNDARY0_BIAS: usize = 5260;
const BOUNDARY2_WEIGHT: usize = 5308;
const BOUNDARY2_BIAS: usize = 5692;
const TRUNK0_WEIGHT: usize = 5700;
const TRUNK0_BIAS: usize = 14020;
const TRUNK2_WEIGHT: usize = 14084;
const TRUNK2_BIAS: usize = 18180;
const VISIBILITY_WEIGHT: usize = 18244;
const VISIBILITY_BIAS: usize = 18308;

#[derive(Clone, Copy)]
struct RuntimeState {
    num_instances: usize,
    num_glbs: usize,
    runtime_features: usize,
    instance_aabbs: usize,
    instance_to_glb: usize,
    weights: usize,
    frequencies: usize,
    chi: usize,
    relations: usize,
}

static mut STATE: RuntimeState = RuntimeState {
    num_instances: 0,
    num_glbs: 0,
    runtime_features: 0,
    instance_aabbs: 0,
    instance_to_glb: 0,
    weights: 0,
    frequencies: 0,
    chi: 0,
    relations: 0,
};

#[derive(Clone, Copy)]
enum Activation {
    None,
    Silu,
    Relu,
    Tanh,
}

struct RayQuery {
    center: [f32; 9],
    axes: [f32; 18],
    distance: f32,
    radius: f32,
}

#[no_mangle]
pub extern "C" fn wasm_simd_enabled() -> i32 {
    if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn allocate(byte_length: usize) -> *mut u8 {
    let layout = Layout::from_size_align(byte_length.max(1), 16).unwrap();
    unsafe { alloc(layout) }
}

#[no_mangle]
/// # Safety
/// `pointer` must have been returned by `allocate` with the same `byte_length`
/// and must not have been deallocated already.
pub unsafe extern "C" fn deallocate(pointer: *mut u8, byte_length: usize) {
    if !pointer.is_null() {
        let layout = Layout::from_size_align(byte_length.max(1), 16).unwrap();
        dealloc(pointer, layout);
    }
}

#[no_mangle]
/// # Safety
/// `source` and `destination` must each reference at least `count` readable or
/// writable elements respectively, and their regions must not overlap.
pub unsafe extern "C" fn decode_fp16(
    source: *const u16,
    count: usize,
    destination: *mut f32,
) -> i32 {
    if source.is_null() || destination.is_null() {
        return -1;
    }
    let input = std::slice::from_raw_parts(source, count);
    let output = std::slice::from_raw_parts_mut(destination, count);
    for (target, value) in output.iter_mut().zip(input.iter().copied()) {
        *target = half_to_f32(value);
    }
    0
}

#[no_mangle]
/// # Safety
/// Every pointer must remain valid in this module's linear memory for the
/// configured instance and model dimensions until the runtime is disposed.
pub unsafe extern "C" fn configure_runtime(
    num_instances: usize,
    num_glbs: usize,
    runtime_features: *const f32,
    instance_aabbs: *const f32,
    instance_to_glb: *const u32,
    weights: *const f32,
    frequencies: *const f32,
    chi: *const f32,
    relations: *mut f32,
) -> i32 {
    if num_instances == 0
        || num_glbs == 0
        || runtime_features.is_null()
        || instance_aabbs.is_null()
        || instance_to_glb.is_null()
        || weights.is_null()
        || frequencies.is_null()
        || chi.is_null()
        || relations.is_null()
    {
        return -1;
    }
    STATE = RuntimeState {
        num_instances,
        num_glbs,
        runtime_features: runtime_features as usize,
        instance_aabbs: instance_aabbs as usize,
        instance_to_glb: instance_to_glb as usize,
        weights: weights as usize,
        frequencies: frequencies as usize,
        chi: chi as usize,
        relations: relations as usize,
    };
    0
}

#[no_mangle]
/// # Safety
/// `configure_runtime` must have been called with valid, live model buffers.
pub unsafe extern "C" fn precompute_relations() -> i32 {
    let state = current_state();
    if state.num_instances == 0 {
        return -1;
    }
    let features = std::slice::from_raw_parts(
        state.runtime_features as *const f32,
        state.num_instances * RUNTIME_FEATURE_DIM,
    );
    let weights = std::slice::from_raw_parts(state.weights as *const f32, VISIBILITY_BIAS + 1);
    let relations = std::slice::from_raw_parts_mut(
        state.relations as *mut f32,
        state.num_instances * RELATION_DIM,
    );
    for instance_id in 0..state.num_instances {
        let feature_offset = instance_id * RUNTIME_FEATURE_DIM + GEOMETRY_DIM;
        let coefficients: &[f32; SURVIVAL_DIM] = features
            [feature_offset..feature_offset + SURVIVAL_DIM]
            .try_into()
            .unwrap();
        let mut hidden = [0.0; 16];
        let mut relation = [0.0; RELATION_DIM];
        linear(
            coefficients,
            &mut hidden,
            weights,
            RELATION0_WEIGHT,
            RELATION0_BIAS,
            Activation::Silu,
        );
        linear(
            &hidden,
            &mut relation,
            weights,
            RELATION2_WEIGHT,
            RELATION2_BIAS,
            Activation::None,
        );
        relations[instance_id * RELATION_DIM..(instance_id + 1) * RELATION_DIM]
            .copy_from_slice(&relation);
    }
    0
}

#[no_mangle]
/// Score a caller-supplied candidate list without candidate generation,
/// thresholding, GLB aggregation, or result compaction.
///
/// # Safety
/// `query_parameters` contains 12 floats, `candidate_ids` contains
/// `candidate_count` IDs, and `scores` has room for the same count.
pub unsafe extern "C" fn benchmark_candidates_v4(
    query_parameters: *const f32,
    candidate_ids: *const u32,
    candidate_count: usize,
    scores: *mut f32,
) -> i32 {
    let state = current_state();
    if state.num_instances == 0
        || query_parameters.is_null()
        || candidate_ids.is_null()
        || scores.is_null()
        || candidate_count > state.num_instances
    {
        return -1;
    }
    let query = std::slice::from_raw_parts(query_parameters, 12);
    let ids = std::slice::from_raw_parts(candidate_ids, candidate_count);
    let output = std::slice::from_raw_parts_mut(scores, candidate_count);
    let camera_center = [query[0], query[1], query[2]];
    let camera_forward = [query[3], query[4], query[5]];
    for (index, &raw_instance_id) in ids.iter().enumerate() {
        let instance_id = raw_instance_id as usize;
        if instance_id >= state.num_instances {
            return -2;
        }
        output[index] = score_instance(
            state,
            instance_id,
            camera_center,
            camera_forward,
            query[6],
            query[7],
            query[8],
            query[9],
            query[10],
            query[11],
        );
    }
    0
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
/// # Safety
/// All input and output pointers must reference buffers sized for the counts
/// registered by `configure_runtime`; output regions must not overlap inputs.
pub unsafe extern "C" fn predict_v4(
    query_parameters: *const f32,
    candidate_planes: *const f32,
    render_planes: *const f32,
    counters: *mut u32,
    visible_ids: *mut u32,
    render_ids: *mut u32,
    glb_scores: *mut f32,
    glb_flags: *mut u32,
    queue_ids: *mut u32,
    queue_scores: *mut f32,
    queue_flags: *mut u32,
    debug_candidate_ids: *mut u32,
    debug_scores: *mut f32,
) -> i32 {
    let state = current_state();
    if state.num_instances == 0
        || query_parameters.is_null()
        || candidate_planes.is_null()
        || render_planes.is_null()
        || counters.is_null()
        || visible_ids.is_null()
        || render_ids.is_null()
        || glb_scores.is_null()
        || glb_flags.is_null()
        || queue_ids.is_null()
        || queue_scores.is_null()
        || queue_flags.is_null()
    {
        return -1;
    }

    let query = std::slice::from_raw_parts(query_parameters, 14);
    let candidate = std::slice::from_raw_parts(candidate_planes, 24);
    let render = std::slice::from_raw_parts(render_planes, 24);
    let count_out = std::slice::from_raw_parts_mut(counters, 4);
    let visible_out = std::slice::from_raw_parts_mut(visible_ids, state.num_instances);
    let render_out = std::slice::from_raw_parts_mut(render_ids, state.num_instances);
    let scores_out = std::slice::from_raw_parts_mut(glb_scores, state.num_glbs);
    let flags_out = std::slice::from_raw_parts_mut(glb_flags, state.num_glbs);
    let queue_ids_out = std::slice::from_raw_parts_mut(queue_ids, state.num_glbs);
    let queue_scores_out = std::slice::from_raw_parts_mut(queue_scores, state.num_glbs);
    let queue_flags_out = std::slice::from_raw_parts_mut(queue_flags, state.num_glbs);
    let aabbs =
        std::slice::from_raw_parts(state.instance_aabbs as *const f32, state.num_instances * 6);
    let instance_to_glb =
        std::slice::from_raw_parts(state.instance_to_glb as *const u32, state.num_instances);
    let mut debug_ids = if debug_candidate_ids.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(
            debug_candidate_ids,
            state.num_instances,
        ))
    };
    let mut debug_probabilities = if debug_scores.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(
            debug_scores,
            state.num_instances,
        ))
    };

    count_out.fill(0);
    scores_out.fill(0.0);
    flags_out.fill(0);
    let camera_center = [query[0], query[1], query[2]];
    let camera_forward = [query[3], query[4], query[5]];
    let tan_x = query[6];
    let tan_y = query[7];
    let viewcell_radius = query[8];
    let depth_q01 = query[9];
    let depth_q99 = query[10];
    let depth_epsilon = query[11];
    let threshold = query[12];
    let prefetch_threshold = query[13];
    let mut candidate_count = 0usize;
    let mut visible_count = 0usize;
    let mut render_count = 0usize;

    for (instance_id, &raw_glb_id) in instance_to_glb.iter().enumerate() {
        if !intersects_frustum(aabbs, instance_id, candidate) {
            continue;
        }
        let probability = score_instance(
            state,
            instance_id,
            camera_center,
            camera_forward,
            tan_x,
            tan_y,
            viewcell_radius,
            depth_q01,
            depth_q99,
            depth_epsilon,
        );
        if let Some(ids) = debug_ids.as_deref_mut() {
            ids[candidate_count] = instance_id as u32;
        }
        if let Some(probabilities) = debug_probabilities.as_deref_mut() {
            probabilities[candidate_count] = probability;
        }
        candidate_count += 1;
        let glb_id = raw_glb_id as usize;
        flags_out[glb_id] |= GLB_FLAG_TOUCHED;
        scores_out[glb_id] = scores_out[glb_id].max(probability);
        if probability < threshold {
            continue;
        }
        visible_out[visible_count] = instance_id as u32;
        visible_count += 1;
        flags_out[glb_id] |= GLB_FLAG_MODEL_VISIBLE;
        if intersects_frustum(aabbs, instance_id, render) {
            render_out[render_count] = instance_id as u32;
            render_count += 1;
            flags_out[glb_id] |= GLB_FLAG_RENDER_VISIBLE;
        }
    }

    let mut queue_count = 0usize;
    for glb_id in 0..state.num_glbs {
        if flags_out[glb_id] & GLB_FLAG_TOUCHED == 0 || scores_out[glb_id] < prefetch_threshold {
            continue;
        }
        queue_ids_out[queue_count] = glb_id as u32;
        queue_scores_out[queue_count] = scores_out[glb_id];
        queue_flags_out[queue_count] = flags_out[glb_id] & !GLB_FLAG_TOUCHED;
        queue_count += 1;
    }
    count_out[0] = candidate_count as u32;
    count_out[1] = visible_count as u32;
    count_out[2] = render_count as u32;
    count_out[3] = queue_count as u32;
    0
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
/// # Safety
/// The visible-ID and plane inputs must be valid for their declared lengths;
/// both bitsets and the two counters must be writable and non-overlapping.
pub unsafe extern "C" fn refilter_v4(
    render_planes: *const f32,
    visible_ids: *const u32,
    visible_count: usize,
    instance_bits: *mut u32,
    instance_bit_words: usize,
    glb_bits: *mut u32,
    glb_bit_words: usize,
    counters: *mut u32,
) -> i32 {
    let state = current_state();
    if render_planes.is_null()
        || visible_ids.is_null()
        || instance_bits.is_null()
        || glb_bits.is_null()
        || counters.is_null()
        || visible_count > state.num_instances
    {
        return -1;
    }
    let planes = std::slice::from_raw_parts(render_planes, 24);
    let visible = std::slice::from_raw_parts(visible_ids, visible_count);
    let instance_output = std::slice::from_raw_parts_mut(instance_bits, instance_bit_words);
    let glb_output = std::slice::from_raw_parts_mut(glb_bits, glb_bit_words);
    let count_out = std::slice::from_raw_parts_mut(counters, 2);
    let aabbs =
        std::slice::from_raw_parts(state.instance_aabbs as *const f32, state.num_instances * 6);
    let instance_to_glb =
        std::slice::from_raw_parts(state.instance_to_glb as *const u32, state.num_instances);
    instance_output.fill(0);
    glb_output.fill(0);
    count_out.fill(0);
    for &raw_instance_id in visible {
        let instance_id = raw_instance_id as usize;
        if instance_id >= state.num_instances || !intersects_frustum(aabbs, instance_id, planes) {
            continue;
        }
        instance_output[instance_id >> 5] |= 1 << (instance_id & 31);
        count_out[0] += 1;
        let glb_id = instance_to_glb[instance_id] as usize;
        let word = glb_id >> 5;
        let mask = 1 << (glb_id & 31);
        if glb_output[word] & mask == 0 {
            glb_output[word] |= mask;
            count_out[1] += 1;
        }
    }
    0
}

unsafe fn current_state() -> RuntimeState {
    *std::ptr::addr_of!(STATE)
}

fn half_to_f32(value: u16) -> f32 {
    let sign = if value & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (value >> 10) & 0x1f;
    let mantissa = value & 0x03ff;
    match exponent {
        0 => sign * mantissa as f32 * 5.960_464_5e-8,
        0x1f if mantissa == 0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + mantissa as f32 / 1024.0) * 2.0_f32.powi(exponent as i32 - 15),
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn softplus(value: f32) -> f32 {
    (-value.abs()).exp().ln_1p() + value.max(0.0)
}

fn activate(value: f32, activation: Activation) -> f32 {
    match activation {
        Activation::None => value,
        Activation::Silu => value * sigmoid(value),
        Activation::Relu => value.max(0.0),
        Activation::Tanh => value.tanh(),
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn dot_product(weights: *const f32, input: *const f32, length: usize) -> f32 {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128_load};
    let mut vector_sum = f32x4_splat(0.0);
    let mut index = 0usize;
    while index + 4 <= length {
        let weight_vector = v128_load(weights.add(index) as *const _);
        let input_vector = v128_load(input.add(index) as *const _);
        vector_sum = f32x4_add(vector_sum, f32x4_mul(weight_vector, input_vector));
        index += 4;
    }
    let mut result = f32x4_extract_lane::<0>(vector_sum)
        + f32x4_extract_lane::<1>(vector_sum)
        + f32x4_extract_lane::<2>(vector_sum)
        + f32x4_extract_lane::<3>(vector_sum);
    while index < length {
        result += *weights.add(index) * *input.add(index);
        index += 1;
    }
    result
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
unsafe fn dot_product(weights: *const f32, input: *const f32, length: usize) -> f32 {
    let mut result = 0.0;
    for index in 0..length {
        result += *weights.add(index) * *input.add(index);
    }
    result
}

fn linear<const INPUT: usize, const OUTPUT: usize>(
    input: &[f32; INPUT],
    output: &mut [f32; OUTPUT],
    weights: &[f32],
    weight_offset: usize,
    bias_offset: usize,
    activation: Activation,
) {
    for row in 0..OUTPUT {
        let row_offset = weight_offset + row * INPUT;
        let sum = unsafe { dot_product(weights.as_ptr().add(row_offset), input.as_ptr(), INPUT) };
        output[row] = activate(weights[bias_offset + row] + sum, activation);
    }
}

fn normalize3(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length <= 1e-6 {
        fallback
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn intersects_frustum(aabbs: &[f32], instance_id: usize, planes: &[f32]) -> bool {
    let offset = instance_id * 6;
    for plane_index in 0..6 {
        let plane_offset = plane_index * 4;
        let nx = planes[plane_offset];
        let ny = planes[plane_offset + 1];
        let nz = planes[plane_offset + 2];
        let x = if nx >= 0.0 {
            aabbs[offset + 3]
        } else {
            aabbs[offset]
        };
        let y = if ny >= 0.0 {
            aabbs[offset + 4]
        } else {
            aabbs[offset + 1]
        };
        let z = if nz >= 0.0 {
            aabbs[offset + 5]
        } else {
            aabbs[offset + 2]
        };
        if nx * x + ny * y + nz * z + planes[plane_offset + 3] < 0.0 {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn make_ray_query(
    aabbs: &[f32],
    instance_id: usize,
    camera_center: [f32; 3],
    camera_forward: [f32; 3],
    tan_x: f32,
    tan_y: f32,
    viewcell_radius: f32,
) -> RayQuery {
    let offset = instance_id * 6;
    let center_world = [
        0.5 * (aabbs[offset] + aabbs[offset + 3]),
        0.5 * (aabbs[offset + 1] + aabbs[offset + 4]),
        0.5 * (aabbs[offset + 2] + aabbs[offset + 5]),
    ];
    let size = [
        (aabbs[offset + 3] - aabbs[offset]).max(1e-4),
        (aabbs[offset + 4] - aabbs[offset + 1]).max(1e-4),
        (aabbs[offset + 5] - aabbs[offset + 2]).max(1e-4),
    ];
    let delta = [
        center_world[0] - camera_center[0],
        center_world[1] - camera_center[1],
        center_world[2] - camera_center[2],
    ];
    let distance = dot3(delta, delta).sqrt().max(1e-4);
    let ray = [
        delta[0] / distance,
        delta[1] / distance,
        delta[2] / distance,
    ];
    let radius = 0.5 * dot3(size, size).sqrt();
    let forward = normalize3(camera_forward, [0.0, 0.0, -1.0]);
    let up_seed = if forward[1].abs() > 0.98 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let right = normalize3(cross3(forward, up_seed), [1.0, 0.0, 0.0]);
    let up = normalize3(cross3(right, forward), [0.0, 1.0, 0.0]);
    let safe_tan_x = tan_x.max(1e-4);
    let safe_tan_y = tan_y.max(1e-4);
    let dot_forward = dot3(ray, forward);
    let dot_right = dot3(ray, right);
    let dot_up = dot3(ray, up);
    let denominator_x_raw = dot_forward.abs() * safe_tan_x;
    let denominator_y_raw = dot_forward.abs() * safe_tan_y;
    let denominator_x = denominator_x_raw.max(1e-4);
    let denominator_y = denominator_y_raw.max(1e-4);
    let screen_u_raw = dot_right / denominator_x;
    let screen_v_raw = dot_up / denominator_y;
    let angular = radius / distance.max(1.0);
    let horizontal_raw = angular / safe_tan_x;
    let vertical_raw = angular / safe_tan_y;
    let mut center = [0.0; 9];
    center[0..3].copy_from_slice(&ray);
    center[3] = ((distance / 100.0).ln_1p() / 4.0).clamp(0.0, 2.0) - 1.0;
    center[4] = dot_forward.clamp(-1.0, 1.0);
    center[5] = screen_u_raw.clamp(-4.0, 4.0) / 4.0;
    center[6] = screen_v_raw.clamp(-4.0, 4.0) / 4.0;
    center[7] = horizontal_raw.clamp(0.0, 4.0) / 2.0 - 1.0;
    center[8] = vertical_raw.clamp(0.0, 4.0) / 2.0 - 1.0;

    let values = RayDifferentialValues {
        ray,
        distance,
        radius,
        forward,
        right,
        up,
        tan_x: safe_tan_x,
        tan_y: safe_tan_y,
        dot_forward,
        dot_right,
        dot_up,
        denominator_x_raw,
        denominator_y_raw,
        denominator_x,
        denominator_y,
        screen_u_raw,
        screen_v_raw,
        horizontal_raw,
        vertical_raw,
    };
    let differential_x = ray_differential([1.0, 0.0, 0.0], &values);
    let differential_z = ray_differential([0.0, 0.0, 1.0], &values);
    let mut axes = [0.0; 18];
    for feature in 0..9 {
        let raw_x = differential_x[feature] * viewcell_radius;
        let raw_z = differential_z[feature] * viewcell_radius;
        let norm = (raw_x * raw_x + raw_z * raw_z).sqrt();
        let budget = (1.0 - center[feature].abs()).max(0.0);
        let scale = if norm > budget {
            budget / norm.max(1e-12)
        } else {
            1.0
        };
        axes[feature * 2] = raw_x * scale;
        axes[feature * 2 + 1] = raw_z * scale;
    }
    RayQuery {
        center,
        axes,
        distance,
        radius,
    }
}

struct RayDifferentialValues {
    ray: [f32; 3],
    distance: f32,
    radius: f32,
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
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
}

fn ray_differential(axis: [f32; 3], values: &RayDifferentialValues) -> [f32; 9] {
    let ray_axis = dot3(values.ray, axis);
    let d_distance = -ray_axis;
    let d_ray = [
        -(axis[0] - values.ray[0] * ray_axis) / values.distance,
        -(axis[1] - values.ray[1] * ray_axis) / values.distance,
        -(axis[2] - values.ray[2] * ray_axis) / values.distance,
    ];
    let d_log = d_distance / (4.0 * (100.0 + values.distance));
    let d_forward = dot3(d_ray, values.forward);
    let d_right = dot3(d_ray, values.right);
    let d_up = dot3(d_ray, values.up);
    let d_denominator_x_raw = values.dot_forward.signum() * d_forward * values.tan_x;
    let d_denominator_y_raw = values.dot_forward.signum() * d_forward * values.tan_y;
    let d_denominator_x = if values.denominator_x_raw > 1e-4 {
        d_denominator_x_raw
    } else {
        0.0
    };
    let d_denominator_y = if values.denominator_y_raw > 1e-4 {
        d_denominator_y_raw
    } else {
        0.0
    };
    let d_u_raw = (d_right * values.denominator_x - values.dot_right * d_denominator_x)
        / (values.denominator_x * values.denominator_x);
    let d_v_raw = (d_up * values.denominator_y - values.dot_up * d_denominator_y)
        / (values.denominator_y * values.denominator_y);
    let d_angular = if values.distance > 1.0 {
        -values.radius * d_distance / (values.distance * values.distance)
    } else {
        0.0
    };
    [
        d_ray[0],
        d_ray[1],
        d_ray[2],
        d_log,
        d_forward,
        if values.screen_u_raw.abs() < 4.0 {
            d_u_raw / 4.0
        } else {
            0.0
        },
        if values.screen_v_raw.abs() < 4.0 {
            d_v_raw / 4.0
        } else {
            0.0
        },
        if values.horizontal_raw > 0.0 && values.horizontal_raw < 4.0 {
            d_angular / values.tan_x / 2.0
        } else {
            0.0
        },
        if values.vertical_raw > 0.0 && values.vertical_raw < 4.0 {
            d_angular / values.tan_y / 2.0
        } else {
            0.0
        },
    ]
}

fn lookup_chi(chi: &[f32], argument: f32) -> f32 {
    let position = argument.clamp(0.0, 320.0) * (8191.0 / 320.0);
    let lower = (position.floor() as usize).min(8190);
    let fraction = position - lower as f32;
    chi[lower] + fraction * (chi[lower + 1] - chi[lower])
}

fn spectral_features(query: &RayQuery, frequencies: &[f32], chi: &[f32]) -> [f32; 64] {
    let mut output = [0.0; 64];
    for frequency in 0..16 {
        let mut phase_cycles = 0.0;
        let mut projected_x = 0.0;
        let mut projected_z = 0.0;
        for feature in 0..9 {
            let cycle = frequencies[frequency * 9 + feature];
            phase_cycles += query.center[feature] * cycle;
            projected_x += query.axes[feature * 2] * cycle;
            projected_z += query.axes[feature * 2 + 1] * cycle;
        }
        let phase = TWO_PI * phase_cycles;
        let radial = TWO_PI * (projected_x * projected_x + projected_z * projected_z).sqrt();
        let chi_s = lookup_chi(chi, radial);
        let chi_2s = lookup_chi(chi, 2.0 * radial);
        let chi_squared = chi_s * chi_s;
        let shared_variance = 0.5 * (1.0 - chi_squared);
        let directed_variance = 0.5 * (2.0 * phase).cos() * (chi_squared - chi_2s);
        let variance_sin = (shared_variance + directed_variance).max(0.0);
        let variance_cos = (shared_variance - directed_variance).max(0.0);
        let base = frequency * 4;
        output[base] = phase.sin() * chi_s;
        output[base + 1] = phase.cos() * chi_s;
        output[base + 2] = if radial == 0.0 || variance_sin <= FLOAT32_EPSILON {
            0.0
        } else {
            variance_sin.sqrt()
        };
        output[base + 3] = if radial == 0.0 || variance_cos <= FLOAT32_EPSILON {
            0.0
        } else {
            variance_cos.sqrt()
        };
    }
    output
}

fn low_rank_summary(query: &RayQuery) -> [f32; 4] {
    let mut x_squared = 0.0;
    let mut z_squared = 0.0;
    let mut maximum = 0.0_f32;
    for feature in 0..9 {
        let x = query.axes[feature * 2];
        let z = query.axes[feature * 2 + 1];
        x_squared += x * x;
        z_squared += z * z;
        maximum = maximum.max(x.abs()).max(z.abs());
    }
    [
        x_squared.sqrt(),
        z_squared.sqrt(),
        (x_squared + z_squared).sqrt(),
        maximum,
    ]
}

fn survival_semantic(basis: &[f32; 4], coefficients: &[f32; 28], depth: f32) -> [f32; 8] {
    let mut parameters = [0.0; 7];
    for parameter in 0..7 {
        for rank in 0..4 {
            parameters[parameter] += basis[rank] * coefficients[rank * 7 + parameter];
        }
    }
    let no_block = sigmoid(parameters[0]);
    let weight_maximum = parameters[1].max(parameters[2]);
    let weight_exp0 = (parameters[1] - weight_maximum).exp();
    let weight_exp1 = (parameters[2] - weight_maximum).exp();
    let mut weight0 = weight_exp0 / (weight_exp0 + weight_exp1);
    let mut weight1 = weight_exp1 / (weight_exp0 + weight_exp1);
    let mut depth0 = 0.05 + 0.9 * sigmoid(parameters[3]);
    let mut depth1 = 0.05 + 0.9 * sigmoid(parameters[4]);
    let mut scale0 = 0.03 + softplus(parameters[5]);
    let mut scale1 = 0.03 + softplus(parameters[6]);
    if depth0 > depth1 {
        mem::swap(&mut depth0, &mut depth1);
        mem::swap(&mut weight0, &mut weight1);
        mem::swap(&mut scale0, &mut scale1);
    }
    let cdf0 = sigmoid((depth - depth0) / scale0.max(1e-3));
    let cdf1 = sigmoid((depth - depth1) / scale1.max(1e-3));
    let survival = (1.0 - (1.0 - no_block) * (weight0 * cdf0 + weight1 * cdf1)).clamp(1e-5, 1.0);
    let expected_depth = weight0 * depth0 + weight1 * depth1;
    let variance =
        weight0 * (depth0 - expected_depth).powi(2) + weight1 * (depth1 - expected_depth).powi(2);
    [
        survival,
        1.0 - survival,
        no_block,
        expected_depth,
        variance.max(1e-8).sqrt(),
        weight0,
        weight1,
        (1.0 - no_block)
            * (weight0 / scale0.max(1e-3) * cdf0 * (1.0 - cdf0)
                + weight1 / scale1.max(1e-3) * cdf1 * (1.0 - cdf1)),
    ]
}

#[allow(clippy::too_many_arguments)]
unsafe fn score_instance(
    state: RuntimeState,
    instance_id: usize,
    camera_center: [f32; 3],
    camera_forward: [f32; 3],
    tan_x: f32,
    tan_y: f32,
    viewcell_radius: f32,
    depth_q01: f32,
    depth_q99: f32,
    depth_epsilon: f32,
) -> f32 {
    let features = std::slice::from_raw_parts(
        state.runtime_features as *const f32,
        state.num_instances * RUNTIME_FEATURE_DIM,
    );
    let aabbs =
        std::slice::from_raw_parts(state.instance_aabbs as *const f32, state.num_instances * 6);
    let weights = std::slice::from_raw_parts(state.weights as *const f32, VISIBILITY_BIAS + 1);
    let frequencies = std::slice::from_raw_parts(state.frequencies as *const f32, 16 * 9);
    let chi = std::slice::from_raw_parts(state.chi as *const f32, 8192);
    let relations = std::slice::from_raw_parts(
        state.relations as *const f32,
        state.num_instances * RELATION_DIM,
    );
    let runtime_offset = instance_id * RUNTIME_FEATURE_DIM;
    let coefficients: &[f32; 28] = features
        [runtime_offset + GEOMETRY_DIM..runtime_offset + RUNTIME_FEATURE_DIM]
        .try_into()
        .unwrap();
    let relation: &[f32; 8] = relations[instance_id * 8..instance_id * 8 + 8]
        .try_into()
        .unwrap();
    let query = make_ray_query(
        aabbs,
        instance_id,
        camera_center,
        camera_forward,
        tan_x,
        tan_y,
        viewcell_radius,
    );
    let spectral = spectral_features(&query, frequencies, chi);
    let low_rank = low_rank_summary(&query);
    let mut boundary_input = [0.0; 85];
    boundary_input[0..64].copy_from_slice(&spectral);
    boundary_input[64..73].copy_from_slice(&query.center);
    boundary_input[73..77].copy_from_slice(&low_rank);
    boundary_input[77..85].copy_from_slice(relation);
    let mut boundary_hidden = [0.0; 48];
    let mut boundary = [0.0; 8];
    linear(
        &boundary_input,
        &mut boundary_hidden,
        weights,
        BOUNDARY0_WEIGHT,
        BOUNDARY0_BIAS,
        Activation::Silu,
    );
    linear(
        &boundary_hidden,
        &mut boundary,
        weights,
        BOUNDARY2_WEIGHT,
        BOUNDARY2_BIAS,
        Activation::Tanh,
    );
    let mut direction_input = [0.0; 19];
    direction_input[0..8].copy_from_slice(&boundary);
    direction_input[8..16].copy_from_slice(relation);
    direction_input[16..19].copy_from_slice(&query.center[0..3]);
    let mut direction_hidden = [0.0; 24];
    let mut basis = [0.0; 4];
    linear(
        &direction_input,
        &mut direction_hidden,
        weights,
        DIRECTION0_WEIGHT,
        DIRECTION0_BIAS,
        Activation::Silu,
    );
    linear(
        &direction_hidden,
        &mut basis,
        weights,
        DIRECTION2_WEIGHT,
        DIRECTION2_BIAS,
        Activation::Tanh,
    );
    let raw_depth = (query.distance / (query.radius + depth_epsilon)).ln_1p();
    let normalized_depth = ((raw_depth - depth_q01) / (depth_q99 - depth_q01)).clamp(0.0, 1.0);
    let semantic = survival_semantic(&basis, coefficients, normalized_depth);
    let mut trunk_input = [0.0; 130];
    trunk_input[0..96].copy_from_slice(&features[runtime_offset..runtime_offset + 96]);
    trunk_input[96..100].copy_from_slice(&basis);
    trunk_input[100..108].copy_from_slice(&semantic);
    trunk_input[108..116].copy_from_slice(&boundary);
    trunk_input[116..125].copy_from_slice(&query.center);
    trunk_input[125..129].copy_from_slice(&low_rank);
    trunk_input[129] = normalized_depth;
    let mut hidden0 = [0.0; 64];
    let mut hidden1 = [0.0; 64];
    let mut visibility = [0.0; 1];
    linear(
        &trunk_input,
        &mut hidden0,
        weights,
        TRUNK0_WEIGHT,
        TRUNK0_BIAS,
        Activation::Relu,
    );
    linear(
        &hidden0,
        &mut hidden1,
        weights,
        TRUNK2_WEIGHT,
        TRUNK2_BIAS,
        Activation::Relu,
    );
    linear(
        &hidden1,
        &mut visibility,
        weights,
        VISIBILITY_WEIGHT,
        VISIBILITY_BIAS,
        Activation::None,
    );
    sigmoid(visibility[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_decode_matches_known_values() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn sigmoid_is_stable_in_both_tails() {
        assert!(sigmoid(20.0) > 0.999_999);
        assert!(sigmoid(-20.0) < 0.000_001);
    }
}
