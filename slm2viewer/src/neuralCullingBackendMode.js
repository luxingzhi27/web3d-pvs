// 当前前端只保留实例级端侧神经剔除，避免历史实验链路继续分叉启动和调度语义。
const INSTANCE_PVS_ASSET_BASE_BY_SCENE = {
  'hkust-v3': './assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best',
  'ifcbench_fantasy_metropolis_instanced_v2': './assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best',
};

export const DEFAULT_INSTANCE_PVS_ASSET_BASE_URL = INSTANCE_PVS_ASSET_BASE_BY_SCENE['hkust-v3'];

export function getInstancePVSAssetBaseUrl(sceneName)
{
  return INSTANCE_PVS_ASSET_BASE_BY_SCENE[sceneName] || DEFAULT_INSTANCE_PVS_ASSET_BASE_URL;
}
