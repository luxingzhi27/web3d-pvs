// 只有已经导出当前 V4 运行包的场景才允许启用神经剔除。
const INSTANCE_PVS_ASSET_BASE_BY_SCENE = {
  'hkust-v3': './assets/neural_instance_culling/pvs_mainline_v4',
};

export const DEFAULT_INSTANCE_PVS_ASSET_BASE_URL = INSTANCE_PVS_ASSET_BASE_BY_SCENE['hkust-v3'];

export function getInstancePVSAssetBaseUrl(sceneName)
{
  return INSTANCE_PVS_ASSET_BASE_BY_SCENE[sceneName] || null;
}
