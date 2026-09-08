import fs from 'node:fs';

export const NVIDIA_VULKAN_ICD = '/etc/vulkan/icd.d/nvidia_icd.json';

export function resolveVulkanEnvironment(environment = process.env) {
  if (environment.VK_ICD_FILENAMES || !fs.existsSync(NVIDIA_VULKAN_ICD)) {
    return { ...environment };
  }
  return { ...environment, VK_ICD_FILENAMES: NVIDIA_VULKAN_ICD };
}

export function headlessWebGpuArgs({
  screenSize = null,
  quietBrowser = false,
} = {}) {
  return [
    '--headless=new',
    '--no-sandbox',
    '--no-first-run',
    '--disable-dev-shm-usage',
    '--enable-gpu',
    '--enable-unsafe-webgpu',
    '--enable-webgpu',
    '--enable-webgl',
    '--enable-features=Vulkan',
    '--use-vulkan',
    '--use-angle=vulkan',
    '--disable-vulkan-surface',
    '--enable-accelerated-2d-canvas',
    '--enable-zero-copy',
    '--ignore-gpu-blocklist',
    '--disable-gpu-sandbox',
    ...(screenSize ? [
      '--ozone-platform=headless',
      `--ozone-override-screen-size=${screenSize}`,
    ] : []),
    ...(quietBrowser ? ['--disable-background-networking', '--disable-extensions'] : []),
  ];
}

export function classifyGpuText(values) {
  const text = values.map((value) => String(value || '')).join(' ').trim();
  const softwareMarkers = text.match(/swiftshader|llvmpipe|softpipe|swrast|software/ig) || [];
  return {
    hardware: Boolean(text) && softwareMarkers.length === 0,
    softwareMarkers: [...new Set(softwareMarkers.map((value) => value.toLowerCase()))],
  };
}
