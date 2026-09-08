import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  NVIDIA_VULKAN_ICD,
  classifyGpuText,
  headlessWebGpuArgs,
  resolveVulkanEnvironment,
} from './chrome_gpu_flags.mjs';

const args = headlessWebGpuArgs({ screenSize: '1280,720', quietBrowser: true });
for (const required of [
  '--headless=new',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--use-vulkan',
  '--use-angle=vulkan',
  '--disable-vulkan-surface',
]) assert(args.includes(required), `Missing WebGPU hardware flag: ${required}`);
assert(!args.includes('--disable-software-rasterizer'));
assert(args.includes('--ozone-override-screen-size=1280,720'));

const environment = resolveVulkanEnvironment({ PATH: '/usr/bin' });
if (fs.existsSync(NVIDIA_VULKAN_ICD)) {
  assert.equal(environment.VK_ICD_FILENAMES, NVIDIA_VULKAN_ICD);
} else {
  assert.equal(environment.VK_ICD_FILENAMES, undefined);
}
assert.equal(resolveVulkanEnvironment({ VK_ICD_FILENAMES: '/custom/icd.json' }).VK_ICD_FILENAMES, '/custom/icd.json');
assert.equal(classifyGpuText(['nvidia', 'ampere']).hardware, true);
assert.equal(classifyGpuText(['google', 'swiftshader']).hardware, false);

console.log('Headless WebGPU hardware flag tests passed.');
