const MAP_FRAGMENT = '#include <map_fragment>';
const ALPHA_ONLY_MAP_FRAGMENT = `
#ifdef USE_MAP
  vec4 sampledDiffuseColor = texture2D( map, vMapUv );
  #ifdef DECODE_VIDEO_TEXTURE
    sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
  #endif
  diffuseColor.a *= sampledDiffuseColor.a;
#endif
`;

export function patchColorIdAlphaFragmentShader(fragmentShader) {
  if (typeof fragmentShader !== 'string' || !fragmentShader.includes(MAP_FRAGMENT)) {
    throw new Error('Color-ID alpha-test material cannot locate map_fragment');
  }
  return fragmentShader.replace(MAP_FRAGMENT, ALPHA_ONLY_MAP_FRAGMENT);
}
