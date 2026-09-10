import assert from 'node:assert/strict';

import { patchColorIdAlphaFragmentShader } from '../web/color_id_alpha.js';

const source = `
void main() {
  vec4 diffuseColor = vec4( 1.0 );
  #include <map_fragment>
  #include <alphatest_fragment>
}
`;
const patched = patchColorIdAlphaFragmentShader(source);
assert.doesNotMatch(patched, /#include <map_fragment>/);
assert.match(patched, /diffuseColor\.a \*= sampledDiffuseColor\.a/);
assert.match(patched, /#include <alphatest_fragment>/);
assert.doesNotMatch(patched, /diffuseColor \*= sampledDiffuseColor/);
assert.throws(() => patchColorIdAlphaFragmentShader('void main() {}'), /map_fragment/);

console.log(JSON.stringify({ status: 'ok', cases: 2 }));
