import assert from 'node:assert/strict';
import { Object3D } from 'three';
import { traverseMaterials } from '../src/ViewerMaterialTraversal.js';

const root = new Object3D();
const single = new Object3D();
const multiple = new Object3D();
const first = { name: 'first' };
const second = { name: 'second' };
const third = { name: 'third' };
single.material = first;
multiple.material = [second, null, third];
root.add(single, multiple);

const visited = [];
traverseMaterials(root, (material, object) => visited.push([material.name, object]));
assert.deepEqual(visited.map(([name]) => name), ['first', 'second', 'third']);
assert.equal(visited[0][1], single);
assert.equal(visited[1][1], multiple);
assert.doesNotThrow(() => traverseMaterials(null, () => {}));

console.log('Viewer material traversal tests passed.');
