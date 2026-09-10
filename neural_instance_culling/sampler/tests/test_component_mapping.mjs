import assert from 'node:assert/strict';

import { buildComponentIdsByGlb } from '../web/component_mapping.js';

const sharedPrototype = buildComponentIdsByGlb(
  {
    componentRecords: [
      { componentGlobalId: 2, globalGlbId: 7 },
      { componentGlobalId: 1, globalGlbId: 7 },
      { componentGlobalId: 3, globalGlbId: 8 },
    ],
  },
  { entries: [{ globalId: 7 }, { globalId: 8 }] },
);
assert.deepEqual(sharedPrototype.get(7), [1, 2]);
assert.deepEqual(sharedPrototype.get(8), [3]);

const oneUnitPerResource = buildComponentIdsByGlb(
  {
    componentRecords: [
      { componentGlobalId: 0, globalGlbId: 0 },
      { componentGlobalId: 1, globalGlbId: 1 },
    ],
  },
  {
    entries: [
      { globalId: 0, componentGlobalIds: [0] },
      { globalId: 1, componentGlobalIds: [1] },
    ],
  },
);
assert.deepEqual(oneUnitPerResource.get(0), [0]);
assert.deepEqual(oneUnitPerResource.get(1), [1]);

assert.throws(
  () => buildComponentIdsByGlb(
    { componentRecords: [{ componentGlobalId: 0, globalGlbId: 0 }] },
    { entries: [{ globalId: 0, componentGlobalIds: [1] }] },
  ),
  /disagrees/,
);
assert.throws(
  () => buildComponentIdsByGlb(
    { componentRecords: [{ componentGlobalId: 0, globalGlbId: 1 }] },
    { entries: [{ globalId: 0 }] },
  ),
  /no component mapping/,
);

console.log(JSON.stringify({ status: 'ok', cases: 4 }));
