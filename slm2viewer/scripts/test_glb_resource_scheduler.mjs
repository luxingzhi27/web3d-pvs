#!/usr/bin/env node
import assert from 'node:assert/strict';
import {
  classifyGlbSchedule,
  GlbResourceScheduler,
} from '../src/GlbResourceScheduler.js';

function modelInfo(id, score) {
  return { id, weight: score, idMode: 'global-glb' };
}

function scheduleEntry(info) {
  return {
    hash: `task-${info.id}`,
    glbId: info.id,
    score: info.weight,
    modelInfo: info,
  };
}

const predicted = Array.from({ length: 600 }, (_, id) => modelInfo(id, 1 - id / 1000));
const speculative = [modelInfo(700, 0.03), modelInfo(701, 0.02)];
const renderIds = Array.from({ length: 500 }, (_, id) => id);
const groups = classifyGlbSchedule(predicted, speculative, renderIds);
assert.equal(groups.urgent.length, 500, 'Every current-frustum GLB must remain urgent.');
assert.equal(groups.warm.length, 100);
assert.equal(groups.speculative.length, 2);
assert.ok(groups.urgent.every((item) => item.prefetch === false));
assert.ok(groups.warm.every((item) => item.prefetch === true));

const inconsistent = classifyGlbSchedule([], [modelInfo(9, 0.05)], [9]);
assert.deepEqual(inconsistent.urgent.map((item) => item.id), [9]);
assert.equal(inconsistent.speculative.length, 0, 'A render GLB cannot remain speculative.');

let clock = 1000;
const scheduler = new GlbResourceScheduler({ clock: () => clock, maxRetries: 2 });
scheduler.setPlan({
  urgent: groups.urgent.map(scheduleEntry),
  warm: groups.warm.map(scheduleEntry),
  speculative: groups.speculative.map(scheduleEntry),
}, 1);
let snapshot = scheduler.getSnapshot();
assert.equal(snapshot.queuedImmediate, 500);
assert.equal(snapshot.queuedPrefetch, 102);
assert.equal(snapshot.urgentMissing, 500);

for (let index = 0; index < 500; index += 1) {
  const entry = scheduler.takeNext();
  assert.equal(entry.tier, 'urgent', 'Warm work started before all urgent work was dispatched.');
  scheduler.markResident(entry.hash);
}
assert.equal(scheduler.getSnapshot().urgentMissing, 0);
assert.equal(scheduler.takeNext().tier, 'warm');

const promotedScheduler = new GlbResourceScheduler({ clock: () => clock });
const warm = scheduleEntry(modelInfo(42, 0.4));
promotedScheduler.setPlan({ warm: [warm] }, 1);
assert.equal(promotedScheduler.getSnapshot().queuedPrefetch, 1);
assert.equal(promotedScheduler.promote([warm]), 1);
assert.equal(promotedScheduler.getSnapshot().queuedImmediate, 1);
assert.equal(promotedScheduler.takeNext().modelInfo.prefetch, false);

const demotedScheduler = new GlbResourceScheduler({ clock: () => clock });
demotedScheduler.setPlan({ urgent: [warm] }, 1);
assert.equal(demotedScheduler.reclassify([warm], 'warm'), 1);
assert.equal(demotedScheduler.getSnapshot().queuedImmediate, 0);
assert.equal(demotedScheduler.getSnapshot().queuedPrefetch, 1);
assert.equal(demotedScheduler.takeNext().modelInfo.prefetch, true);

const retryScheduler = new GlbResourceScheduler({
  clock: () => clock,
  maxRetries: 2,
  retryDelaysMs: [250, 1000],
});
const retryItem = scheduleEntry(modelInfo(88, 0.8));
retryScheduler.setPlan({ urgent: [retryItem] }, 1);
const firstAttempt = retryScheduler.takeNext();
assert.equal(firstAttempt.glbId, 88);
assert.deepEqual(retryScheduler.markFailed(firstAttempt.hash, new Error('network')), {
  retrying: true,
  delayMs: 250,
});
assert.equal(retryScheduler.takeNext(), null);
clock += 250;
assert.equal(retryScheduler.takeNext().glbId, 88);

const changed = new GlbResourceScheduler({ clock: () => clock });
const changingItem = scheduleEntry(modelInfo(12, 0.6));
changed.setPlan({ urgent: [changingItem] }, 1);
changed.setPlan({ warm: [changingItem] }, 2);
assert.equal(changed.getSnapshot().queuedImmediate, 0);
assert.equal(changed.getSnapshot().queuedPrefetch, 1);

const stale = new GlbResourceScheduler({ clock: () => clock });
const staleItem = scheduleEntry(modelInfo(99, 0.2));
stale.setPlan({ warm: [staleItem] }, 1);
assert.equal(stale.takeNext().glbId, 99);
stale.setPlan({}, 2);
stale.markDiscarded(staleItem.hash);
stale.setPlan({ urgent: [staleItem] }, 3);
assert.equal(stale.takeNext().glbId, 99, 'A discarded stale task could not be scheduled again.');

scheduler.reset();
promotedScheduler.reset();
demotedScheduler.reset();
retryScheduler.reset();
changed.reset();
stale.reset();
console.log('GLB resource scheduler tests passed.');
