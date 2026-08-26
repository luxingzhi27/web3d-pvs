function validSourceIndex(state, rawIndex) {
  const index = Number(rawIndex);
  return Number.isInteger(index) && index >= 0 && index < state.originalCount ? index : -1;
}

function copySourceToSlot(meshState, sourceIndex, slotIndex) {
  const mesh = meshState && meshState.mesh;
  const target = mesh && mesh.instanceMatrix ? mesh.instanceMatrix.array : null;
  const source = meshState && meshState.originalMatrixArray;
  if (!mesh || !target || !source || source.length < meshState.originalCount * 16) return false;
  const sourceOffset = sourceIndex * 16;
  target.set(source.subarray(sourceOffset, sourceOffset + 16), slotIndex * 16);
  return true;
}

function commitMeshState(meshState, changedSlots, activeCount) {
  const mesh = meshState && meshState.mesh;
  if (!mesh) return;
  mesh.count = activeCount;
  mesh.visible = activeCount > 0;
  mesh.frustumCulled = false;
  if (!mesh.instanceMatrix || changedSlots.length === 0) return;
  const uniqueSlots = Array.from(new Set(changedSlots)).sort((a, b) => a - b);
  if (typeof mesh.instanceMatrix.clearUpdateRanges === 'function') {
    mesh.instanceMatrix.clearUpdateRanges();
  }
  if (typeof mesh.instanceMatrix.addUpdateRange === 'function' && uniqueSlots.length <= 32) {
    for (const slot of uniqueSlots) mesh.instanceMatrix.addUpdateRange(slot * 16, 16);
  } else if (typeof mesh.instanceMatrix.addUpdateRange === 'function') {
    const first = uniqueSlots[0];
    const last = uniqueSlots[uniqueSlots.length - 1];
    mesh.instanceMatrix.addUpdateRange(first * 16, (last - first + 1) * 16);
  }
  mesh.instanceMatrix.needsUpdate = true;
}

export function initializeDenseInstancedState(state) {
  state.activeSourceIndices = [];
  state.slotBySourceIndex = new Int32Array(Math.max(0, Number(state.originalCount) || 0));
  state.slotBySourceIndex.fill(-1);
  return state;
}

export function clearDenseInstancedState(state) {
  initializeDenseInstancedState(state);
  for (const meshState of state.meshStates || []) commitMeshState(meshState, [], 0);
  return 0;
}

export function applyDenseInstancedDelta(state, addedSourceIndices, removedSourceIndices) {
  if (!state.slotBySourceIndex || !Array.isArray(state.activeSourceIndices)) {
    initializeDenseInstancedState(state);
  }
  const active = state.activeSourceIndices;
  const slots = state.slotBySourceIndex;
  const changedSlots = [];

  for (const rawSourceIndex of removedSourceIndices || []) {
    const sourceIndex = validSourceIndex(state, rawSourceIndex);
    if (sourceIndex < 0) continue;
    const removedSlot = slots[sourceIndex];
    if (removedSlot < 0) continue;
    const lastSlot = active.length - 1;
    const lastSourceIndex = active[lastSlot];
    if (removedSlot !== lastSlot) {
      active[removedSlot] = lastSourceIndex;
      slots[lastSourceIndex] = removedSlot;
      for (const meshState of state.meshStates || []) {
        copySourceToSlot(meshState, lastSourceIndex, removedSlot);
      }
      changedSlots.push(removedSlot);
    }
    active.pop();
    slots[sourceIndex] = -1;
  }

  for (const rawSourceIndex of addedSourceIndices || []) {
    const sourceIndex = validSourceIndex(state, rawSourceIndex);
    if (sourceIndex < 0 || slots[sourceIndex] >= 0) continue;
    const slotIndex = active.length;
    active.push(sourceIndex);
    slots[sourceIndex] = slotIndex;
    for (const meshState of state.meshStates || []) {
      copySourceToSlot(meshState, sourceIndex, slotIndex);
    }
    changedSlots.push(slotIndex);
  }

  for (const meshState of state.meshStates || []) {
    commitMeshState(meshState, changedSlots, active.length);
  }
  return active.length;
}

export function replaceDenseInstancedSlots(state, sourceIndices) {
  clearDenseInstancedState(state);
  return applyDenseInstancedDelta(state, sourceIndices, []);
}
