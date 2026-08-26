export function diffSortedIds(previousValues, nextValues) {
  const previous = previousValues || [];
  const next = nextValues || [];
  const added = [];
  const removed = [];
  let previousIndex = 0;
  let nextIndex = 0;

  while (previousIndex < previous.length || nextIndex < next.length) {
    const previousId = previousIndex < previous.length
      ? Number(previous[previousIndex])
      : Number.POSITIVE_INFINITY;
    const nextId = nextIndex < next.length
      ? Number(next[nextIndex])
      : Number.POSITIVE_INFINITY;
    if (previousId === nextId) {
      previousIndex += 1;
      nextIndex += 1;
    } else if (previousId < nextId) {
      removed.push(previousId);
      previousIndex += 1;
    } else {
      added.push(nextId);
      nextIndex += 1;
    }
  }

  return {
    added: Uint32Array.from(added),
    removed: Uint32Array.from(removed),
  };
}

export function applySortedIdDelta(previousValues, addedValues, removedValues) {
  const previous = previousValues || [];
  const added = addedValues || [];
  const removed = removedValues || [];
  const kept = [];
  let previousIndex = 0;
  let removedIndex = 0;

  while (previousIndex < previous.length) {
    const previousId = Number(previous[previousIndex]);
    while (removedIndex < removed.length && Number(removed[removedIndex]) < previousId) {
      removedIndex += 1;
    }
    if (removedIndex >= removed.length || Number(removed[removedIndex]) !== previousId) {
      kept.push(previousId);
    }
    previousIndex += 1;
  }

  const merged = [];
  let keptIndex = 0;
  let addedIndex = 0;
  while (keptIndex < kept.length || addedIndex < added.length) {
    const keptId = keptIndex < kept.length ? kept[keptIndex] : Number.POSITIVE_INFINITY;
    const addedId = addedIndex < added.length
      ? Number(added[addedIndex])
      : Number.POSITIVE_INFINITY;
    if (keptId === addedId) {
      merged.push(keptId);
      keptIndex += 1;
      addedIndex += 1;
    } else if (keptId < addedId) {
      merged.push(keptId);
      keptIndex += 1;
    } else {
      merged.push(addedId);
      addedIndex += 1;
    }
  }
  return merged;
}

export function bitsetFromIds(values, bitCount) {
  const words = new Uint32Array(Math.ceil(Math.max(0, Number(bitCount) || 0) / 32));
  for (const rawId of values || []) {
    const id = Number(rawId);
    if (!Number.isInteger(id) || id < 0 || id >= bitCount) continue;
    words[id >>> 5] |= (1 << (id & 31)) >>> 0;
  }
  return words;
}

function appendSetBitIds(output, rawBits, wordIndex, bitCount) {
  let bits = rawBits >>> 0;
  while (bits !== 0) {
    const leastBit = bits & -bits;
    const bitIndex = 31 - Math.clz32(leastBit);
    const id = wordIndex * 32 + bitIndex;
    if (id < bitCount) output.push(id);
    bits = (bits & (bits - 1)) >>> 0;
  }
}

export function countBitsetIds(values, bitCount) {
  const words = values || [];
  const maxBits = Math.max(0, Number(bitCount) || 0);
  let count = 0;
  for (let wordIndex = 0; wordIndex < words.length; wordIndex += 1) {
    let bits = Number(words[wordIndex]) >>> 0;
    if ((wordIndex + 1) * 32 > maxBits && (maxBits & 31) !== 0) {
      bits &= (2 ** (maxBits & 31) - 1) >>> 0;
    }
    bits -= (bits >>> 1) & 0x55555555;
    bits = (bits & 0x33333333) + ((bits >>> 2) & 0x33333333);
    count += (((bits + (bits >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  }
  return count;
}

export function diffIdBitsets(previousValues, nextValues, bitCount) {
  const previous = previousValues || [];
  const next = nextValues || [];
  const wordCount = Math.ceil(Math.max(0, Number(bitCount) || 0) / 32);
  const added = [];
  const removed = [];
  for (let wordIndex = 0; wordIndex < wordCount; wordIndex += 1) {
    const previousWord = Number(previous[wordIndex] || 0) >>> 0;
    const nextWord = Number(next[wordIndex] || 0) >>> 0;
    appendSetBitIds(added, nextWord & ~previousWord, wordIndex, bitCount);
    appendSetBitIds(removed, previousWord & ~nextWord, wordIndex, bitCount);
  }
  return {
    added: Uint32Array.from(added),
    removed: Uint32Array.from(removed),
  };
}
