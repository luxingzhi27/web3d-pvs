function integer(value, label) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
  return number;
}

function sortedUniqueIds(values, label) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error(`${label} must be a non-empty array`);
  }
  const ids = values.map((value, index) => integer(value, `${label}[${index}]`));
  ids.sort((left, right) => left - right);
  for (let index = 1; index < ids.length; index += 1) {
    if (ids[index] === ids[index - 1]) throw new Error(`${label} contains duplicate component IDs`);
  }
  return ids;
}

function equalIds(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

export function buildComponentIdsByGlb(runtimeMeta, glbIndex) {
  const records = runtimeMeta?.componentRecords;
  const entries = glbIndex?.entries;
  if (!Array.isArray(records) || records.length === 0) {
    throw new Error('runtimeVisibilityMeta.componentRecords must be non-empty');
  }
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error('glbIndex.entries must be non-empty');
  }

  const inferred = new Map();
  const componentIds = new Set();
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    const componentId = integer(record?.componentGlobalId, `componentRecords[${index}].componentGlobalId`);
    const glbId = integer(record?.globalGlbId, `componentRecords[${index}].globalGlbId`);
    if (componentIds.has(componentId)) throw new Error(`duplicate componentGlobalId ${componentId}`);
    componentIds.add(componentId);
    if (!inferred.has(glbId)) inferred.set(glbId, []);
    inferred.get(glbId).push(componentId);
  }
  for (const ids of inferred.values()) ids.sort((left, right) => left - right);

  const result = new Map();
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const glbId = integer(entry?.globalId, `glbIndex.entries[${index}].globalId`);
    if (result.has(glbId)) throw new Error(`duplicate glbIndex globalId ${glbId}`);
    const fromRuntime = inferred.get(glbId) || [];
    const explicit = entry?.componentGlobalIds === undefined
      ? null
      : sortedUniqueIds(entry.componentGlobalIds, `glbIndex.entries[${index}].componentGlobalIds`);
    if (explicit && fromRuntime.length > 0 && !equalIds(explicit, fromRuntime)) {
      throw new Error(`component mapping disagrees for globalGlbId ${glbId}`);
    }
    const ids = explicit || fromRuntime;
    if (ids.length === 0) throw new Error(`globalGlbId ${glbId} has no component mapping`);
    result.set(glbId, ids.slice());
  }

  for (const glbId of inferred.keys()) {
    if (!result.has(glbId)) throw new Error(`runtime globalGlbId ${glbId} is missing from glbIndex`);
  }
  return result;
}
