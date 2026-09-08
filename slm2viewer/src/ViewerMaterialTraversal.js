export function traverseMaterials(root, callback) {
  if (!root || typeof callback !== 'function') return;
  root.traverse((object) => {
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (material) callback(material, object);
    }
  });
}
