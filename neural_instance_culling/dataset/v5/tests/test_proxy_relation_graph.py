from __future__ import annotations

import unittest

import numpy as np

from neural_instance_culling.dataset.v5.proxy_relation_graph import (
    _aabb_corners,
    _projected_overlap_candidates,
    build_proxy_relation_graph,
    icosahedron12_directions,
    icosahedron12_screen_bases,
)


def _brute_overlap_candidates(u_min, u_max, v_min, v_max):
    result = [[] for _ in range(len(u_min))]
    for left in range(len(u_min)):
        for right in range(left + 1, len(u_min)):
            if (
                min(u_max[left], u_max[right]) > max(u_min[left], u_min[right])
                and min(v_max[left], v_max[right]) > max(v_min[left], v_min[right])
            ):
                result[left].append(right)
                result[right].append(left)
    return result


class ProxyRelationGraphTest(unittest.TestCase):
    def test_degenerate_placeholder_has_no_relation_edges(self) -> None:
        boxes = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, -0.5, -0.5], [2.0, 0.5, 0.5]],
            ],
            dtype=np.float32,
        )
        graph = build_proxy_relation_graph(boxes)
        self.assertFalse(bool(graph.valid_mask[0].any()))
        self.assertFalse(bool((graph.source_ids[graph.valid_mask] == 0).any()))

    def test_sweep_broad_phase_matches_brute_force_for_every_anchor(self) -> None:
        rng = np.random.default_rng(91)
        boxes = np.stack(
            [rng.uniform(-8.0, 8.0, size=(32, 3)), rng.uniform(-8.0, 8.0, size=(32, 3))], axis=1
        )
        boxes.sort(axis=1)
        corners = _aabb_corners(boxes)
        anchors = icosahedron12_directions().astype(np.float64)
        right, up = icosahedron12_screen_bases()
        for index, anchor in enumerate(anchors):
            u = corners @ right[index].astype(np.float64)
            v = corners @ up[index].astype(np.float64)
            u_min, u_max = u.min(axis=1), u.max(axis=1)
            v_min, v_max = v.min(axis=1), v.max(axis=1)
            sweep = _projected_overlap_candidates(u_min, u_max, v_min, v_max)
            brute = _brute_overlap_candidates(u_min, u_max, v_min, v_max)
            self.assertEqual([set(row) for row in sweep], [set(row) for row in brute])

    def test_front_source_is_kept_and_rear_source_is_rejected(self) -> None:
        anchors = icosahedron12_directions()
        direction = anchors[0].astype(np.float64)
        target = np.asarray([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        front_center = direction * 4.0
        rear_center = -direction * 4.0
        boxes = np.asarray([
            target,
            np.stack([front_center - 0.5, front_center + 0.5]),
            np.stack([rear_center - 0.5, rear_center + 0.5]),
        ])
        graph = build_proxy_relation_graph(boxes)
        sources = graph.source_ids[0, 0][graph.valid_mask[0, 0]]
        self.assertIn(1, sources)
        self.assertNotIn(2, sources)
        self.assertFalse(graph.manifest["usesVisibilityLabels"])
        self.assertEqual(graph.edge_features.shape, (3, 12, 8, 8))

    def test_thousand_unit_smoke(self) -> None:
        rng = np.random.default_rng(7)
        centers = rng.uniform(-100.0, 100.0, size=(1000, 3))
        half = rng.uniform(0.2, 1.5, size=(1000, 3))
        boxes = np.stack([centers - half, centers + half], axis=1)
        graph = build_proxy_relation_graph(boxes)
        self.assertEqual(graph.source_ids.shape, (1000, 12, 8))
        self.assertLessEqual(graph.edge_count, 1000 * 12 * 8)
        self.assertTrue(np.isfinite(graph.edge_features).all())


if __name__ == "__main__":
    unittest.main()
