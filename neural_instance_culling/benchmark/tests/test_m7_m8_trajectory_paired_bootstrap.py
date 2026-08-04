import unittest

from neural_instance_culling.benchmark.summarize_m7_m8_trajectory_paired_bootstrap import (
    _self_test,
    per_pose_metrics,
    validate_pair,
)


def _model(*, candidate_count=2, first_complete=50.0):
    return {
        "trajectory": {
            "poses": [
                {
                    "poseIndex": 1,
                    "timeMs": 0.0,
                    "candidateCount": candidate_count,
                    "candidateGlbCount": 2,
                    "weakUtilityByGlb": {"0": 1.0},
                    "weakUtilityTotal": 1.0,
                },
                {
                    "poseIndex": 2,
                    "timeMs": 100.0,
                    "candidateCount": candidate_count,
                    "candidateGlbCount": 2,
                    "weakUtilityByGlb": {"1": 1.0},
                    "weakUtilityTotal": 1.0,
                },
            ]
        },
        "replay": {
            "replayHorizonMs": 200.0,
            "cache": {"initialGlbIds": []},
            "events": {
                "decodeUploads": [
                    {"glbId": 0, "completeMs": first_complete},
                    {"glbId": 1, "completeMs": first_complete + 10.0},
                ]
            },
        },
    }


class M7M8TrajectoryPairedBootstrapTests(unittest.TestCase):
    def test_completion_events_are_reconstructed_per_pose(self):
        rows = per_pose_metrics(_model())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["deadlineUtilityRecall"], 1.0)
        self.assertEqual(rows[1]["deadlineUtilityRecall"], 1.0)

    def test_pairing_rejects_candidate_mismatch(self):
        left = _model(candidate_count=2)
        right = _model(candidate_count=3)
        with self.assertRaises(ValueError):
            validate_pair(left, right)

    def test_registered_self_test(self):
        self.assertEqual(_self_test()["status"], "passed")


if __name__ == "__main__":
    unittest.main()
