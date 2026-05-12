from __future__ import annotations

import unittest

from workflows.HyperKGConstruction.run import build_parser


class RunCliTest(unittest.TestCase):
    def test_common_runtime_flags_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--summaries",
                "summaries.json",
                "--local-kgs",
                "local_kgs.json",
                "--unified-kg",
                "unified.json",
                "--workers",
                "4",
                "--checkpoint-dir",
                "output/checkpoints",
                "--no-checkpoint",
                "--no-resume",
                "--no-progress",
                "--log-file",
                "output/run.log",
            ]
        )
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.checkpoint_dir, "output/checkpoints")
        self.assertEqual(args.no_checkpoint, True)
        self.assertEqual(args.no_resume, True)
        self.assertEqual(args.no_progress, True)
        self.assertEqual(args.log_file, "output/run.log")


if __name__ == "__main__":
    unittest.main()
