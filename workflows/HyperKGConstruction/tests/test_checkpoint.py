from __future__ import annotations

import tempfile
import unittest

from workflows.HyperKGConstruction.checkpoint import CheckpointError, HyperKGCheckpointStore


class CheckpointStoreTest(unittest.TestCase):
    def test_append_load_dedup_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HyperKGCheckpointStore(tmpdir)
            store.prepare({"fingerprint_id": "same"})
            store.append_record("claim_split", {"object_id": "P:1", "index": 0, "value": 1})
            store.append_record("claim_split", {"object_id": "P:1", "index": 0, "value": 2})
            store.append_record("claim_split", {"object_id": "P:2", "index": 1, "value": 3})

            loaded = store.load_records("claim_split")
            self.assertEqual(loaded["P:1"]["value"], 2)
            self.assertEqual(loaded["P:2"]["value"], 3)

            store.write_records("claim_split", loaded.values())
            compacted = store.load_records("claim_split")
            self.assertEqual(len(compacted), 2)
            self.assertEqual(compacted["P:1"]["value"], 2)

    def test_fingerprint_mismatch_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HyperKGCheckpointStore(tmpdir)
            store.prepare({"fingerprint_id": "old"})

            resumed = HyperKGCheckpointStore(tmpdir)
            with self.assertRaises(CheckpointError):
                resumed.prepare({"fingerprint_id": "new"})


if __name__ == "__main__":
    unittest.main()
