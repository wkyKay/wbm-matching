"""Exercise moved entry points and root-relative configuration from another cwd."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared import common


class LayoutTests(unittest.TestCase):
    def test_direct_entrypoints_work_outside_workspace(self):
        entries = (
            "scripts/build_protocol.py", "scripts/build_retrieval_protocol.py",
            "methods/wu/wu.py", "methods/yu/yu.py", "tests/check_workers.py",
            "methods/wbbs/wbbs.py", "methods/wmhd/wmhd.py", "methods/nakazawa/nakazawa.py",
        )
        with tempfile.TemporaryDirectory() as elsewhere:
            for entry in entries:
                with self.subTest(entry=entry):
                    result = subprocess.run([sys.executable, "-B", str(ROOT / entry), "--help"],
                                            cwd=elsewhere, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_config_paths_remain_relative_to_repository_root(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                for method in ("wu", "yu", "wbbs", "wmhd", "nakazawa"):
                    for filename in ("config.json", "smoke.json"):
                        config = common.load_config(ROOT / "methods" / method / filename,
                                                    require_ks=method in ("wu", "yu"))
                        self.assertEqual(Path(config["data_file"]), ROOT.parent / "data/wm38k/Wafer_Map_Datasets.npz")
                        self.assertEqual(Path(config["split_file"]), ROOT / "protocol/deduplicated/split.csv")
                        self.assertEqual(Path(config["retrieval_protocol_dir"]), ROOT / "protocol/retrieval_v1")
                        self.assertEqual(Path(config["output_dir"]).parent, ROOT / "artifacts")
            finally:
                os.chdir(original_cwd)

    def test_shared_imports_have_one_module_identity(self):
        script = """
import sys
sys.path.insert(0, sys.argv[1])
from methods.wu import wu
from methods.yu import yu
from methods.wbbs import wbbs
from methods.wmhd import wmhd
from methods.nakazawa import nakazawa
from scripts import build_protocol, build_retrieval_protocol
from shared import common, data_protocol, retrieval_protocol
assert common.deduplicate_maps is data_protocol.deduplicate_maps
assert build_protocol.deduplicate_maps is data_protocol.deduplicate_maps
assert build_retrieval_protocol.build_retrieval_protocol is retrieval_protocol.build_retrieval_protocol
assert {'common', 'data_protocol', 'retrieval_protocol'}.isdisjoint(sys.modules)
"""
        with tempfile.TemporaryDirectory() as elsewhere:
            result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT)], cwd=elsewhere,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
