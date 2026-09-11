import importlib.util
import unittest
from pathlib import Path

from verl.utils.dataset.rl_dataset import RLHFDataset


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "examples/agent/qwen3_vl_vtc_tool/zoom_region_tools.json"
ZOOM_MODULE_PATH = ROOT / "verl/workers/agent/envs/mm_process_engine/vtc_zoom.py"
SPEC = importlib.util.spec_from_file_location("vtc_zoom_schema_under_test", ZOOM_MODULE_PATH)
ZOOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ZOOM)


class RLHFDatasetToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = RLHFDataset._load_tools_schema(str(SCHEMA_PATH))

    def setUp(self):
        self.dataset = object.__new__(RLHFDataset)
        self.dataset.tools_key = "tools"
        self.dataset.tools_enabled_key = "enable_tools"
        self.dataset.default_tools = self.schema

    def test_global_schema_matches_zoom_executor(self):
        self.assertEqual(self.schema, [ZOOM.TOOL_SCHEMA])

    def test_global_schema_is_default(self):
        self.assertIs(self.dataset._resolve_tools({}), self.schema)

    def test_false_gate_disables_tools(self):
        self.assertIsNone(self.dataset._resolve_tools({"enable_tools": False}))

    def test_per_row_schema_overrides_global_schema(self):
        row_tools = [{"type": "function", "function": {"name": "special"}}]
        self.assertIs(self.dataset._resolve_tools({"tools": row_tools}), row_tools)

    def test_gate_must_be_boolean_or_null(self):
        with self.assertRaisesRegex(ValueError, "enable_tools must be a boolean or null"):
            self.dataset._resolve_tools({"enable_tools": "false"})


if __name__ == "__main__":
    unittest.main()
