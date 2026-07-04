"""Wrapper to run gap_b_jarzynski_test.py with Lightning.ai paths."""
import importlib.util
import os

NPZ  = "/teamspace/studios/this_studio/claude_work/thermobridge_cv/audit/layer18_qk_pre_rope.npz"
CSV  = "/teamspace/studios/this_studio/claude_work/thermobridge_cv/validation/results/gap_b_jarzynski_results.csv"
SCRIPT = "/teamspace/studios/this_studio/claude_work/thermobridge_cv/validation/experiments/gap_b_jarzynski_test.py"

spec = importlib.util.spec_from_file_location("gap_b", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.NPZ_PATH = NPZ
m.OUT_CSV  = CSV
os.makedirs(os.path.dirname(CSV), exist_ok=True)

m.main()
