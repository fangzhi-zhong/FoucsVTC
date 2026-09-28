# Register the local FocusVTC zoom environment before starting rollout.
from .envs.mm_process_engine.visual_toolbox_qwen3_vtc import Qwen3VLVTCZoomTool
from .parallel_env import agent_rollout_loop
