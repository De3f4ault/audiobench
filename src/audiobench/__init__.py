"""AudioBench — Core package."""

import os

# Prevent LiteLLM from attempting slow remote cost map downloads from GitHub
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

__version__ = "2.0.0"
