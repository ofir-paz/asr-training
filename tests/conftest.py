import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def train_whisper_module():
    """Loads train-whisper.py as an importable module.

    Its filename has a hyphen, so it can't be `import`-ed normally. Session-scoped since
    importing it pulls in transformers/torch/peft/etc. (the expensive part) - that cost is
    paid once and reused by every test that needs the module, not per test.
    """
    spec = importlib.util.spec_from_file_location("train_whisper", REPO_ROOT / "train-whisper.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_whisper"] = module
    spec.loader.exec_module(module)
    return module
