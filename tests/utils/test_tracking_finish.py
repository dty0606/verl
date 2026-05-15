import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRACKING = ROOT / "verl" / "utils" / "tracking.py"
SPEC = importlib.util.spec_from_file_location("tracking_module", TRACKING)
tracking_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules.setdefault("orjson", types.SimpleNamespace(OPT_SERIALIZE_NUMPY=0, dumps=lambda data, option=0: b"{}"))
sys.modules[SPEC.name] = tracking_module
SPEC.loader.exec_module(tracking_module)

Tracking = tracking_module.Tracking


class FakeWandb:
    def __init__(self):
        self.finish_calls = 0

    def finish(self, exit_code=0):
        assert exit_code == 0
        self.finish_calls += 1


def test_tracking_finish_is_explicit_and_idempotent():
    tracker = Tracking.__new__(Tracking)
    fake_wandb = FakeWandb()
    tracker.logger = {"wandb": fake_wandb}
    tracker._finished = False

    tracker.finish()
    tracker.finish()

    assert tracker._finished is True
    assert fake_wandb.finish_calls == 1
