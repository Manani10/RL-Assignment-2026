"""The bundled single-file Kaggle agent must import and play, with NO access to src/.

We stub the two framework modules the starter provides (`arcengine`,
`agents.agent`) and drive the bundled MyAgent against the mock game.
"""
import enum
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _install_stubs():
    class GameAction(enum.Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        def set_data(self, data):
            self._data = data

    class GameState(enum.Enum):
        NOT_PLAYED = 0
        NOT_FINISHED = 1
        WIN = 2
        GAME_OVER = 3

    arcengine = types.ModuleType("arcengine")
    arcengine.GameAction, arcengine.GameState, arcengine.FrameData = GameAction, GameState, object
    agents = types.ModuleType("agents")
    agents.__path__ = []
    agent_mod = types.ModuleType("agents.agent")

    class Agent:
        def __init__(self, *a, **k):
            self.game_id = "mock"

    agent_mod.Agent = Agent
    sys.modules.update({"arcengine": arcengine, "agents": agents, "agents.agent": agent_mod})
    return GameAction


def test_bundled_agent_plays(tmp_path):
    _play_bundle(tmp_path, [])


def test_bundled_agent_plays_with_shared_action_space(tmp_path):
    _play_bundle(tmp_path, ["--click-mode", "objects", "--prune-noops"])


def _play_bundle(tmp_path, extra):
    out = tmp_path / "my_agent.py"
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "bundle_agent.py"),
                           "--agent", "random", "--max-actions", "400", "--out", str(out), *extra])
    GameAction = _install_stubs()
    # make sure the bundle does NOT rely on src/ being importable
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if not p.rstrip("/").endswith("src")]
    for m in [m for m in sys.modules if m == "arcrl" or m.startswith("arcrl.")]:
        del sys.modules[m]
    try:
        spec = importlib.util.spec_from_file_location("bundled_my_agent", out)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        agent = mod.MyAgent()

        sys.path[:] = saved_path + [str(ROOT / "src")]
        from arcrl.env.mock import MockAction, MockGame

        game = MockGame(0)
        frame = types.SimpleNamespace(state=GameAction.RESET, frame=[], levels_completed=0,
                                      available_actions=[])
        frame.state = types.SimpleNamespace(name="NOT_PLAYED")
        steps = 0
        while not agent.is_done([], frame) and steps < 500:
            act = agent.choose_action([], frame)
            if act.name == "RESET":
                obs = game.reset()
            else:
                data = getattr(act, "_data", None) if act.name == "ACTION6" else None
                obs = game.step(MockAction[act.name], data=data)
            frame = types.SimpleNamespace(state=types.SimpleNamespace(name=obs.state.name), frame=obs.frame,
                                          levels_completed=obs.levels_completed,
                                          available_actions=obs.available_actions)
            steps += 1
        assert steps >= 100  # played until the action cap / win without crashing
    finally:
        sys.path[:] = saved_path
        for m in ("arcengine", "agents", "agents.agent"):
            sys.modules.pop(m, None)
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_ArcrlFinder"]
