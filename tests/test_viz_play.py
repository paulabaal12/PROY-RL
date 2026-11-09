"""Tests del reproductor (viz/play.py): paneles, rangos, exportación y CLI."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from agents.baselines import FifoHeuristicPolicy  # noqa: E402
from envs.kitchen_env import KitchenEnv  # noqa: E402
from viz.play import Player, export, main, pane_rects  # noqa: E402
from viz.record import record_episode  # noqa: E402
from viz.replay import load_replay  # noqa: E402

pygame.init()
SAMPLE = "results/replays/sample_fifo.jsonl"


def test_pane_rects_layouts():
    one = pane_rects(1, (1280, 720))
    assert one == [pygame.Rect(0, 0, 1280, 720)]
    two = pane_rects(2, (1280, 720))
    assert [r.width for r in two] == [640, 640] and two[1].x == 640
    four = pane_rects(4, (1280, 720))
    assert {(r.x, r.y) for r in four} == {(0, 0), (640, 0), (0, 360), (640, 360)}
    three = pane_rects(3, (1280, 720))
    assert all(r.y == 0 and r.height == 720 for r in three)      # una fila, sin celda vacía
    assert sum(r.width for r in three) == 1280
    with pytest.raises(ValueError):
        pane_rects(5, (1280, 720))


def test_shorter_replay_freezes_on_last_step(tmp_path):
    short = record_episode(KitchenEnv(window=4, kitchen="easy", horizon=30), FifoHeuristicPolicy(4),
                           seed=10000, path=tmp_path / "short.jsonl")
    player = Player([load_replay(short), load_replay(SAMPLE)], ["short", "long"], size=(640, 360))
    assert player.end == 200
    surf = pygame.Surface((640, 360))
    player.render(surf, 100.5)
    player.render(surf, 200.0)


def test_from_to_are_clamped():
    player = Player([load_replay(SAMPLE)], ["x"], start=500)
    assert player.start == 200 and player.end == 200
    player = Player([load_replay(SAMPLE)], ["x"], start=-3, end=9999)
    assert player.start == 0 and player.end == 200


def test_export_gif_writes_file(tmp_path):
    player = Player([load_replay(SAMPLE)], ["x"], size=(640, 360), rate=4, start=0, end=4)
    out = tmp_path / "a.gif"
    export(player, str(out))
    with Image.open(out) as img:
        assert img.n_frames > 1
        assert img.size == (320, 180)


def test_export_mp4_writes_file(tmp_path):
    player = Player([load_replay(SAMPLE)], ["x"], size=(640, 360), rate=8, start=10, end=12)
    out = tmp_path / "a.mp4"
    export(player, str(out))
    assert out.stat().st_size > 1000


def test_main_missing_file_exits_2(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--agent", "nope.jsonl", "X"])
    assert e.value.code == 2
    assert "nope.jsonl" in capsys.readouterr().err


def test_main_odd_agent_args_exits_2():
    with pytest.raises(SystemExit) as e:
        main(["--agent", SAMPLE])
    assert e.value.code == 2


def test_main_exports(tmp_path):
    out = tmp_path / "b.gif"
    main(["--agent", SAMPLE, "FIFO", "--agent", SAMPLE, "FIFO 2", "--rate", "8", "--from", "20",
          "--to", "22", "--size", "640x360", "--export", str(out)])
    assert out.exists()
