"""Tests de dibujo de la escena y el HUD (viz/scene.py, viz/hud.py), sin ventana."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

from viz.hud import draw_hud, order_column_layout  # noqa: E402
from viz.replay import load_replay  # noqa: E402
from viz.scene import draw_scene  # noqa: E402
from viz.style import SHIRTS  # noqa: E402
from viz.timeline import build_timeline  # noqa: E402

pygame.init()


def render(replay, snaps, scripts, i, frac, size=(640, 360)):
    surf = pygame.Surface(size)
    rect = surf.get_rect()
    draw_scene(surf, rect, snaps[i - 1], snaps[i], scripts[i], frac, SHIRTS[0])
    draw_hud(surf, rect, snaps[i - 1], snaps[i], scripts[i], frac, title="Test", shirt=SHIRTS[0],
             patience=replay.patience, horizon=replay.horizon)
    return surf


@pytest.mark.parametrize("name", ["sample_fifo", "sample_fifo_medium", "sample_random"])
def test_render_every_step_of_samples_headless(name):
    replay = load_replay(f"results/replays/{name}.jsonl")
    snaps, scripts = build_timeline(replay)
    for i in range(1, len(snaps)):
        for frac in (0.0, 0.3, 0.6, 0.95):
            render(replay, snaps, scripts, i, frac)


def test_render_is_deterministic():
    replay = load_replay("results/replays/sample_fifo.jsonl")
    snaps, scripts = build_timeline(replay)
    i = next(i for i, s in enumerate(scripts) if s.anim == "serve_ok")
    a = pygame.image.tobytes(render(replay, snaps, scripts, i, 0.7), "RGB")
    b = pygame.image.tobytes(render(replay, snaps, scripts, i, 0.7), "RGB")
    assert a == b


def test_order_column_overflow_collapses():
    shown, hidden = order_column_layout(13, 240)
    assert shown + hidden == 13 and hidden > 0
    assert order_column_layout(3, 240) == (3, 0)
