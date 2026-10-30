"""Tests de la proyección isométrica y los sprites (viz/iso.py, viz/sprites.py)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from viz import sprites  # noqa: E402
from viz.iso import STAND_TILES, STATION_TILES, Iso  # noqa: E402
from viz.style import PALETTE, SHIRTS  # noqa: E402


def test_iso_projection_matches_formula():
    iso = Iso((300, 78), 50, 25)
    assert iso.to_screen(1, 0, 0) == (325, 90.5)
    assert iso.to_screen(0, 1, 10) == (275, 80.5)


def test_every_station_has_a_standing_spot():
    assert set(STATION_TILES) - {"counter"} <= set(STAND_TILES)
    assert "center" in STAND_TILES


def test_particles_are_deterministic_and_fade():
    a = sprites.particles(7, 8, 0.5, (100, 100), 20, 30)
    assert a == sprites.particles(7, 8, 0.5, (100, 100), 20, 30)
    assert a != sprites.particles(8, 8, 0.5, (100, 100), 20, 30)
    early = sum(p[2] for p in sprites.particles(7, 8, 0.2, (100, 100), 20, 30))
    late = sum(p[2] for p in sprites.particles(7, 8, 1.0, (100, 100), 20, 30))
    assert late < early


def changed(surf: pygame.Surface) -> bool:
    bg = pygame.Color(*PALETTE["bg"])
    w, h = surf.get_size()
    return any(surf.get_at((x, y)) != bg for x in range(0, w, 4) for y in range(0, h, 4))


def test_every_sprite_draws_without_error():
    pygame.init()
    iso = Iso((200, 60), 50, 25)
    draws = [
        lambda s: sprites.draw_block(s, iso, 1, 1, 18),
        lambda s: sprites.draw_chef(s, (200, 200), SHIRTS[0], lean=8, wiggle=0.5, face="worried"),
        lambda s: sprites.draw_chef(s, (200, 200), SHIRTS[1], face="ponder"),
        lambda s: sprites.draw_blob(s, (200, 150), (255, 210, 63), 8),
        lambda s: sprites.draw_carrier(s, (200, 150), "board", (108, 193, 90)),
        lambda s: sprites.draw_carrier(s, (200, 150), "plate", (43, 36, 34)),
        lambda s: sprites.draw_burner(s, iso, 2, available=True, lid=0.0, flame=1.0, steam=1.0,
                                      revealed_burnt=False, t_anim=0.3),
        lambda s: sprites.draw_burner(s, iso, 2, available=True, lid=1.0, flame=1.0, steam=0.0,
                                      revealed_burnt=True, t_anim=0.3),
        lambda s: sprites.draw_burner(s, iso, 2, available=False, lid=0.0, flame=0.0, steam=0.0,
                                      revealed_burnt=False, t_anim=0.0),
        lambda s: sprites.draw_lid(s, (200, 150), 1.0, 30),
        lambda s: sprites.draw_trash(s, iso, open_amount=0.7, bounce=0.2),
    ] + [lambda s, n=name: sprites.draw_icon(s, n, (200, 150), 24) for name in sprites.ICONS]
    for draw in draws:
        surf = pygame.Surface((400, 300))
        surf.fill(PALETTE["bg"])
        draw(surf)
        assert changed(surf)
