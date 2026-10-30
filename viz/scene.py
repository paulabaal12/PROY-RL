"""Dibuja la cocina de un panel en un instante: paso i del replay más una fracción.

Cada paso se divide en tres fases (style.PHASES): el chef camina por `script.path`,
hace la acción y luego se asienta el resultado (órdenes expiradas, destellos).
Todo depende solo de (snapshots, script, frac), así que el mismo instante siempre
se dibuja igual.

Vista previa:
    python -m viz.scene --replay results/replays/sample_fifo.jsonl --step 40
"""

import argparse
import math

import pygame

from viz import sprites
from viz.iso import (
    COUNTER_HEIGHT, FLOOR, KITCHEN_ORIGIN, STAND_TILES, STATION_TILES, TILE, Iso, design_frame,
)
from viz.style import DISH_COLORS, PALETTE, PHASES
from viz.timeline import Snapshot, StepScript

TRAVEL_END = PHASES[0]
ACTION_END = PHASES[0] + PHASES[1]
HAND = (14, -17)        # dónde lleva el chef la tabla o el plato, respecto a sus pies


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ease(x: float) -> float:
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


class Phase:
    """Avance dentro de cada fase del paso, cada uno en [0, 1]."""

    def __init__(self, frac: float):
        self.frac = frac
        self.travel = clamp01(frac / TRAVEL_END)
        self.action = clamp01((frac - TRAVEL_END) / PHASES[1])
        self.settle = clamp01((frac - ACTION_END) / PHASES[2])


def make_iso(rect) -> Iso:
    k, x0, y0 = design_frame(rect)
    return Iso((x0 + KITCHEN_ORIGIN[0] * k, y0 + KITCHEN_ORIGIN[1] * k), TILE[0] * k, TILE[1] * k)


def stand_point(iso: Iso, where: str):
    return iso.to_screen(*STAND_TILES[where])


def walk(iso: Iso, path: list[str], u: float, pickup: str | None = None):
    """Posición a lo largo del camino (u en [0, 1]) y si el chef ya pasó por `pickup`."""
    pts = [stand_point(iso, p) for p in path]
    lengths = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(lengths)
    d = ease(u) * total
    reached = True
    if pickup in path:
        reached = d >= sum(lengths[:path.index(pickup)]) - 1e-6
    if not lengths or total == 0:
        return pts[-1], reached
    for i, seg in enumerate(lengths):
        if d <= seg or i == len(lengths) - 1:
            return lerp(pts[i], pts[i + 1], clamp01(d / seg if seg else 1.0)), reached
        d -= seg
    return pts[-1], reached


def _burner_view(k: int, prev: Snapshot, cur: Snapshot, script: StepScript, ph: Phase,
                 picked_up: bool):
    """Estado de la estación k que se muestra en este instante (antes o después del cambio)."""
    before, after = prev.burners[k], cur.burners[k]
    if script.burner == k:
        if script.anim == "place":
            return after if ph.action >= 0.5 else before
        if script.carry == "plate":                      # servir o descartar
            return after if picked_up else before
        if script.anim == "taste_burnt":
            return after if ph.action >= 0.35 else before
    if k in script.expired_burners:
        return after if ph.settle > 0 else before
    return after if ph.frac >= 1.0 else before


def draw_scene(surf: pygame.Surface, rect: pygame.Rect, prev: Snapshot, cur: Snapshot,
               script: StepScript, frac: float, shirt) -> None:
    ph = Phase(frac)
    iso = make_iso(rect)
    k = iso.k
    t_anim = script.t - 1 + frac          # tiempo continuo, para llamas y vapor
    surf.fill(PALETTE["bg"], rect)
    clip = surf.get_clip()
    surf.set_clip(rect)

    pickup = None
    if script.carry == "board":
        pickup = "prep"
    elif script.carry == "plate":
        pickup = f"burner{script.burner}"
    chef_pos, picked_up = walk(iso, script.path, ph.travel, pickup)
    anim, a = script.anim, ph.action

    sprites.draw_floor(surf, iso, FLOOR)
    # Orden del pintor: por x + y
    tiles = [(0, 0, "counter"), (1, 0, "prep"), (0, 1, "counter"), (2, 0, "burner0"),
             (0, 2, "trash"), (3, 0, "burner1"), (0, 3, "counter"), (4, 0, "burner2"), (0, 4, "serve")]
    burner_tops = {}
    window_top = None
    board_top = None
    trash_top = None
    for x, y, kind in tiles:
        if kind == "counter":
            sprites.draw_block(surf, iso, x, y, COUNTER_HEIGHT)
        elif kind == "prep":
            board_top = sprites.draw_block(surf, iso, x, y, COUNTER_HEIGHT)
            bx, by = board_top
            sprites.poly(surf, PALETTE["board"], [(bx - 14 * k, by), (bx, by - 7 * k), (bx + 14 * k, by),
                                                   (bx, by + 7 * k)], k)
            pygame.draw.line(surf, (200, 200, 205), (bx + 3 * k, by - 2 * k), (bx + 10 * k, by - 5.5 * k),
                             max(2, round(2.2 * k)))
        elif kind == "trash":
            opening, bounce = 0.0, 0.0
            if anim == "discard":
                opening = ease(a / 0.25) if a < 0.6 else 1 - ease((a - 0.6) / 0.12)
                bounce = math.sin(clamp01((a - 0.72) / 0.28) * math.pi) * (1 - clamp01((a - 0.72) / 0.28))
            trash_top = sprites.draw_trash(surf, iso, opening, bounce)
        elif kind == "serve":
            window_top = sprites.draw_block(surf, iso, x, y, COUNTER_HEIGHT, top=PALETTE["serve"])
            sprites.ellipse(surf, PALETTE["plate"], window_top, 11 * k, 5 * k, k)
        else:
            b = int(kind[-1])
            view = _burner_view(b, prev, cur, script, ph, picked_up)
            occupied = view.order_id is not None
            lid = 0.0
            flame = 1.0 if occupied else 0.0
            if script.burner == b and anim == "place":
                flame = ease((a - 0.4) / 0.4) if prev.burners[b].order_id is None else 1.0
                lid = 0.6 if a < 0.55 else 0.6 * (1 - ease((a - 0.55) / 0.2))
            elif script.burner == b and anim.startswith("taste") and anim != "taste_burnt":
                lid = 0.35 * math.sin(math.pi * clamp01(a * 1.2))
            elif script.burner == b and anim == "taste_burnt" and a < 0.35:
                lid = 0.35 * math.sin(math.pi * clamp01(a / 0.35) * 0.5)
            if view.revealed_burnt:
                lid = 1.0
            steam = 1.0 if occupied else 0.0
            rim = sprites.draw_burner(surf, iso, b + 2, available=view.available, lid=lid, flame=flame,
                                      steam=steam, revealed_burnt=view.revealed_burnt, t_anim=t_anim)
            burner_tops[b] = rim
            if script.burner == b and anim == "taste_burnt" and a >= 0.35:
                u = clamp01((a - 0.35) / 0.65)
                lid_pos = (rim[0] + 34 * k * u, rim[1] - 30 * k * math.sin(math.pi * u * 0.9) + 20 * k * u * u)
                sprites.draw_lid(surf, lid_pos, k, angle=-200 * u, alpha=int(255 * (1 - u)))
            if b in script.expired_burners and ph.settle > 0:
                for j in range(5):
                    ang = j / 5 * math.tau
                    r = 6 * k + 16 * k * ph.settle
                    sprites.soft_circle(surf, (230, 230, 230), (rim[0] + r * math.cos(ang),
                                                                rim[1] - 6 * k + 0.5 * r * math.sin(ang)),
                                        5 * k * (1 + ph.settle), 220 * (1 - ph.settle))
            if occupied:
                pill = (rim[0], rim[1] - 36 * k)
                label = f"#{view.order_id}"
                r = sprites.text(surf, label, pill, int(9.5 * k), PALETTE["hud_fg"], alpha=0)
                box = r.inflate(10 * k, 4 * k)
                sprites.rrect(surf, PALETTE["hud_bg"], box, box.height / 2, k, ow=1.2)
                pygame.draw.circle(surf, DISH_COLORS[view.dish], (box.left + 1 * k, box.centery), 4 * k)
                pygame.draw.circle(surf, PALETTE["outline"], (box.left + 1 * k, box.centery), 4 * k, max(1, round(k)))
                sprites.text(surf, label, (pill[0] + 2 * k, pill[1]), int(9.5 * k), PALETTE["hud_fg"])

    _draw_chef_and_action(surf, iso, script, ph, chef_pos, picked_up, shirt, burner_tops,
                          window_top, board_top, trash_top, t_anim)
    _draw_flash(surf, rect, script, ph, k)
    surf.set_clip(clip)


def _draw_chef_and_action(surf, iso, script, ph, chef_pos, picked_up, shirt, burner_tops,
                          window_top, board_top, trash_top, t_anim):
    k = iso.k
    anim, a = script.anim, ph.action
    seed = hash((script.t, script.burner or 0, anim))
    lean, wiggle, face, hop = 0.0, 0.0, "normal", 0.0
    head = (chef_pos[0], chef_pos[1] - 80 * k)

    if ph.travel < 1 and len(script.path) > 1:
        hop = 2.5 * abs(math.sin(ph.travel * math.pi * 6))
    if anim == "prep":
        wiggle = a
    elif anim.startswith("taste"):
        lean = 9 * math.sin(math.pi * clamp01(a * 1.2))
        if anim == "taste_raw" and a > 0.2:
            face = "ponder"
        if anim == "taste_burnt" and a > 0.3:
            face = "worried"
    elif anim == "invalid":
        hop = 4 * abs(math.sin(a * math.pi * 2))
        lean = 7 * math.sin(a * math.tau * 2)
    elif anim == "wait":
        hop = 1.2 * abs(math.sin(a * math.pi * 2))

    # Qué lleva en las manos y hasta cuándo
    carrying = False
    if script.carry == "board":
        carrying = picked_up and (ph.travel < 1 or a < 0.15)
    elif script.carry == "plate":
        carrying = picked_up and (ph.travel < 1 or a < 0.12)

    sprites.draw_chef(surf, chef_pos, shirt, lean=lean, wiggle=wiggle, face=face, hop=hop, k=k)
    hand = (chef_pos[0] + HAND[0] * k, chef_pos[1] + HAND[1] * k - hop * k)
    color = DISH_COLORS.get(script.carry_color) if script.carry_color else None
    if carrying and color:
        sprites.draw_carrier(surf, hand, script.carry, color, k)

    # Lo que pasa en la estación durante la acción
    if anim == "prep" and board_top and script.dish:
        for x, y, al in sprites.particles(seed, 9, a, (board_top[0], board_top[1] - 4 * k), 16 * k, 26 * k):
            sprites.soft_circle(surf, DISH_COLORS[script.dish], (x, y), 2.6 * k, al)
    elif anim == "place" and script.burner in burner_tops and 0.12 <= a < 0.55:
        u = clamp01((a - 0.12) / 0.43)
        rim = burner_tops[script.burner]
        pos = lerp(hand, (rim[0], rim[1] - 4 * k), ease(u))
        pos = (pos[0], pos[1] - 18 * k * math.sin(math.pi * u))
        sprites.draw_carrier(surf, pos, "board", color, k) if u < 0.5 else \
            sprites.draw_blob(surf, pos, color, 5.5 * k, k)
    elif anim.startswith("serve") and window_top and color:
        if 0.12 <= a < 0.5:
            u = ease((a - 0.12) / 0.38)
            sprites.draw_carrier(surf, lerp(hand, (window_top[0], window_top[1] - 2 * k), u), "plate", color, k)
        elif a >= 0.5:
            sprites.draw_carrier(surf, (window_top[0], window_top[1] - 2 * k), "plate", color, k)
        _serve_effects(surf, script, ph, window_top, seed, k)
    elif anim == "discard" and trash_top and color and 0.12 <= a < 0.6:
        u = clamp01((a - 0.12) / 0.48)
        pos = lerp(hand, (trash_top[0], trash_top[1] + 6 * k), ease(u))
        pos = (pos[0], pos[1] - 22 * k * math.sin(math.pi * u))
        sprites.draw_blob(surf, pos, color, 5.5 * k, k)

    # Íconos sobre la cabeza
    if anim == "taste_raw" and a > 0.15:
        _icon_stream(surf, "confused", head, seed, clamp01((a - 0.15) / 0.85 + ph.settle * 0.2), k, n=3)
    elif anim == "taste_ok" and a > 0.15:
        u = clamp01((a - 0.15) / 0.85 + ph.settle * 0.2)
        _icon_stream(surf, "check", head, seed, u, k, n=2)
        _icon_stream(surf, "yum", head, seed + 1, u, k, n=2)
    elif anim == "taste_burnt" and a > 0.1:
        pop = ease((a - 0.1) / 0.15)
        alpha = 255 if ph.settle < 0.5 else int(255 * (1 - (ph.settle - 0.5) * 2))
        sprites.draw_icon(surf, "warning", (head[0], head[1] - 4 * k * pop), 22 * k * (0.6 + 0.4 * pop), alpha)
    elif anim == "wait" and ph.travel >= 1:
        sprites.draw_icon(surf, "dots", head, 20 * k, int(255 * min(1, a * 4)))
    elif anim == "invalid":
        sprites.draw_icon(surf, "question", head, 20 * k, int(255 * min(1, a * 4)))


def _icon_stream(surf, name, origin, seed, u, k, n=3):
    for x, y, al in sprites.particles(seed, n, u, origin, 18 * k, 30 * k):
        if al > 0:
            sprites.draw_icon(surf, name, (x, y), 15 * k, al)


def _serve_effects(surf, script, ph, window_top, seed, k):
    u = clamp01((ph.action - 0.45) / 0.55 * 0.8 + ph.settle * 0.2)
    if u <= 0:
        return
    origin = (window_top[0], window_top[1] - 8 * k)
    anim = script.anim
    if anim in ("serve_ok", "serve_late"):
        n = 10 if anim == "serve_ok" else 4
        for x, y, al in sprites.particles(seed, n, u, origin, 22 * k, 46 * k):
            if al > 0:
                sprites.draw_icon(surf, "coin", (x, y), 11 * k, al)
        if anim == "serve_late":
            _icon_stream(surf, "angry", (origin[0] + 6 * k, origin[1] - 10 * k), seed + 3, u, k, n=2)
        label, color = f"+{script.reward:g}", PALETTE["money_green"]
    else:
        for x, y, al in sprites.particles(seed, 4, u, origin, 20 * k, 40 * k):
            if al > 0:
                sprites.draw_icon(surf, "minus_money", (x, y), 13 * k, al)
        _icon_stream(surf, "angry", origin, seed + 5, u, k, n=2)
        _icon_stream(surf, "puke", origin, seed + 6, u, k, n=2)
        label, color = f"{script.reward:g}", PALETTE["money_red"]
    sprites.text(surf, label, (origin[0], origin[1] - 40 * k - 14 * k * u), int(15 * k), color,
                 outline=(255, 255, 255), alpha=int(255 * min(1, (1 - u) * 3)))


def _draw_flash(surf, rect, script, ph, k):
    if not script.flash:
        return
    # Sube al final de la acción, llega al máximo al empezar a asentarse y se desvanece
    if ph.frac < ACTION_END:
        strength = ease((ph.action - 0.6) / 0.4)
    else:
        strength = 1 - ph.settle
    if strength <= 0:
        return
    color = PALETTE["red_flash" if script.flash == "red" else "orange_flash"]
    over = pygame.Surface(rect.size, pygame.SRCALPHA)
    over.fill((*color, int(55 * strength)))
    w = max(4, int(12 * k))
    for i in range(w):
        a = int(170 * strength * (1 - i / w) ** 2)
        pygame.draw.rect(over, (*color, a), over.get_rect().inflate(-2 * i, -2 * i), 1)
    surf.blit(over, rect.topleft)


def _preview():
    from viz.replay import load_replay
    from viz.style import SHIRTS
    from viz.timeline import build_timeline
    from viz.hud import draw_hud

    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default="results/replays/sample_fifo.jsonl")
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--out", default="results/figures/scene_preview.png")
    args = parser.parse_args()
    pygame.init()
    replay = load_replay(args.replay)
    snaps, scripts = build_timeline(replay)
    i = args.step
    w, h = 940, 640
    surf = pygame.Surface((w * 2, h * 2))
    for j, frac in enumerate((0.2, 0.55, 0.75, 0.9)):
        rect = pygame.Rect((j % 2) * w, (j // 2) * h, w, h)
        draw_scene(surf, rect, snaps[i - 1], snaps[i], scripts[i], frac, SHIRTS[0])
        draw_hud(surf, rect, snaps[i - 1], snaps[i], scripts[i], frac, title=replay.header.get("label", "?"),
                 shirt=SHIRTS[0], patience=replay.patience, horizon=replay.horizon)
    pygame.image.save(surf, args.out)
    print(f"{scripts[i].anim} -> {args.out}")


if __name__ == "__main__":
    _preview()
