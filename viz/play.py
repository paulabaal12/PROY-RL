"""Reproductor de replays de HellKitchen: uno o varios agentes lado a lado.

Uso:
    python play_sim.py --rate 1 \
        --agent results/replays/sample_edf.jsonl "EDF heuristic" \
        --agent results/replays/sample_fifo.jsonl "Naive FIFO Baseline"

Controles: Espacio pausa, +/- velocidad, flechas paso a paso (en pausa),
R reinicia, Esc sale. Con --export out.mp4 / out.gif se graba sin ventana.
"""

import argparse
import math
import os
import sys

import numpy as np
import pygame

from viz import sprites
from viz.hud import draw_hud
from viz.replay import Replay, ReplayError, compatibility_warnings, load_replay
from viz.scene import draw_scene
from viz.style import PALETTE, SHIRTS, SUPERSAMPLE
from viz.timeline import build_timeline

MIN_RATE, MAX_RATE, RATE_STEP = 0.25, 20.0, 1.5


def pane_rects(n: int, size: tuple[int, int]) -> list[pygame.Rect]:
    """De 1 a 3 paneles van en una fila, sin celdas vacías; 4 van en una cuadrícula de 2 x 2."""
    w, h = size
    if n in (1, 2, 3):
        edges = [w * i // n for i in range(n + 1)]
        return [pygame.Rect(edges[i], 0, edges[i + 1] - edges[i], h) for i in range(n)]
    if n == 4:
        return [pygame.Rect(0, 0, w // 2, h // 2), pygame.Rect(w // 2, 0, w - w // 2, h // 2),
                pygame.Rect(0, h // 2, w // 2, h - h // 2), pygame.Rect(w // 2, h // 2, w - w // 2, h - h // 2)]
    raise ValueError(f"Se pueden mostrar de 1 a 4 agentes, no {n}")


class Player:
    def __init__(self, replays: list[Replay], labels: list[str], size=(1280, 720), rate=1.0,
                 start=0, end=None):
        self.replays = replays
        self.labels = labels
        self.size = size
        self.rate = rate
        self.timelines = [build_timeline(r) for r in replays]
        last = max(len(r.steps) for r in replays) - 1
        self.start = max(0, min(int(start), last))
        self.end = last if end is None else max(self.start, min(int(end), last))
        self.rects = pane_rects(len(replays), size)
        self._buffers: dict[int, pygame.Surface] = {}

    def _pane(self, idx: int, pos: float) -> pygame.Surface:
        """Dibuja el panel idx con supermuestreo y lo devuelve al tamaño final."""
        rect = self.rects[idx]
        big = self._buffers.get(idx)
        if big is None:
            big = self._buffers[idx] = pygame.Surface((rect.width * SUPERSAMPLE, rect.height * SUPERSAMPLE))
        snaps, scripts = self.timelines[idx]
        last = len(scripts) - 1
        base = math.floor(pos)
        j, frac = base + 1, pos - base
        if j > last:
            j, frac = last, 1.0
        j = max(j, 1)
        replay = self.replays[idx]
        local = big.get_rect()
        shirt = SHIRTS[idx % len(SHIRTS)]
        draw_scene(big, local, snaps[j - 1], snaps[j], scripts[j], frac, shirt)
        draw_hud(big, local, snaps[j - 1], snaps[j], scripts[j], frac, title=self.labels[idx],
                 shirt=shirt, patience=replay.patience, horizon=replay.horizon)
        return pygame.transform.smoothscale(big, rect.size)

    def render(self, surf: pygame.Surface, pos: float) -> None:
        for idx, rect in enumerate(self.rects):
            surf.blit(self._pane(idx, pos), rect)
        for rect in self.rects[1:]:
            if rect.x > 0:
                pygame.draw.line(surf, PALETTE["outline"], rect.topleft, rect.bottomleft, 2)
            if rect.y > 0:
                pygame.draw.line(surf, PALETTE["outline"], rect.topleft, rect.topright, 2)


def _badge(surf, msg, center):
    r = sprites.text(surf, msg, center, 18, PALETTE["hud_fg"], alpha=0).inflate(24, 10)
    pygame.draw.rect(surf, PALETTE["hud_bg"], r, border_radius=r.height // 2)
    pygame.draw.rect(surf, PALETTE["outline"], r, 2, border_radius=r.height // 2)
    sprites.text(surf, msg, r.center, 18, PALETTE["hud_fg"])


def run_window(player: Player) -> None:
    screen = pygame.display.set_mode(player.size)
    pygame.display.set_caption("He(RL)l Kitchen - replay")
    clock = pygame.time.Clock()
    pos, playing, notice, notice_until = float(player.start), True, "", 0
    while True:
        dt = clock.tick(60) / 1000
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_SPACE:
                playing = not playing
                if playing and pos >= player.end:
                    pos = float(player.start)
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                player.rate = min(MAX_RATE, player.rate * RATE_STEP)
                notice, notice_until = f"{player.rate:.2g} steps/s", now + 1500
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                player.rate = max(MIN_RATE, player.rate / RATE_STEP)
                notice, notice_until = f"{player.rate:.2g} steps/s", now + 1500
            elif event.key == pygame.K_RIGHT and not playing:
                pos = min(float(player.end), math.floor(pos) + 1.0)
            elif event.key == pygame.K_LEFT and not playing:
                pos = max(float(player.start), math.ceil(pos) - 1.0)
            elif event.key == pygame.K_r:
                pos = float(player.start)
        if playing:
            pos += dt * player.rate
            if pos >= player.end:
                pos, playing = float(player.end), False
        player.render(screen, pos)
        if not playing:
            _badge(screen, "paused  ·  space play  ·  ←/→ step  ·  +/− speed", (player.size[0] // 2, player.size[1] - 24))
        if now < notice_until:
            _badge(screen, notice, (player.size[0] // 2, 26))
        pygame.display.flip()


def export(player: Player, path: str, fps: int | None = None, scale: float | None = None) -> None:
    is_gif = path.lower().endswith(".gif")
    fps = fps or (15 if is_gif else 30)
    scale = scale or (0.5 if is_gif else 1.0)
    out_size = (int(player.size[0] * scale) // 2 * 2, int(player.size[1] * scale) // 2 * 2)
    n_frames = max(2, int((player.end - player.start) / player.rate * fps) + 1)
    if is_gif and n_frames > 1500:
        print(f"Aviso: {n_frames} cuadros en un GIF es mucho; usa --from/--to o exporta a .mp4")
    surf = pygame.Surface(player.size)

    def frames():
        for f in range(n_frames):
            pos = min(player.end, player.start + f * player.rate / fps)
            player.render(surf, pos)
            img = surf if scale == 1.0 else pygame.transform.smoothscale(surf, out_size)
            yield np.transpose(pygame.surfarray.array3d(img), (1, 0, 2))

    if is_gif:
        from PIL import Image
        imgs = [Image.fromarray(a).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for a in frames()]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0)
    else:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
            for frame in frames():
                writer.append_data(frame)
    print(f"Exportado {n_frames} cuadros a {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce replays de HellKitchen.")
    parser.add_argument("--agent", nargs=2, action="append", required=True, metavar=("REPLAY", "LABEL"),
                        help="archivo .jsonl y nombre a mostrar; se puede repetir hasta 4 veces")
    parser.add_argument("--rate", type=float, default=1.0, help="pasos del ambiente por segundo")
    parser.add_argument("--from", dest="start", type=int, default=0, help="primer paso")
    parser.add_argument("--to", dest="end", type=int, default=None, help="último paso")
    parser.add_argument("--export", default=None, help="out.mp4 o out.gif: graba sin abrir ventana")
    parser.add_argument("--size", default="1280x720", help="ancho x alto, p. ej. 1280x720")
    args = parser.parse_args(argv)

    if not 1 <= len(args.agent) <= 4:
        parser.error("se pueden mostrar de 1 a 4 agentes")
    try:
        size = tuple(int(v) for v in args.size.lower().split("x"))
        assert len(size) == 2 and min(size) > 0
    except (ValueError, AssertionError):
        parser.error(f"--size debe ser ANCHOxALTO, no {args.size!r}")
    if args.rate <= 0:
        parser.error("--rate debe ser positivo")
    try:
        replays = [load_replay(path) for path, _ in args.agent]
    except ReplayError as e:
        parser.error(str(e))
    for warning in compatibility_warnings(replays):
        print(f"Aviso: {warning}", file=sys.stderr)

    if args.export:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    player = Player(replays, [label for _, label in args.agent], size=size, rate=args.rate,
                    start=args.start, end=args.end)
    if args.export:
        export(player, args.export)
    else:
        run_window(player)
    pygame.quit()


if __name__ == "__main__":
    main()
