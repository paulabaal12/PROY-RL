"""HUD de un panel: título, columna de órdenes, reloj y recompensa acumulada."""

import math

import pygame

from viz import sprites
from viz.iso import design_frame, layout_for
from viz.scene import Phase, ease
from viz.style import DISH_NAMES, PALETTE
from viz.timeline import OrderView, Snapshot, StepScript

CARD_W, CARD_H, CARD_GAP = 124, 36, 4


def order_column_layout(n_orders: int, height: float, columns: int = 1) -> tuple[int, int]:
    """(tarjetas que se muestran, órdenes escondidas en el chip "+k más")."""
    fit = max(1, int(height // (CARD_H + CARD_GAP))) * columns
    if n_orders <= fit:
        return n_orders, 0
    return fit - 1, n_orders - fit + 1


def visible_orders(ids: list[int], focus: int | None, height: float,
                   columns: int = 1) -> tuple[list[int], int]:
    """Órdenes con tarjeta (en orden) y cuántas van al chip. La orden en la que
    trabaja el chef siempre se muestra: si no cabe, toma el último lugar."""
    shown, hidden = order_column_layout(len(ids), height, columns)
    out = ids[:shown]
    if focus in ids and focus not in out and out:
        out[-1] = focus
    return out, hidden


def _shadow_box(surf, rect, k, fill=None, radius=10, border=None, border_w=1.5):
    shadow = rect.move(0, round(2 * k))
    pygame.draw.rect(surf, (0, 0, 0, 30) if surf.get_flags() & pygame.SRCALPHA else (226, 210, 190),
                     shadow, border_radius=int(radius * k))
    pygame.draw.rect(surf, fill or PALETTE["hud_bg"], rect, border_radius=int(radius * k))
    if border:
        pygame.draw.rect(surf, border, rect, max(1, round(border_w * k)), border_radius=int(radius * k))


def _card(order: OrderView, k: float, patience: int, focused: bool, shirt, mark: str | None):
    """Tarjeta de una orden en su propia capa (para poder atenuarla)."""
    w, h = round(CARD_W * k), round(CARD_H * k)
    pad = round(4 * k)
    img = sprites.layer((w + 2 * pad, h + 2 * pad))
    box = pygame.Rect(pad, pad, w, h)
    _shadow_box(img, box, k, border=shirt if focused else None, border_w=2.6 if focused else 1.5)
    ol, fg = PALETTE["outline"], PALETTE["hud_fg"]

    # Fila 1: plato, nombre y número
    sprites.draw_food(img, (box.x + 11 * k, box.y + 8 * k), order.dish, 1.15, 7.5 * k, k * 0.7)
    sprites.text(img, DISH_NAMES[order.dish], (box.x + 21 * k, box.y + 8 * k), int(10.5 * k), fg, anchor="midleft")
    sprites.text(img, f"#{order.id}", (box.right - 7 * k, box.y + 8 * k), int(9.5 * k), (150, 120, 100),
                 anchor="midright")

    # Fila 2: barra de tiempo hasta la entrega y luego zona de paciencia rayada
    span = order.due + patience
    bar = pygame.Rect(box.x + 7 * k, box.y + 15 * k, box.width - 14 * k, 6 * k)
    due_x = bar.x + bar.width * order.due / span
    pygame.draw.rect(img, (236, 226, 214), bar, border_radius=int(3 * k))
    late_zone = pygame.Rect(due_x, bar.y, bar.right - due_x, bar.height)
    stripes = sprites.layer(late_zone.size)
    for sx in range(-late_zone.height, late_zone.width, max(3, int(5 * k))):
        pygame.draw.line(stripes, (*PALETTE["bar_late"], 90), (sx, late_zone.height), (sx + late_zone.height, 0),
                         max(1, int(1.6 * k)))
    img.blit(stripes, late_zone.topleft)
    fill_w = bar.width * min(order.age, order.due) / span
    color = PALETTE["bar_ok"] if order.age <= 0.75 * order.due else PALETTE["bar_warn"]
    if fill_w > 0:
        pygame.draw.rect(img, color, pygame.Rect(bar.x, bar.y, fill_w, bar.height),
                         border_top_left_radius=int(3 * k), border_bottom_left_radius=int(3 * k))
    if order.age > order.due:
        over = bar.width * min(order.age - order.due, patience) / span
        pygame.draw.rect(img, PALETTE["bar_late"], pygame.Rect(due_x, bar.y, over, bar.height))
    pygame.draw.rect(img, ol, bar, max(1, round(k)), border_radius=int(3 * k))
    pygame.draw.line(img, ol, (due_x, bar.y - 1.5 * k), (due_x, bar.bottom + 1.5 * k), max(1, round(1.6 * k)))

    # Fila 3: dónde está y cuánto falta
    small = int(8.5 * k)
    y3 = box.y + 28.5 * k
    if order.burner is not None:
        fx, fy = box.x + 10 * k, y3
        pygame.draw.polygon(img, PALETTE["flame"], [(fx - 3 * k, fy + 3 * k), (fx, fy - 4 * k), (fx + 3 * k, fy + 3 * k)])
        sprites.text(img, f"burner {order.burner + 1}", (box.x + 16 * k, y3), small, fg, anchor="midleft")
    else:
        sprites.text(img, f"prep {order.prep_done}/{order.prep_required}", (box.x + 7 * k, y3), small, fg,
                     anchor="midleft")
    left = order.due - order.age
    status = f"due {left}" if left >= 0 else f"late {-left} · exp {order.due + patience - order.age}"
    sprites.text(img, status, (box.right - 7 * k, y3), small,
                 PALETTE["bar_late"] if left < 0 else fg, anchor="midright")

    if mark == "expired":
        pygame.draw.line(img, PALETTE["money_red"], (box.x + 4 * k, box.centery + 6 * k),
                         (box.right - 4 * k, box.centery - 6 * k), max(2, round(3 * k)))
        pygame.draw.line(img, PALETTE["money_red"], (box.x + 4 * k, box.centery - 6 * k),
                         (box.right - 4 * k, box.centery + 6 * k), max(2, round(3 * k)))
    elif mark == "served":
        sprites.draw_icon(img, "check", (box.right - 14 * k, box.centery), 18 * k)
    return img, pad


def draw_hud(surf: pygame.Surface, rect: pygame.Rect, prev: Snapshot, cur: Snapshot,
             script: StepScript, frac: float, *, title: str, shirt, patience: int, horizon: int) -> None:
    layout = layout_for(rect)
    k, x0, y0 = design_frame(rect, layout.design)
    width, height = layout.design
    ph = Phase(frac)
    s = ease(ph.settle)
    fg = PALETTE["hud_fg"]

    def D(x, y):
        return (x0 + x * k, y0 + y * k)

    # Título con la franja del color del chef
    label = sprites.font(int(13 * k)).render(title, True, fg)
    tag = pygame.Rect(0, 0, label.get_width() + 22 * k, 22 * k)
    tag.topleft = D(10, 8)
    _shadow_box(surf, tag, k)
    pygame.draw.rect(surf, shirt, pygame.Rect(tag.x, tag.y, 7 * k, tag.height),
                     border_top_left_radius=int(10 * k), border_bottom_left_radius=int(10 * k))
    surf.blit(label, label.get_rect(midleft=(tag.x + 13 * k, tag.centery)))

    _draw_orders(surf, k, D, layout, prev, cur, script, s, shirt, patience)
    _draw_legend(surf, k, D(width / 2, height - 19))

    # Reloj
    t_shown = cur.t if s > 0 else prev.t
    clock = sprites.font(int(11 * k)).render(f"t = {t_shown} / {horizon}", True, fg)
    box = pygame.Rect(0, 0, clock.get_width() + 30 * k, 22 * k)
    box.bottomleft = D(10, height - 8)
    _shadow_box(surf, box, k)
    sprites.draw_icon(surf, "clock", (box.x + 12 * k, box.centery), 13 * k)
    surf.blit(clock, clock.get_rect(midleft=(box.x + 22 * k, box.centery)))

    # Recompensa acumulada, con un pequeño salto cuando cambia
    value = prev.score + (cur.score - prev.score) * s
    bump = 1 + 0.25 * math.sin(math.pi * s) if cur.score != prev.score else 1.0
    color = fg
    if cur.score != prev.score and 0 < s < 1:
        color = PALETTE["money_green"] if cur.score > prev.score else PALETTE["money_red"]
    score = sprites.font(int(15 * k * bump)).render(f"{value:+.1f}", True, color)
    box = pygame.Rect(0, 0, score.get_width() + 20 * k, 24 * k)
    box.bottomright = D(width - 10, height - 8)
    _shadow_box(surf, box, k)
    surf.blit(score, score.get_rect(center=box.center))


def _draw_legend(surf, k, center):
    """Explica los dos marcadores de la barra de cocción."""
    cx, cy = center
    ol, fg = PALETTE["outline"], PALETTE["hud_fg"]
    size = int(8.5 * k)
    pygame.draw.polygon(surf, ol, [(cx - 44 * k, cy + 3 * k), (cx - 47.5 * k, cy - 3 * k), (cx - 40.5 * k, cy - 3 * k)])
    sprites.text(surf, "real", (cx - 37 * k, cy), size, fg, anchor="midleft")
    triangle = [(cx - 6 * k, cy - 3 * k), (cx - 9.5 * k, cy + 3 * k), (cx - 2.5 * k, cy + 3 * k)]
    pygame.draw.polygon(surf, PALETTE["hud_bg"], triangle)
    pygame.draw.polygon(surf, ol, triangle, max(1, round(k)))
    sprites.text(surf, "agent sees", (cx + 1 * k, cy), size, fg, anchor="midleft")


def _draw_orders(surf, k, D, layout, prev, cur, script, s, shirt, patience):
    pitch = CARD_H + CARD_GAP
    columns = layout.columns
    height = layout.cards_bottom - layout.cards_top

    def spot(index: float) -> tuple[float, float]:
        """Posición de la tarjeta número `index` en la grilla, en unidades del lienzo."""
        whole = round(index)
        return layout.cards_x + (whole % columns) * layout.column_step, layout.cards_top + (whole // columns) * pitch

    prev_ids, _ = visible_orders([o.id for o in prev.orders], script.focus_order, height, columns)
    cur_ids, _ = visible_orders([o.id for o in cur.orders], script.focus_order, height, columns)
    shown, hidden = order_column_layout(len(cur.orders if s > 0 else prev.orders), height, columns)
    prev_idx = {oid: i for i, oid in enumerate(prev_ids)}
    cur_idx = {oid: i for i, oid in enumerate(cur_ids)}
    views = {o.id: o for o in prev.orders}
    if s > 0:
        views.update({o.id: o for o in cur.orders})
    served = script.focus_order if script.anim.startswith("serve") else None

    for oid in sorted(set(prev_idx) | set(cur_idx)):
        yp, yc = prev_idx.get(oid), cur_idx.get(oid)
        alpha, dx, mark = 255, 0.0, None
        if yp is not None and yc is not None:
            ax, ay = spot(yp)
            bx, by = spot(yc)
            card_x, card_y = ax + (bx - ax) * s, ay + (by - ay) * s
            y = yc if s >= 0.5 else yp
        elif yp is not None:                 # sale de la lista: servida o expirada
            y = yp
            card_x, card_y = spot(yp)
            alpha = 255 * (1 - s)
            if oid in script.expired:
                mark = "expired"
            elif oid == served:
                mark = "served"
            if s <= 0 and mark is None:
                alpha = 255
        else:                                # llegó en este paso
            y, alpha, dx = yc, 255 * s, -30 * (1 - s)
            card_x, card_y = spot(yc)
        if round(y) >= shown or alpha <= 0:
            continue
        order = views[oid]
        focused = oid == script.focus_order
        img, pad = _card(order, k, patience, focused, shirt, mark if s > 0 else None)
        if not order.in_window:
            alpha *= 0.45
        img.set_alpha(int(alpha))
        x, top = D(card_x + dx, card_y)
        surf.blit(img, (x - pad, top - pad))
        if focused:
            tri = [(x - 13 * k, top + 6 * k), (x - 4 * k, top + 11 * k), (x - 13 * k, top + 16 * k)]
            sprites.poly(surf, shirt, tri, k, ow=1.2)

    if hidden:
        x, top = D(*spot(shown))
        chip = pygame.Rect(x, top, 70 * k, 18 * k)
        _shadow_box(surf, chip, k)
        sprites.text(surf, f"+{hidden} more", chip.center, int(9.5 * k), PALETTE["hud_fg"])
