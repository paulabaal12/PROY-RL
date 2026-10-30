"""Línea de tiempo del visualizador: qué hay en la cocina en cada paso y qué
animación lleva de un paso al siguiente. Es lógica pura, sin dibujar nada.

`snapshots[i]` es la cocina después del paso i del replay. `scripts[i]` describe
la transición de `snapshots[i - 1]` a `snapshots[i]`: por dónde camina el chef,
qué lleva en las manos, qué animación hace y qué destellos se ven.
"""

from dataclasses import dataclass

from envs.kitchen_config import classify
from viz.replay import Replay, Step

SERVED = ("served_ok", "served_late", "served_raw", "served_burnt")
SERVE_ANIM = {"served_ok": "serve_ok", "served_late": "serve_late",
              "served_raw": "serve_bad", "served_burnt": "serve_bad"}


@dataclass(frozen=True)
class OrderView:
    id: int
    dish: str
    age: int
    due: int
    prep_done: int
    prep_required: int
    burner: int | None     # estación donde se cocina, o None si está en la cola
    in_window: bool        # el agente la ve (primeras N de la cola, o en una estación)


@dataclass(frozen=True)
class BurnerView:
    available: bool
    order_id: int | None
    dish: str | None
    revealed_burnt: bool   # una prueba dijo "quemado": la tapa queda abierta


@dataclass(frozen=True)
class Snapshot:
    t: int
    orders: list[OrderView]
    burners: list[BurnerView]
    score: float
    chef_at: str


@dataclass(frozen=True)
class StepScript:
    t: int
    path: list[str]            # puntos por los que camina el chef, desde donde estaba
    carry: str | None          # "board" o "plate", desde el segundo punto del camino
    carry_color: str | None    # plato ("stirfry", "egg", "steak") o "burnt"
    anim: str
    burner: int | None
    dish: str | None
    focus_order: int | None    # orden en la que trabaja el chef
    flash: str | None          # "orange", "red" o None
    reward: float
    expired: list[int]
    expired_burners: list[int]
    arrived: list[int]
    serve_reward: float | None = None   # solo la entrega, sin las expiraciones del mismo paso


def _burner(k: int) -> str:
    return f"burner{k}"


def _collapse(path: list[str]) -> list[str]:
    out = [path[0]]
    for p in path[1:]:
        if p != out[-1]:
            out.append(p)
    return out


def _snapshot(step: Step, window: int, score: float, chef_at: str,
              revealed: list[bool]) -> Snapshot:
    info = step.info
    n_stations = info["kitchen"]["n_stations"]
    orders = []
    for pos, o in enumerate(info["queue"]):
        orders.append(OrderView(o["id"], o["dish"], o["age"], o["due"], o["prep_done"],
                                o["prep_required"], None, pos < window))
    burners = []
    for k, o in enumerate(info["stations"]):
        if o is not None:
            orders.append(OrderView(o["id"], o["dish"], o["age"], o["due"], o["prep_done"],
                                    o["prep_required"], k, True))
        burners.append(BurnerView(k < n_stations, o["id"] if o else None,
                                  o["dish"] if o else None, revealed[k] and o is not None))
    orders.sort(key=lambda o: o.id)
    return Snapshot(step.t, orders, burners, score, chef_at)


def _script(step: Step, prev: Snapshot, expire_penalty: float) -> StepScript:
    events = step.info["events"]
    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], e)
    here = prev.chef_at
    path, carry, carry_color, burner, dish, focus = [here], None, None, None, None, None

    if "invalid" in by_type:
        anim = "invalid"
    elif step.kind == "wait":
        anim, path = "wait", [here, "center"]
    elif step.kind == "prep":
        e = by_type["prepped"]
        anim, path, dish, focus = "prep", [here, "prep"], e["order"]["dish"], e["order"]["id"]
    elif step.kind == "place":
        e = by_type["placed"]
        burner, dish, focus = e["station"], e["order"]["dish"], e["order"]["id"]
        anim, path, carry, carry_color = "place", [here, "prep", _burner(burner)], "board", dish
    elif step.kind == "taste":
        e = by_type["tasted"]
        burner = e["station"]
        anim, path = f"taste_{classify(e['result'])}", [here, _burner(burner)]
        dish, focus = prev.burners[burner].dish, prev.burners[burner].order_id
    else:  # serve o discard
        e = next(e for e in events if e["type"] in SERVED + ("discarded",))
        burner, dish, focus = e["station"], e["order"]["dish"], e["order"]["id"]
        target = "serve" if step.kind == "serve" else "trash"
        anim = SERVE_ANIM.get(e["type"], "discard")
        path, carry = [here, _burner(burner), target], "plate"
        carry_color = "burnt" if classify(e["cook"]) == "burnt" else dish

    expired = [e["order"]["id"] for e in events if e["type"] == "expired"]
    if expired or "served_raw" in by_type or "served_burnt" in by_type:
        flash = "red"
    elif "served_late" in by_type:
        flash = "orange"
    else:
        flash = None
    return StepScript(
        t=step.t, path=_collapse(path), carry=carry, carry_color=carry_color, anim=anim,
        burner=burner, dish=dish, focus_order=focus, flash=flash, reward=step.reward,
        expired=expired,
        expired_burners=[k for k, b in enumerate(prev.burners) if b.order_id in expired],
        arrived=[e["order"]["id"] for e in events if e["type"] == "arrived"],
        serve_reward=step.reward + expire_penalty * len(expired) if anim.startswith("serve") else None,
    )


def build_timeline(replay: Replay) -> tuple[list[Snapshot], list[StepScript]]:
    window = replay.window
    expire_penalty = replay.header["env_params"].get("expire_penalty", 8.0)
    revealed = [False] * len(replay.steps[0].info["stations"])
    snapshots = [_snapshot(replay.steps[0], window, 0.0, "center", revealed)]
    scripts = [StepScript(0, ["center"], None, None, "idle", None, None, None, None, 0.0,
                          [], [], [])]
    score = 0.0
    for step in replay.steps[1:]:
        prev = snapshots[-1]
        script = _script(step, prev, expire_penalty)
        score += step.reward
        # La tapa se abre si la prueba dice "quemado" y se cierra cuando la estación cambia de orden
        for k, o in enumerate(step.info["stations"]):
            if o is None or o["id"] != prev.burners[k].order_id:
                revealed[k] = False
        if script.anim == "taste_burnt":
            revealed[script.burner] = True
        snapshots.append(_snapshot(step, window, score, script.path[-1], revealed))
        scripts.append(script)
    return snapshots, scripts
