"""Ambiente He(RL)l Kitchen v2: cocina estocástica y parcialmente observable (POMDP).

La especificación completa (platos, cola y ventana visible, acciones, dinámica,
recompensas y observaciones, con la justificación de cada decisión) está en
docs/env_design.md. Las referencias "(sección X)" de los comentarios apuntan a
ese documento.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.kitchen_config import (
    DISH_DUE_RANGE, DISH_PREP, DISH_RATE_RANGE, DISHES, KitchenParams, classify,
    sample_kitchen,
)

# --- Acciones (sección 3.4) -------------------------------------------------
OP_TASTE, OP_SERVE, OP_DISCARD = range(3)
OP_NAMES = ("taste", "serve", "discard")


class ActionLayout:
    """Traduce entre acciones con nombre y el ID de `Discrete(1 + 2N + 3·K_max)`.

    0 = wait; luego prep_i y place_i para cada lugar i de la ventana; luego
    (taste, serve, discard) para cada estación k.
    """

    WAIT = 0

    def __init__(self, window: int, max_stations: int):
        self.window = window
        self.max_stations = max_stations
        self.n = 1 + 2 * window + 3 * max_stations

    def prep(self, i: int) -> int:
        return 1 + i

    def place(self, i: int) -> int:
        return 1 + self.window + i

    def station(self, k: int, op: int) -> int:
        return 1 + 2 * self.window + 3 * k + op

    def decode(self, action: int) -> tuple[str, int | None]:
        """Devuelve (tipo, índice): el lugar de la ventana o la estación."""
        if action == self.WAIT:
            return "wait", None
        if action <= self.window:
            return "prep", action - 1
        if action <= 2 * self.window:
            return "place", action - 1 - self.window
        k, op = divmod(action - 1 - 2 * self.window, 3)
        return OP_NAMES[op], k


# --- Layout de la observación (sección 3.6) ---------------------------------
# Por estación: 12 características, en este orden.
ST_AVAILABLE, ST_OCCUPIED, ST_DISH = 0, 1, 2          # ST_DISH ocupa 3 lugares (one-hot)
ST_VISUAL, ST_HAS_TASTED, ST_LAST_TASTE, ST_SINCE_TASTE, ST_COOK_TIME, \
    ST_TIME_TO_DUE, ST_PATIENCE = range(5, 12)
STATION_FEATURES = 12
# Por lugar de la ventana: 7 características.
SL_PRESENT, SL_DISH = 0, 1                            # SL_DISH ocupa 3 lugares
SL_PREP_PROGRESS, SL_TIME_TO_DUE, SL_PATIENCE = 4, 5, 6
SLOT_FEATURES = 7
GLOBAL_FEATURES = 2                                   # tiempo restante, backlog

# Constantes de normalización
TIME_NORM = 50.0          # horas de entrega
SINCE_TASTE_CAP = 20      # pasos desde la última prueba
COOK_TIME_CAP = 50        # pasos cocinando
COOK_OBS_LOW, COOK_OBS_HIGH = -1.0, 3.0  # rango de lecturas de cocción observadas


@dataclass
class Order:
    order_id: int
    dish: int
    due: int          # hora de entrega, contada desde la llegada
    rate: float       # velocidad de cocción r (oculta), fija para la orden
    age: int = 0      # pasos desde que llegó
    prep_done: int = 0

    @property
    def prep_required(self) -> int:
        return DISH_PREP[self.dish]

    @property
    def is_prepped(self) -> bool:
        return self.prep_done >= self.prep_required

    def to_dict(self) -> dict:
        return {"id": self.order_id, "dish": DISHES[self.dish], "due": self.due,
                "rate": self.rate, "age": self.age, "prep_done": self.prep_done,
                "prep_required": self.prep_required}


@dataclass
class Station:
    order: Order | None = None
    cook: float = 0.0           # nivel real de cocción c (oculto)
    cook_steps: int = 0
    has_tasted: bool = False
    last_taste: float = 0.0
    steps_since_taste: int = 0

    @property
    def occupied(self) -> bool:
        return self.order is not None


class KitchenEnv(gym.Env):
    """Cocina con preparación, estaciones, cola con ventana visible y observación ruidosa."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        window: int = 4,
        max_stations: int = 3,
        kitchen: str | dict = "random",
        horizon: int = 200,
        queue_max: int = 10,
        patience: int = 10,
        sigma_taste: float = 0.02,
        sigma_proc: float = 0.01,
        taste_cost: float = 0.2,
        invalid_penalty: float = 0.1,
        serve_reward: float = 10.0,
        late_frac: float = 0.25,
        expire_penalty: float = 8.0,
        bad_serve_penalty: float = 10.0,
        render_mode: str | None = None,
    ):
        self.window = window
        self.max_stations = max_stations
        self.kitchen = kitchen
        self.horizon = horizon
        self.queue_max = queue_max
        self.patience = patience
        self.sigma_taste = sigma_taste
        self.sigma_proc = sigma_proc
        self.taste_cost = taste_cost
        self.invalid_penalty = invalid_penalty
        self.serve_reward = serve_reward
        self.late_frac = late_frac
        self.expire_penalty = expire_penalty
        self.bad_serve_penalty = bad_serve_penalty
        self.render_mode = render_mode

        self.layout = ActionLayout(window, max_stations)
        self.action_space = spaces.Discrete(self.layout.n)

        obs_size = STATION_FEATURES * max_stations + SLOT_FEATURES * window + GLOBAL_FEATURES
        low = np.zeros(obs_size, dtype=np.float32)
        high = np.ones(obs_size, dtype=np.float32)
        # Las lecturas de cocción pueden pasar de 1 (quemado) o ser negativas por
        # el ruido, y time_to_due es negativo cuando la orden va atrasada.
        due_low = -patience / TIME_NORM
        for k in range(max_stations):
            base = k * STATION_FEATURES
            for feat in (ST_VISUAL, ST_LAST_TASTE):
                low[base + feat], high[base + feat] = COOK_OBS_LOW, COOK_OBS_HIGH
            low[base + ST_TIME_TO_DUE] = due_low
        for i in range(window):
            low[STATION_FEATURES * max_stations + SLOT_FEATURES * i + SL_TIME_TO_DUE] = due_low
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Cuatro generadores independientes derivados de self.np_random (sección 3.7).
        # Separar la cocina y las órdenes del resto garantiza que, con la misma
        # seed, todas las políticas enfrenten la misma cocina y las mismas órdenes
        # (incluida la velocidad r de cada una), hagan lo que hagan.
        child_seeds = self.np_random.integers(0, 2**63 - 1, size=4)
        self.rng_kitchen = np.random.default_rng(child_seeds[0])
        self.rng_orders = np.random.default_rng(child_seeds[1])
        self.rng_cooking = np.random.default_rng(child_seeds[2])
        self.rng_obs = np.random.default_rng(child_seeds[3])

        self.params: KitchenParams = sample_kitchen(self.rng_kitchen, self.horizon, self.kitchen)
        self.t = 0
        self.next_order_id = 0
        self.stations = [Station() for _ in range(self.max_stations)]
        self.queue: list[Order] = []
        self.last_action: int | None = None

        # Estado inicial b0: estaciones vacías y una orden en la cola (sección 3.9)
        first = self._new_order()
        self.queue.append(first)
        self.last_events = [{"type": "arrived", "order": first.to_dict()}]

        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"Acción inválida: {action}"
        reward = 0.0
        events: list[dict] = []
        taste_station = None

        # 1. Acción del agente
        kind, idx = self.layout.decode(action)
        window = self.queue[: self.window]
        valid = True
        if kind == "wait":
            prep_available = any(not o.is_prepped for o in window)
            events.append({"type": "waited", "prep_available": prep_available})
        elif kind == "prep" and idx < len(window) and not window[idx].is_prepped:
            order = window[idx]
            all_busy = self._free_station() is None
            order.prep_done += 1
            events.append({"type": "prepped", "slot": idx, "all_busy": all_busy,
                           "order": order.to_dict()})
        elif kind == "place" and idx < len(window) and window[idx].is_prepped \
                and self._free_station() is not None:
            order = window[idx]
            k = self._free_station()
            self.queue.remove(order)
            self.stations[k] = Station(order=order)
            events.append({"type": "placed", "slot": idx, "station": k, "order": order.to_dict()})
        elif kind in OP_NAMES and idx < self.params.n_stations and self.stations[idx].occupied:
            station = self.stations[idx]
            if kind == "taste":
                # La lectura se toma después de cocinar este paso (paso 2b)
                taste_station = idx
                reward -= self.taste_cost
            elif kind == "serve":
                # Se evalúa con el c de ANTES de cocinar este paso
                reward += self._serve(idx, events)
            else:
                # La orden vuelve a la cola en su lugar por orden de llegada, con la
                # preparación por rehacer y su misma hora de entrega (sección 3.4)
                order = station.order
                events.append({"type": "discarded", "station": idx, "cook": station.cook,
                               "order": order.to_dict()})
                order.prep_done = 0
                self.queue.append(order)
                self.queue.sort(key=lambda o: o.order_id)
                self.stations[idx] = Station()
        else:
            valid = False
        if not valid:
            # Acción inválida: se ejecuta como wait, con una penalización pequeña
            reward -= self.invalid_penalty
            events.append({"type": "invalid", "action": kind, "index": idx})

        # 2. Cocción. Se saca un ruido por estación aunque esté vacía, para que el
        #    consumo de rng_cooking no dependa del estado.
        for st in self.stations:
            noise = float(self.rng_cooking.normal(0.0, self.sigma_proc))
            if st.occupied:
                st.cook = max(0.0, st.cook + st.order.rate + noise)
                st.cook_steps += 1
                st.steps_since_taste += 1

        # 2b. Resultado de probar, con el nivel ya actualizado. El ruido se saca
        #     siempre, se use o no.
        taste_noise = float(self.rng_obs.normal(0.0, self.sigma_taste))
        if taste_station is not None:
            st = self.stations[taste_station]
            st.last_taste = st.cook + taste_noise
            st.has_tasted = True
            st.steps_since_taste = 0
            events.append({"type": "tasted", "station": taste_station,
                           "result": st.last_taste, "cook": st.cook})

        # 3. Tiempo: todas las órdenes envejecen un paso. Las que pasan su hora de
        #    entrega más la paciencia expiran: el cliente se fue (sección 3.3).
        for order in self._active_orders():
            order.age += 1
        expired = [o for o in self.queue if self._is_expired(o)]
        self.queue = [o for o in self.queue if not self._is_expired(o)]
        for k, st in enumerate(self.stations):
            if st.occupied and self._is_expired(st.order):
                expired.append(st.order)
                self.stations[k] = Station()
        for order in expired:
            reward -= self.expire_penalty
            events.append({"type": "expired", "order": order.to_dict()})

        # 4. Llegadas. Los números aleatorios se sacan SIEMPRE, aunque la cola
        #    esté llena, para que la secuencia de rng_orders no dependa del agente.
        arrival_roll = self.rng_orders.random()
        candidate = self._new_order()
        if arrival_roll < self.params.arrival_prob_at(self.t) and len(self.queue) < self.queue_max:
            self.queue.append(candidate)
            events.append({"type": "arrived", "order": candidate.to_dict()})

        self.t += 1
        terminated = False                     # no hay estados terminales (sección 3.9)
        truncated = self.t >= self.horizon
        self.last_events = events
        self.last_action = action

        obs = self._get_obs()
        if self.render_mode == "human":
            print(self.render())
        return obs, reward, terminated, truncated, self._get_info()

    # ---------------------------------------------------------------- helpers
    def _new_order(self) -> Order:
        dish = int(self.rng_orders.integers(len(DISHES)))
        lo, hi = DISH_DUE_RANGE[dish]
        due = int(self.rng_orders.integers(lo, hi + 1))
        rate = float(self.rng_orders.uniform(*DISH_RATE_RANGE[dish]))
        order = Order(order_id=self.next_order_id, dish=dish, due=due, rate=rate)
        self.next_order_id += 1
        return order

    def _serve(self, k: int, events: list[dict]) -> float:
        """Entrega el plato de la estación k y devuelve su recompensa (sección 3.5)."""
        station = self.stations[k]
        order = station.order
        quality = classify(station.cook)
        delay = max(order.age - order.due, 0)
        if quality != "ok":
            reward, kind = -self.bad_serve_penalty, f"served_{quality}"
        elif delay == 0:
            reward, kind = self.serve_reward, "served_ok"
        else:
            # Cae a late_frac apenas pasa la hora de entrega y baja en línea recta
            reward = self.serve_reward * self.late_frac * (1 - (delay - 1) / self.patience)
            kind = "served_late"
        events.append({"type": kind, "station": k, "cook": station.cook, "delay": delay,
                       "order": order.to_dict()})
        self.stations[k] = Station()
        return reward

    def _is_expired(self, order: Order) -> bool:
        return order.age > order.due + self.patience

    def _free_station(self) -> int | None:
        """Estación disponible libre de menor índice, o None."""
        for k in range(self.params.n_stations):
            if not self.stations[k].occupied:
                return k
        return None

    def _active_orders(self) -> list[Order]:
        return self.queue + [st.order for st in self.stations if st.occupied]

    def _time_features(self, order: Order) -> tuple[float, float]:
        """(time_to_due, patience_left) normalizados (sección 3.6)."""
        time_to_due = max((order.due - order.age) / TIME_NORM, -self.patience / TIME_NORM)
        if order.age <= order.due:
            patience_left = 1.0
        else:
            patience_left = max(order.due + self.patience - order.age, 0) / self.patience
        return time_to_due, patience_left

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.last_visual = []
        for k, st in enumerate(self.stations):
            base = k * STATION_FEATURES
            # El ruido visual se sortea en cada paso, aunque la estación esté vacía,
            # para que el consumo de rng_obs no dependa del estado.
            visual = st.cook + self.rng_obs.normal(0.0, self.params.sigma_obs)
            self.last_visual.append(float(visual) if st.occupied else None)
            if k < self.params.n_stations:
                obs[base + ST_AVAILABLE] = 1.0
            if not st.occupied:
                continue
            obs[base + ST_OCCUPIED] = 1.0
            obs[base + ST_DISH + st.order.dish] = 1.0
            obs[base + ST_VISUAL] = np.clip(visual, COOK_OBS_LOW, COOK_OBS_HIGH)
            obs[base + ST_HAS_TASTED] = float(st.has_tasted)
            if st.has_tasted:
                obs[base + ST_LAST_TASTE] = np.clip(st.last_taste, COOK_OBS_LOW, COOK_OBS_HIGH)
                obs[base + ST_SINCE_TASTE] = min(st.steps_since_taste, SINCE_TASTE_CAP) / SINCE_TASTE_CAP
            obs[base + ST_COOK_TIME] = min(st.cook_steps, COOK_TIME_CAP) / COOK_TIME_CAP
            obs[base + ST_TIME_TO_DUE], obs[base + ST_PATIENCE] = self._time_features(st.order)

        wbase = STATION_FEATURES * self.max_stations
        for i, order in enumerate(self.queue[: self.window]):
            base = wbase + i * SLOT_FEATURES
            obs[base + SL_PRESENT] = 1.0
            obs[base + SL_DISH + order.dish] = 1.0
            obs[base + SL_PREP_PROGRESS] = min(order.prep_done / order.prep_required, 1.0)
            obs[base + SL_TIME_TO_DUE], obs[base + SL_PATIENCE] = self._time_features(order)

        backlog = max(len(self.queue) - self.window, 0)
        obs[-2] = (self.horizon - self.t) / self.horizon
        obs[-1] = min(backlog / self.queue_max, 1.0)
        return obs

    def _get_info(self) -> dict:
        """Estado oculto real y eventos del paso. El agente NO debe usarlo para decidir."""
        return {
            "t": self.t,
            "kitchen": self.params.to_dict(),
            "true_cook": [st.cook if st.occupied else None for st in self.stations],
            "true_rate": [st.order.rate if st.occupied else None for st in self.stations],
            "stations": [st.order.to_dict() if st.occupied else None for st in self.stations],
            "queue": [o.to_dict() for o in self.queue],
            "events": self.last_events,
        }

    # ----------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            return None
        rush = " | HORA PICO" if any(s <= self.t < e for s, e in self.params.rushes) else ""
        lines = [f"=== t = {self.t:3d}/{self.horizon}{rush} " + "=" * 40]
        if self.last_action is not None:
            kind, idx = self.layout.decode(self.last_action)
            target = "" if idx is None else (f" lugar {idx}" if kind in ("prep", "place")
                                             else f" estación {idx}")
            lines.append(f"Acción: {kind}{target}")
        for k, st in enumerate(self.stations):
            if k >= self.params.n_stations:
                lines.append(f"[Estación {k}] no existe en esta cocina")
                continue
            if not st.occupied:
                lines.append(f"[Estación {k}] vacía")
                continue
            # Barra del nivel REAL: '.' crudo, '#' bien cocido, '!' quemado.
            filled = int(min(st.cook, 1.6) / 1.6 * 32)
            bar = "".join("." if i < 20 else ("#" if i < 26 else "!") for i in range(filled))
            taste = f"{st.last_taste:.2f} (hace {st.steps_since_taste})" if st.has_tasted else "-"
            o = st.order
            lines.append(
                f"[Estación {k}] {DISHES[o.dish]:<7} orden#{o.order_id} edad {o.age:2d}/entrega {o.due} "
                f"| real c={st.cook:.2f} ({classify(st.cook)}) r={o.rate:.3f} "
                f"| agente ve ~{self.last_visual[k]:.2f} | última prueba: {taste}"
            )
            lines.append(f"             |{bar:<32}|  (crudo < 1.0 <= ok < 1.3 <= quemado)")
        window = ", ".join(
            f"{DISHES[o.dish]}#{o.order_id}(prep {o.prep_done}/{o.prep_required}, "
            f"edad {o.age}/{o.due})" for o in self.queue[: self.window]) or "vacía"
        lines.append(f"Ventana (N={self.window}): {window}")
        backlog = len(self.queue) - self.window
        if backlog > 0:
            lines.append(f"Fuera de la ventana: {backlog} órdenes")
        if self.last_events:
            lines.append("Eventos: " + "; ".join(e["type"] for e in self.last_events))
        return "\n".join(lines)
