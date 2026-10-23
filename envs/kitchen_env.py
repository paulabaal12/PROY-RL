"""Ambiente He(RL)l Kitchen: cocina estocástica y parcialmente observable (POMDP).

La especificación completa (estados, acciones, transiciones, recompensas y
observaciones, con la justificación de cada decisión) está en docs/env_design.md.
Las referencias "(sección X)" de los comentarios apuntan a ese documento.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- Platos (sección 3.3) ---------------------------------------------------
DISHES = ("egg", "meat")
DISH_RATE_RANGE = {0: (0.08, 0.14), 1: (0.04, 0.08)}  # velocidad r ~ U(lo, hi)
DISH_DEADLINE_RANGE = {0: (20, 30), 1: (35, 50)}      # plazo tau0 ~ U{lo, ..., hi}

# Calidad según el nivel real de cocción c (sección 3.1)
RAW_BELOW = 1.0     # c < 1.0          -> crudo
BURNT_FROM = 1.3    # c >= 1.3         -> quemado; en medio -> bien cocido

# --- Acciones (sección 3.2) -------------------------------------------------
# 0 = wait; para la estación k: 1 + 4k + op, con op en (start, taste, serve, discard)
WAIT = 0
OP_START, OP_TASTE, OP_SERVE, OP_DISCARD = range(4)
OP_NAMES = ("start", "taste", "serve", "discard")

# --- Layout de la observación (sección 3.5) ---------------------------------
# Por estación: 9 características, en este orden.
ST_OCCUPIED, ST_EGG, ST_MEAT, ST_VISUAL, ST_HAS_TASTED, ST_LAST_TASTE, \
    ST_SINCE_TASTE, ST_COOK_TIME, ST_TIME_LEFT = range(9)
STATION_FEATURES = 9
# Por lugar de la cola: hay orden, one-hot del plato, tiempo restante.
QUEUE_FEATURES = 4

# Constantes de normalización
TIME_NORM = 50.0          # plazos y tiempos restantes
SINCE_TASTE_CAP = 20      # pasos desde la última prueba
COOK_TIME_CAP = 50        # pasos cocinando
COOK_OBS_LOW, COOK_OBS_HIGH = -1.0, 3.0  # rango de lecturas de cocción observadas


def action_id(station: int, op: int) -> int:
    """Convierte (estación, operación) al ID de acción discreta."""
    return 1 + 4 * station + op


def decode_action(action: int):
    """Inverso de action_id. Devuelve (None, None) para wait."""
    if action == WAIT:
        return None, None
    return (action - 1) // 4, (action - 1) % 4


def classify(cook: float) -> str:
    """Calidad de la comida según su nivel real de cocción."""
    if cook < RAW_BELOW:
        return "raw"
    if cook >= BURNT_FROM:
        return "burnt"
    return "ok"


@dataclass
class Order:
    order_id: int
    dish: int
    deadline: int     # plazo original tau0
    time_left: int    # tiempo restante tau

    def to_dict(self):
        return {"id": self.order_id, "dish": DISHES[self.dish],
                "deadline": self.deadline, "time_left": self.time_left}


@dataclass
class Station:
    order: Order | None = None
    cook: float = 0.0           # nivel real de cocción c (oculto)
    rate: float = 0.0           # velocidad de cocción r (oculta)
    cook_steps: int = 0
    has_tasted: bool = False
    last_taste: float = 0.0
    steps_since_taste: int = 0

    @property
    def occupied(self) -> bool:
        return self.order is not None


class KitchenEnv(gym.Env):
    """Cocina con varias estaciones, órdenes con plazo y observación ruidosa."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        n_stations: int = 2,
        queue_size: int = 3,
        horizon: int = 200,
        arrival_prob: float = 0.10,
        sigma_obs: float = 0.15,
        sigma_taste: float = 0.02,
        sigma_proc: float = 0.01,
        taste_cost: float = 0.2,
        invalid_penalty: float = 0.1,
        serve_reward: float = 10.0,
        speed_bonus: float = 5.0,
        bad_serve_penalty: float = 5.0,
        expire_penalty: float = 5.0,
        render_mode: str | None = None,
    ):
        self.n_stations = n_stations
        self.queue_size = queue_size
        self.horizon = horizon
        self.arrival_prob = arrival_prob
        self.sigma_obs = sigma_obs
        self.sigma_taste = sigma_taste
        self.sigma_proc = sigma_proc
        self.taste_cost = taste_cost
        self.invalid_penalty = invalid_penalty
        self.serve_reward = serve_reward
        self.speed_bonus = speed_bonus
        self.bad_serve_penalty = bad_serve_penalty
        self.expire_penalty = expire_penalty
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(1 + 4 * n_stations)

        obs_size = STATION_FEATURES * n_stations + QUEUE_FEATURES * queue_size + 1
        low = np.zeros(obs_size, dtype=np.float32)
        high = np.ones(obs_size, dtype=np.float32)
        # Las lecturas de cocción (visual y de prueba) pueden pasar de 1 (quemado)
        # o ser negativas por el ruido; el resto de características está en [0, 1].
        for k in range(n_stations):
            for feat in (ST_VISUAL, ST_LAST_TASTE):
                low[k * STATION_FEATURES + feat] = COOK_OBS_LOW
                high[k * STATION_FEATURES + feat] = COOK_OBS_HIGH
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Tres generadores independientes derivados de self.np_random.
        # Separar las órdenes del resto garantiza que, con la misma seed, la
        # secuencia de órdenes (cuándo llegan, qué plato, qué plazo) sea idéntica
        # sin importar qué acciones tome el agente. Así DQN, PPO y los baselines
        # se evalúan sobre exactamente las mismas órdenes.
        child_seeds = self.np_random.integers(0, 2**63 - 1, size=3)
        self.rng_orders = np.random.default_rng(child_seeds[0])
        self.rng_cooking = np.random.default_rng(child_seeds[1])
        self.rng_obs = np.random.default_rng(child_seeds[2])

        self.t = 0
        self.next_order_id = 0
        self.stations = [Station() for _ in range(self.n_stations)]
        self.queue: list[Order] = []
        self.last_events: list[dict] = []
        self.last_action: int | None = None

        # Estado inicial b0: estaciones vacías y una orden en cola (sección 3.7)
        first = self._new_order()
        self.queue.append(first)
        self.last_events = [{"type": "arrived", **first.to_dict()}]

        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"Acción inválida: {action}"
        reward = 0.0
        events: list[dict] = []
        taste_station = None

        # 1. Acción del agente
        k, op = decode_action(action)
        if k is not None:
            station = self.stations[k]
            if op == OP_START and not station.occupied and self.queue:
                order = self.queue.pop(0)
                lo, hi = DISH_RATE_RANGE[order.dish]
                self.stations[k] = Station(order=order, rate=float(self.rng_cooking.uniform(lo, hi)))
                events.append({"type": "started", "station": k, **order.to_dict()})
            elif op == OP_TASTE and station.occupied:
                # La lectura se toma después de cocinar este paso (paso 2b)
                taste_station = k
                reward -= self.taste_cost
            elif op == OP_SERVE and station.occupied:
                # Se evalúa con el c de ANTES de cocinar este paso
                quality = classify(station.cook)
                order = station.order
                if quality == "ok":
                    reward += self.serve_reward + self.speed_bonus * order.time_left / order.deadline
                else:
                    reward -= self.bad_serve_penalty
                events.append({"type": f"served_{quality}", "station": k, "cook": station.cook,
                               "time_used": order.deadline - order.time_left, **order.to_dict()})
                self.stations[k] = Station()
            elif op == OP_DISCARD and station.occupied:
                # La orden vuelve al frente de la cola y conserva su plazo
                events.append({"type": "discarded", "station": k, "cook": station.cook,
                               **station.order.to_dict()})
                self.queue.insert(0, station.order)
                self.stations[k] = Station()
            else:
                # Acción inválida: se ejecuta como wait, con una penalización pequeña
                reward -= self.invalid_penalty
                events.append({"type": "invalid", "station": k, "op": OP_NAMES[op]})

        # 2. Cocción de todas las estaciones ocupadas
        for st in self.stations:
            if st.occupied:
                st.cook = max(0.0, st.cook + st.rate + float(self.rng_cooking.normal(0.0, self.sigma_proc)))
                st.cook_steps += 1
                st.steps_since_taste += 1

        # 2b. Resultado de probar, con el nivel ya actualizado
        if taste_station is not None:
            st = self.stations[taste_station]
            st.last_taste = st.cook + float(self.rng_obs.normal(0.0, self.sigma_taste))
            st.has_tasted = True
            st.steps_since_taste = 0
            events.append({"type": "tasted", "station": taste_station,
                           "result": st.last_taste, "cook": st.cook})

        # 3. Plazos: se descuenta un paso a todas las órdenes activas
        for order in self.queue:
            order.time_left -= 1
        for st in self.stations:
            if st.occupied:
                st.order.time_left -= 1
        expired = [o for o in self.queue if o.time_left <= 0]
        self.queue = [o for o in self.queue if o.time_left > 0]
        for k, st in enumerate(self.stations):
            if st.occupied and st.order.time_left <= 0:
                expired.append(st.order)
                self.stations[k] = Station()
        for order in expired:
            reward -= self.expire_penalty
            events.append({"type": "expired", **order.to_dict()})

        # 4. Llegada de órdenes. Los números aleatorios se sacan SIEMPRE, aunque
        #    la cola esté llena, para que la secuencia de rng_orders no dependa
        #    de lo que haga el agente.
        arrival_roll = self.rng_orders.random()
        candidate = self._new_order()
        if arrival_roll < self.arrival_prob and len(self.queue) < self.queue_size:
            self.queue.append(candidate)
            events.append({"type": "arrived", **candidate.to_dict()})

        self.t += 1
        terminated = False                     # no hay estados terminales (sección 4)
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
        lo, hi = DISH_DEADLINE_RANGE[dish]
        deadline = int(self.rng_orders.integers(lo, hi + 1))
        order = Order(order_id=self.next_order_id, dish=dish, deadline=deadline, time_left=deadline)
        self.next_order_id += 1
        return order

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.last_visual = []
        for k, st in enumerate(self.stations):
            base = k * STATION_FEATURES
            # El ruido visual se sortea en cada paso, aunque la estación esté vacía,
            # para que el consumo de rng_obs no dependa del estado.
            visual = st.cook + self.rng_obs.normal(0.0, self.sigma_obs)
            self.last_visual.append(float(visual) if st.occupied else None)
            if not st.occupied:
                continue
            obs[base + ST_OCCUPIED] = 1.0
            obs[base + ST_EGG + st.order.dish] = 1.0
            obs[base + ST_VISUAL] = np.clip(visual, COOK_OBS_LOW, COOK_OBS_HIGH)
            obs[base + ST_HAS_TASTED] = float(st.has_tasted)
            if st.has_tasted:
                obs[base + ST_LAST_TASTE] = np.clip(st.last_taste, COOK_OBS_LOW, COOK_OBS_HIGH)
                obs[base + ST_SINCE_TASTE] = min(st.steps_since_taste, SINCE_TASTE_CAP) / SINCE_TASTE_CAP
            obs[base + ST_COOK_TIME] = min(st.cook_steps, COOK_TIME_CAP) / COOK_TIME_CAP
            obs[base + ST_TIME_LEFT] = min(st.order.time_left / TIME_NORM, 1.0)

        # Solo se observan los primeros queue_size lugares de la cola. Un
        # `discard` con la cola llena puede dejar una orden extra al final: no se
        # ve, pero es la última en salir por FIFO (ver env_design.md, sección 10).
        qbase = STATION_FEATURES * self.n_stations
        for i, order in enumerate(self.queue[: self.queue_size]):
            base = qbase + i * QUEUE_FEATURES
            obs[base] = 1.0
            obs[base + 1 + order.dish] = 1.0
            obs[base + 3] = min(order.time_left / TIME_NORM, 1.0)

        obs[-1] = (self.horizon - self.t) / self.horizon
        return obs

    def _get_info(self) -> dict:
        """Estado oculto real y eventos del paso. El agente NO debe usarlo para decidir."""
        return {
            "t": self.t,
            "true_cook": [st.cook if st.occupied else None for st in self.stations],
            "true_rate": [st.rate if st.occupied else None for st in self.stations],
            "stations": [st.order.to_dict() if st.occupied else None for st in self.stations],
            "queue": [o.to_dict() for o in self.queue],
            "events": self.last_events,
        }

    # ----------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            return None
        lines = [f"=== t = {self.t:3d}/{self.horizon} " + "=" * 40]
        if self.last_action is not None:
            k, op = decode_action(self.last_action)
            lines.append("Acción: " + ("wait" if k is None else f"{OP_NAMES[op]} estación {k}"))
        for k, st in enumerate(self.stations):
            if not st.occupied:
                lines.append(f"[Estación {k}] vacía")
                continue
            # Barra del nivel REAL: '.' crudo, '#' bien cocido, '!' quemado.
            filled = int(min(st.cook, 1.6) / 1.6 * 32)
            bar = "".join("." if i < 20 else ("#" if i < 26 else "!") for i in range(filled))
            taste = f"{st.last_taste:.2f} (hace {st.steps_since_taste})" if st.has_tasted else "-"
            lines.append(
                f"[Estación {k}] {DISHES[st.order.dish]:<4} orden#{st.order.order_id} "
                f"plazo {st.order.time_left:2d}/{st.order.deadline} | real c={st.cook:.2f} "
                f"({classify(st.cook)}) r={st.rate:.3f} | agente ve ~{self.last_visual[k]:.2f} "
                f"| última prueba: {taste}"
            )
            lines.append(f"             |{bar:<32}|  (crudo < 1.0 <= ok < 1.3 <= quemado)")
        queue = ", ".join(f"{DISHES[o.dish]}#{o.order_id}({o.time_left})" for o in self.queue) or "vacía"
        lines.append(f"Cola: {queue}")
        if self.last_events:
            lines.append("Eventos: " + "; ".join(e["type"] for e in self.last_events))
        return "\n".join(lines)
