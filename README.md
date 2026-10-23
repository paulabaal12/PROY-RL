# He(RL)l Kitchen

Proyecto Final — Reinforcement Learning, Universidad del Valle de Guatemala (UVG).

Entrenamos agentes de aprendizaje por refuerzo profundo para manejar una cocina simulada, estocástica y **parcialmente observable (POMDP)**, en la que deben completar órdenes antes de su tiempo límite. Como cualquier chef, el agente solo puede hacer **una acción a la vez** y no ve directamente el estado real de cocción: recibe observaciones ruidosas o incompletas y puede gastar una acción en **probar (taste)** la comida para obtener información más precisa.

Comparamos dos familias de algoritmos:

- **Off-policy:** DQN
- **On-policy:** PPO

El eje del análisis es cómo cada agente equilibra **recolección de información** (probar), **velocidad** (tiempo de entrega) y **calidad** (comida bien cocinada). Este dilema de "informarse más vs. actuar rápido" es análogo al de dominios críticos como el diagnóstico médico, el mantenimiento predictivo o el trading.

## Integrantes

| Nombre          | Carné |
|-----------------|-------|
| Jose Marchena   | 22398 |
| Paula Barillas  | 22764 |

**Fecha límite:** `[PENDIENTE: ver Canvas]`

---

## Tablero de tareas

Trabajamos una tarea a la vez. Una tarea se marca como terminada solo cuando se cumple su criterio de "terminado" y se registró su entrada en [`docs/prompts_log.md`](docs/prompts_log.md).

> **Ajustes respecto a la lista original (y por qué):**
> - Se agregó **T0** para que la creación de este tablero quede registrada como una tarea más.
> - En **T3** se añadió un **script de evaluación común** (`experiments/evaluate.py`). Así los baselines, DQN y PPO se miden exactamente con el mismo código y las mismas seeds de evaluación; si cada agente tuviera su propio evaluador, la comparación de T6 y T7 no sería justa.
> - En **T3** también se exige que el ambiente exponga el estado oculto real en `info`. Lo necesitan T8 (replays) y T9 (barra de cocción real vs. observada), y es más barato diseñarlo desde el inicio que agregarlo después.
> - La decisión **desde cero vs. stable-baselines3** se toma **una sola vez al inicio de T4** y aplica también a T5, para que ambos agentes compartan infraestructura y la comparación no dependa de la implementación.

- [x] **T0. Tablero maestro del proyecto**
  - **Objetivo:** tener un punto central con el plan, la estructura y las reglas de trabajo.
  - **Entregable:** `README.md` (este archivo) y `docs/prompts_log.md` con su plantilla y la primera entrada.
  - **Terminado cuando:** ambos archivos existen y los revisaron los dos integrantes.

- [x] **T1. Setup del repositorio**
  - **Objetivo:** dejar un entorno reproducible para que cualquiera pueda clonar el repo y trabajar.
  - **Entregable:** estructura de carpetas (ver abajo), `requirements.txt` (gymnasium, torch, numpy, matplotlib, pandas, pytest, etc. con versiones fijas), `.gitignore` (venv, `__pycache__`, checkpoints pesados, `node_modules`), instrucciones para crear el entorno virtual y `utils/seeding.py` con una función `set_global_seed(seed)` que fije las seeds de `random`, `numpy`, `torch` y el ambiente.
  - **Terminado cuando:** desde un clon limpio, `python -m venv .venv` + `pip install -r requirements.txt` funciona sin errores, `pytest` corre (aunque solo haya un test de humo) y un test verifica que dos corridas con la misma seed producen los mismos números aleatorios.

- [x] **T2. Diseño formal del ambiente (POMDP)**
  - **Objetivo:** definir formalmente el problema antes de programarlo, justificando cada decisión.
  - **Entregable:** `docs/env_design.md` con: estados ocultos (nivel real de cocción, estado de cada orden), espacio de observaciones (ruidosas/incompletas) y qué revela la acción "probar", espacio de acciones, dinámica de transición estocástica, función de recompensa (entregas a tiempo, comida cruda/quemada, costo de probar, penalización por tiempo), condiciones de término/truncamiento, manejo de múltiples órdenes con tiempo límite y los parámetros configurables (ruido, costo de probar, etc.) que usaremos en las ablaciones.
  - **Terminado cuando:** el documento cubre todos los elementos de la tupla del POMDP (S, A, T, R, Ω, O, γ), cada decisión tiene su justificación y ambos integrantes lo aprobamos.

- [x] **T3. Implementación del ambiente + baselines**
  - **Objetivo:** convertir el diseño de T2 en un ambiente Gymnasium verificado, con referencias de desempeño.
  - **Entregable:**
    - `envs/kitchen_env.py` (subclase de `gymnasium.Env`) que expone el estado oculto real en `info`.
    - Render en texto/ASCII (`render_mode="ansi"`).
    - `tests/test_env.py`: pasa `gymnasium.utils.env_checker.check_env`, determinismo con seed, límites de observaciones, recompensas y término esperados en casos controlados.
    - `agents/baselines.py`: política aleatoria y heurística hecha a mano (p. ej. "probar cada k pasos y servir cuando la prueba diga *listo*").
    - `experiments/evaluate.py`: evaluador común (N episodios, seeds fijas) que reporta las métricas base.
  - **Terminado cuando:** `pytest` pasa, se puede ver un episodio completo en ASCII y existe una tabla con las métricas de ambos baselines (la heurística debe superar claramente a la aleatoria; si no, el ambiente o la recompensa tienen un problema).

- [ ] **T4. DQN**
  - **Objetivo:** entrenar un agente off-policy sobre el ambiente.
  - **Entregable:** primero, una decisión documentada **desde cero en PyTorch vs. stable-baselines3** (con pros y contras, la decidimos juntos). Luego `agents/dqn/` (o la configuración de SB3), `experiments/train_dqn.py` con configuración en archivo (YAML/JSON), logging (CSV y/o TensorBoard) de recompensa por episodio, loss y epsilon, checkpoints en `results/models/` y curvas de aprendizaje en `results/figures/`.
  - **Terminado cuando:** el agente entrenado supera a la política aleatoria en el evaluador común, el entrenamiento es reproducible con la misma seed y las curvas están guardadas.

- [ ] **T5. PPO**
  - **Objetivo:** entrenar un agente on-policy sobre el mismo ambiente.
  - **Entregable:** `agents/ppo/` (misma decisión de implementación que en T4), `experiments/train_ppo.py` con configuración en archivo, logging de recompensa, policy loss, value loss, entropía y KL aproximada, checkpoints y curvas de aprendizaje.
  - **Terminado cuando:** el agente supera a la política aleatoria en el evaluador común, el entrenamiento es reproducible y las curvas están guardadas.

- [ ] **T6. Experimentos y comparación**
  - **Objetivo:** obtener resultados estadísticamente defendibles y estudiar el equilibrio información vs. velocidad vs. calidad.
  - **Entregable:** `experiments/run_experiments.py` (o scripts por experimento) que corra:
    - Ambos agentes con **≥ 5 seeds** cada uno.
    - Un **tuning de hiperparámetros** acotado y documentado (qué se probó y qué se eligió).
    - Al menos **una ablación**: con/sin memoria (frame stacking o red recurrente) y/o variando el costo de "probar" y el nivel de ruido de observación.
    - Resultados crudos guardados en `results/runs/`.
  - **Terminado cuando:** todos los runs terminaron, sus logs están guardados y existe un registro (configuraciones + seeds) suficiente para repetir cualquier experimento.

- [ ] **T7. Análisis de resultados**
  - **Objetivo:** convertir los runs en evidencia para el reporte.
  - **Entregable:** `experiments/analyze.py`, que calcula: recompensa acumulada, tasa de órdenes completadas a tiempo, % de comida mal cocinada (cruda/quemada), número de pruebas por orden, sample efficiency (pasos necesarios para alcanzar un umbral) y estabilidad entre seeds (media ± desviación estándar o intervalos de confianza). Genera figuras en `results/figures/` y tablas en `results/tables/` (CSV + Markdown/LaTeX), además de un borrador de conclusiones en `docs/analysis.md`.
  - **Terminado cuando:** un solo comando regenera todas las figuras y tablas a partir de `results/runs/`, y `docs/analysis.md` responde explícitamente cuál agente equilibra mejor información, velocidad y calidad, y por qué.

- [ ] **T8. Exportación de trayectorias**
  - **Objetivo:** producir datos para visualizar el comportamiento de los agentes.
  - **Entregable:** `experiments/export_replays.py`, que corre episodios con DQN y PPO entrenados **con la misma seed y las mismas órdenes** y guarda cada paso (estado real oculto, observación, acción, recompensa, timer, órdenes activas) en JSON dentro de `results/replays/`, con un esquema documentado.
  - **Terminado cuando:** los JSON se generan de forma determinista (dos corridas producen archivos idénticos) y un test valida el esquema.

- [ ] **T9. Visualizador (opcional pero deseable)**
  - **Objetivo:** que el resultado se pueda **ver** en la presentación y en el video.
  - **Entregable:** app React + TypeScript (Vite) en `viewer/` que reproduce los JSON de T8: cocina animada, DQN vs. PPO lado a lado en la misma orden, barra de cocción real vs. lo que observa el agente, timer, eventos de "probar" resaltados y controles de reproducción (play/pausa/velocidad/paso a paso). Incluye un panel con curvas de entrenamiento y tabla de métricas (datos exportados en T7).
  - **Terminado cuando:** `npm install && npm run dev` levanta la app, `npm run build` genera un build estático funcional y se puede reproducir al menos un replay completo de cada agente.

- [ ] **T10. Cierre del repositorio**
  - **Objetivo:** que el repo público sea autocontenido y reproducible.
  - **Entregable:** este README con instrucciones finales de instalación y ejecución, un único flujo de comandos (p. ej. `make all` o `scripts/run_all.sh` / `.ps1`) que corra todo desde cero y `docs/prompts_log.md` completo.
  - **Terminado cuando:** en un clon limpio, siguiendo solo el README, se reproduce el pipeline completo (entrenamiento → análisis → replays → viewer) y el prompts log tiene entradas para todas las tareas.

- [ ] **Transversal: registro de uso de IA**
  - **Objetivo:** cumplir el requisito de uso responsable de IA.
  - **Entregable:** una entrada en `docs/prompts_log.md` por cada prompt relevante.
  - **Terminado cuando:** cada tarea T0–T10 tiene al menos una entrada con resultado, causa, evaluación y aporte.

---

## Estructura de carpetas (propuesta)

```
PROY-RL/
├── README.md               # Este tablero
├── requirements.txt        # Dependencias de Python con versiones fijas
├── envs/                   # Ambiente Gymnasium (KitchenEnv) y wrappers (p. ej. frame stacking)
├── agents/
│   ├── baselines.py        # Política aleatoria y heurística
│   ├── dqn/                # Implementación/configuración de DQN
│   └── ppo/                # Implementación/configuración de PPO
├── utils/                  # Seeds, logging, utilidades compartidas
├── experiments/            # Scripts de entrenamiento, evaluación, análisis y exportación
│   └── configs/            # Hiperparámetros en YAML/JSON
├── results/
│   ├── models/             # Checkpoints entrenados
│   ├── runs/               # Logs crudos de cada corrida (CSV / TensorBoard)
│   ├── figures/            # Gráficas para el reporte
│   ├── tables/             # Tablas de métricas
│   └── replays/            # Trayectorias JSON para el visualizador
├── tests/                  # Tests unitarios (pytest)
├── docs/
│   ├── env_design.md       # Diseño formal del POMDP (T2)
│   ├── analysis.md         # Análisis y conclusiones (T7)
│   └── prompts_log.md      # Historial de prompts (uso responsable de IA)
└── viewer/                 # Frontend React + TypeScript (Vite) (T9)
```

---

## Cómo correr el proyecto

> *Placeholder: se irá llenando conforme avancen las tareas.*

### Requisitos

- Python 3.13 (probado en Windows, solo CPU; no se necesita GPU).

### 1. Instalación

```powershell
git clone <url-del-repo>
cd PROY-RL
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

> En Linux, instalar primero PyTorch para CPU para no descargar la versión con CUDA:
> `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`

### 2. Correr los tests

```powershell
pytest
```

### 3. Ver un episodio en ASCII

```powershell
python -m experiments.demo_episode --policy heuristic --seed 0 --steps 40
```

Cada paso muestra la acción, el nivel **real** de cocción de cada estación (con una barra), lo que **ve** el agente (lectura con ruido), la última prueba, los plazos y la cola. Con `--policy random` se ve la política aleatoria; si se omite `--steps`, corre el episodio completo.

### 4. Evaluar los baselines

```powershell
python -m experiments.evaluate --episodes 100
```

Escribe `results/tables/baselines_summary.{csv,md}` y `baselines_episodes.csv`. Resultados actuales (100 episodios, media ± desviación estándar):

| Política | Retorno | Órdenes a tiempo | Servidas mal | Pruebas por orden | Órdenes expiradas |
|---|---|---|---|---|---|
| Aleatoria | −88.7 ± 14.7 | 2.0 % | 97.0 % | 1.14 | 5.2 |
| Heurística | 142.3 ± 31.4 | 78.5 % | 0.8 % | 2.38 | 3.8 |

### Pendientes

5. **Entrenar DQN / PPO** — *(T4/T5)*
5. **Correr experimentos** — *(T6)*
6. **Generar métricas, figuras y tablas** — *(T7)*
7. **Exportar replays** — *(T8)*
8. **Levantar el visualizador** — *(T9)*

---

## Decisiones de diseño

> *Placeholder: aquí registramos cada decisión importante, las opciones consideradas y por qué elegimos la que elegimos.*

| # | Tarea | Decisión | Alternativas consideradas | Justificación |
|---|-------|----------|---------------------------|---------------|
| 1 | T1 | Entorno con `venv` + `requirements.txt` con versiones fijas | `conda`, `uv`, Poetry | Es lo más simple, viene con Python y no requiere instalar herramientas extra. Fijar versiones garantiza que ambos integrantes tengamos el mismo entorno. |
| 2 | T1 | Python 3.13 y PyTorch solo CPU | GPU con CUDA | Ninguno de los dos tiene GPU NVIDIA. Para redes pequeñas (MLP) y un ambiente ligero, la CPU alcanza. |
| 3 | T1 | Hiperparámetros en archivos YAML (`experiments/configs/`) | Argumentos de línea de comandos, JSON | YAML admite comentarios para justificar cada valor, y cada experimento queda registrado en un archivo versionado. |
| 4 | T1 | Seeds en tres niveles: generadores globales (`random`, NumPy, PyTorch), generador del ambiente (`env.reset(seed=...)`) y espacios (`space.seed`) | Fijar solo la seed global | Gymnasium tiene su propio generador por ambiente y por espacio; si se fija solo la seed global, la cocina y la política aleatoria no serían reproducibles. |
| 5 | T1 | Se versionan los modelos finales, figuras, tablas y replays; se ignoran los checkpoints intermedios y los eventos de TensorBoard | Ignorar todo `results/` | Permite regenerar el análisis y usar el visualizador sin reentrenar, sin inflar el repo. |
| 6 | T2 | Escala del ambiente: 2 estaciones, 2 platos (huevo y carne), cola de hasta 3 órdenes, 9 acciones | 1 estación (mínima); 3 estaciones con recetas de varios pasos (extendida) | Es la escala más chica en la que probar tiene costo de oportunidad (la otra estación sigue cocinando), y entrena en minutos en CPU. Detalle en `docs/env_design.md`. |
| 7 | T2 | Probar devuelve un valor continuo con poco ruido (σ = 0.02) | Categoría exacta (crudo / listo / quemado) | Indica cuánto falta para que esté listo y facilita las ablaciones moviendo σ. |
| 8 | T2 | Las acciones inválidas se ejecutan como `wait` con −0.1 | Action masking | SB3 no soporta masking en DQN; usar masking solo en PPO haría injusta la comparación. |
| 9 | T2 | La observación incluye memoria mínima: última prueba, pasos desde esa prueba y tiempo de cocción | Observación sin memoria | Evita que el agente base sea inútil; la ablación mide el aporte del frame stacking adicional. |
| 10 | T2 | Bonus por velocidad: +5·(tiempo restante / plazo) al servir bien | Recompensa fija +10 | Hace explícito y medible el equilibrio entre velocidad y calidad. |
| 11 | T2 | Velocidad de cocción oculta y sorteada por plato | Velocidad fija por tipo de plato | Si fuera fija, el agente podría memorizar tiempos sin necesidad de observar ni probar. |
| 12 | T3 | Tres generadores aleatorios separados (órdenes, cocción, observación) | Un solo generador (`env.np_random`) | Con la misma seed, todas las políticas reciben exactamente las mismas órdenes: la comparación es justa y T8 puede mostrar DQN y PPO "en la misma orden". |
| 13 | T3 | Evaluador común con seeds de evaluación propias (10000 + i) | Que cada agente reporte sus propias métricas de entrenamiento | Mismo código y mismas seeds para todos; las seeds de evaluación no se usan en entrenamiento. |
| 14 | T3 | Se mantienen los parámetros iniciales de T2 | Ajustarlos antes de entrenar | La calibración cumplió el criterio: la heurística supera claramente a la aleatoria sin llegar al 100 % de órdenes a tiempo. |

---

## Uso responsable de IA

Este proyecto se desarrolla con apoyo de asistentes de IA (Claude Code). Para cumplir con los lineamientos del curso:

- Todo el código generado con IA se revisa, se prueba (tests, métricas, inspección manual) y debe poder ser explicado por ambos integrantes en la defensa. No incorporamos "cajas negras".
- Las decisiones de diseño con varias opciones razonables las toma el equipo; la IA presenta alternativas con pros y contras.
