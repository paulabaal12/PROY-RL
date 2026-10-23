# Diseño formal del ambiente: He(RL)l Kitchen

Este documento define formalmente el ambiente como un **POMDP** y justifica cada decisión de diseño. Es la especificación que se implementa en T3 (`envs/kitchen_env.py`). Los valores numéricos son **iniciales**: se calibran en T3 y cualquier cambio se registra en la sección 10.

---

## 1. Resumen

Una cocina con **2 estaciones** (hornillas) y **2 tipos de plato** (huevo y carne). Llegan órdenes al azar, cada una con un plazo. El agente es un chef que hace **una acción por paso**: empezar a cocinar, probar, servir, descartar o esperar.

La comida se cocina a una **velocidad desconocida** y el agente solo ve su nivel de cocción **con ruido**. Si gasta una acción en **probar**, obtiene una lectura casi exacta, pero pierde un paso en el que la otra estación sigue cocinando y los plazos siguen corriendo.

El agente tiene que decidir en todo momento entre **informarse** (probar), **ser rápido** (servir pronto, atender la otra estación) y **acertar** (servir la comida bien cocida).

### Por qué esta escala (variante B)

Se consideraron tres escalas (ver la entrada 3 de `docs/prompts_log.md`):

- **Una estación (A).** Descartada: con una sola estación, probar no tiene costo de oportunidad (no hay nada más que atender) y una heurística simple podría ser casi óptima.
- **Tres estaciones con recetas de varios pasos (C).** Descartada: entrenar tomaría horas por corrida en CPU y no cabría el presupuesto de T6 (2 agentes × al menos 5 seeds × ablaciones).
- **Dos estaciones (B).** Elegida: es la escala más chica donde el dilema información vs. velocidad vs. calidad es real, y cada corrida entrena en minutos.

---

## 2. Por qué es un POMDP y no un MDP

Si el agente viera el nivel real de cocción `c` y la velocidad `r`, el problema sería un MDP y bastaría con servir cuando `c` entra en la ventana correcta. Aquí, en cambio:

- `c` solo se observa con ruido gaussiano (σ = 0.15, la mitad del ancho de la ventana de "bien cocido").
- `r` nunca se observa: cambia en cada plato.

Por eso la observación del paso actual **no es un estado de Markov**. La política óptima depende de la historia completa. En teoría, el agente debería mantener una **creencia** (*belief*) `b(s)`, es decir, una distribución sobre el estado oculto. DQN y PPO con redes feed-forward no mantienen creencias, así que les damos dos aproximaciones:

1. **Características de memoria en la observación:** el último resultado de probar, los pasos desde esa prueba y el tiempo que lleva cocinándose el plato. Con estos datos, el agente puede estimar `r` aproximadamente como `prueba / pasos_cocinando`.
2. **Frame stacking (ablación de T6):** concatenar las últimas k observaciones. Como el ruido es independiente en cada paso, promediar k lecturas reduce su desviación estándar en un factor de `√k`.

---

## 3. Definición formal

El POMDP es la tupla **(S, A, T, R, Ω, O, γ)** más una distribución inicial **b₀**.

### 3.1 Estados ocultos S

El estado completo `s_t` contiene:

| Componente | Descripción | Dominio |
|---|---|---|
| `t` | Paso actual del episodio | {0, …, H}, con H = 200 |
| Por cada estación k ∈ {0, 1}: | | |
| `occupied_k` | Si hay comida en la estación | {0, 1} |
| `dish_k` | Tipo de plato | {huevo, carne} |
| `c_k` | **Nivel real de cocción** (oculto) | ℝ≥0 |
| `r_k` | **Velocidad de cocción** del plato (oculta) | ver 3.3 |
| `cook_steps_k` | Pasos que lleva cocinándose | ℕ |
| `last_taste_k`, `steps_since_taste_k` | Resultado y antigüedad de la última prueba | ℝ, ℕ |
| `order_k` | Orden asociada: tipo de plato, tiempo restante τ y plazo original τ₀ | |
| Cola de espera | Hasta Q = 3 órdenes que aún no se empiezan, cada una con (plato, τ, τ₀) | |

**Calidad según `c`:**

| Rango de `c` | Calidad |
|---|---|
| `c < 1.0` | Crudo |
| `1.0 ≤ c < 1.3` | **Bien cocido** |
| `c ≥ 1.3` | Quemado |

### 3.2 Acciones A

`Discrete(9)`. Es un espacio discreto pequeño, compatible tanto con DQN como con PPO.

| ID | Acción | Válida si… | Efecto |
|---|---|---|---|
| 0 | `wait` | siempre | Nada |
| 1 + 4k | `start_k` | la estación k está libre y la cola no está vacía | Toma la **primera** orden de la cola (FIFO) y la empieza en la estación k con `c = 0` |
| 2 + 4k | `taste_k` | la estación k está ocupada | Devuelve una lectura precisa de `c_k` (ver 3.5) |
| 3 + 4k | `serve_k` | la estación k está ocupada | Entrega el plato, evalúa su calidad con el `c_k` actual y libera la estación |
| 4 + 4k | `discard_k` | la estación k está ocupada | Bota la comida. La orden **vuelve al frente de la cola** y conserva su plazo |

**Acciones inválidas.** Se ejecutan como `wait` con una penalización de −0.1. **No** se usa action masking: SB3 no lo soporta en DQN, y si solo PPO lo tuviera, la comparación sería injusta. La penalización pequeña basta para que ambos agentes aprendan a evitarlas.

**FIFO en `start`.** El agente no elige qué orden empezar; siempre toma la más antigua. Así el espacio de acciones queda en 9 en vez de 2 × 3 + …, y el dilema central (probar o actuar) no se diluye en un problema de asignación de tareas.

**Por qué existe `discard`.** Si el agente prueba y descubre que la comida se quemó, puede botarla y empezar de nuevo, en vez de servir mal (−5). Esto le da **valor práctico** a la información obtenida al probar.

### 3.3 Transiciones T(s' | s, a): estocásticas

Orden de los eventos dentro de un paso:

1. **Acción del agente.** Se aplica la acción. `serve` evalúa la calidad con el `c` de **antes** de cocinar este paso.
2. **Cocción.** En cada estación ocupada:
   `c_k ← max(0, c_k + r_k + ε)`, con ε ~ N(0, σ_proc²) y σ_proc = 0.01. El término ε modela fluctuaciones pequeñas del fuego.
3. **Plazos.** A cada orden activa (en cola o en una estación) se le resta 1 a τ. Si τ llega a 0, la orden **expira**: se quita, y si estaba en una estación, la comida se bota y la estación queda libre.
4. **Llegadas.** Con probabilidad p = 0.10 llega una orden nueva. El plato se elige uniformemente y el plazo τ₀ se sortea según el plato (ver tabla). Si la cola está llena, la orden no llega (no hay penalización).
5. **Observación.** Se genera la observación del nuevo estado (sección 3.5).

**Parámetros por plato:**

| Plato | Velocidad `r` (se sortea al hacer `start`) | Pasos hasta estar crudo → listo | Ventana "bien cocido" | Plazo τ₀ |
|---|---|---|---|---|
| Huevo | U(0.08, 0.14) | 8 a 13 | 2 a 4 pasos | U{20, …, 30} |
| Carne | U(0.04, 0.08) | 13 a 25 | 4 a 8 pasos | U{35, …, 50} |

**Fuentes de estocasticidad:** la velocidad oculta de cada plato, el ruido del fuego, las llegadas de órdenes, el tipo de plato y el plazo, además del ruido de observación (sección 3.5).

Que `r` se sortee **por plato** y no sea fija por tipo es lo que obliga a probar. Si fuera fija, el agente podría memorizar "la carne está lista en 18 pasos" sin mirar nunca la comida.

### 3.4 Recompensas R(s, a, s')

| Evento | Recompensa | Justificación |
|---|---|---|
| Servir **bien cocido** | **+10 + 5·(τ / τ₀)** | Premia la calidad, con un bonus proporcional al tiempo que sobra. Así la **velocidad** tiene un valor explícito y medible |
| Servir crudo o quemado | **−5** | Servir mal debe ser peor que no hacer nada, para que el agente tenga un motivo para informarse |
| Orden expirada | **−5** | Igual que servir mal: perder una orden no debe ser "gratis". Si costara menos que servir mal, al agente le convendría dejar que las órdenes expiren |
| Probar | **−0.2** | Además del paso perdido, un costo explícito y configurable. Variarlo es una de las ablaciones de T6 |
| Acción inválida | **−0.1** | Señal pequeña para aprender la estructura del espacio de acciones |
| `wait` y `discard` | 0 | `discard` ya tiene su costo implícito: el tiempo perdido y la orden que tiene que volver a empezar |

El rango de recompensa por orden va de −5 a +15. La señal es **densa**, porque el agente recibe una recompensa cada vez que sirve u ocurre algo, pero no lleva shaping artificial: cada término corresponde a algo que pasa en la cocina.

### 3.5 Observaciones Ω y modelo de observación O(o | s', a)

La observación es un vector `Box` de **31 números reales**, normalizados aproximadamente a [0, 1] (o [−1, 1] donde aplique).

**Por estación (9 × 2 = 18):**

| Característica | Valor | ¿Ruidosa? |
|---|---|---|
| `occupied` | 0 / 1 | No |
| `dish_onehot` | 2 valores | No |
| `visual_cook` | `c + N(0, σ_obs²)`, con σ_obs = 0.15 y **ruido nuevo en cada paso**; 0 si la estación está vacía | **Sí** |
| `has_tasted` | 0 / 1 | No |
| `last_taste` | resultado de la última prueba de este plato (0 si no se ha probado) | Poco ruido |
| `steps_since_taste` | `min(pasos, 20) / 20` | No |
| `cook_time` | `min(cook_steps, 50) / 50` | No |
| `order_time_left` | τ / 50 | No |

**Por cada lugar de la cola (4 × 3 = 12):** si hay orden, `dish_onehot` y τ / 50.

**Global (1):** tiempo restante del episodio, `(H − t) / H`.

**Modelo del resultado de probar:** `taste = c' + N(0, σ_taste²)`, con σ_taste = 0.02. Se mide **después** de la cocción de ese paso, así que refleja el nivel en el momento en que el agente lo recibe.

**Por qué se eligió así:**

- **Probar da un valor continuo, no una categoría.** Una categoría ("crudo", "listo", "quemado") no dice *cuánto* falta, y el agente no podría estimar cuándo estará listo. Además, con valores continuos, para las ablaciones basta con mover σ_taste.
- **σ_obs = 0.15 vs. ventana de 0.3.** Con una sola lectura visual, cuando `c` está a mitad de la ventana (1.15), la probabilidad de que la lectura caiga fuera de [1.0, 1.3] es de aproximadamente 32 %. Sin probar, el agente se equivoca con frecuencia; con la prueba (σ = 0.02) el error es casi nulo. Este contraste es justamente lo que queremos estudiar.
- **Las características de memoria van incluidas en la observación**, para que el agente sin frame stacking no quede inútil. La ablación de T6 mide cuánto aporta la memoria **adicional**.
- **Las variables observables no llevan ruido.** Plazos, tipo de plato y si la estación está ocupada son cosas que un chef ve perfectamente. La incertidumbre se concentra solo en la cocción, para que el análisis sea claro.

### 3.6 Factor de descuento γ

**γ = 0.99.** El horizonte efectivo es `1 / (1 − γ) = 100` pasos, que cubre de sobra el ciclo de vida de una orden (≤ 50 pasos). Con un γ menor, el agente subvaloraría las recompensas de platos lentos como la carne.

### 3.7 Distribución inicial b₀

Al hacer `reset(seed)`, las dos estaciones empiezan vacías y en la cola hay **una orden** con plato y plazo sorteados. Esa orden inicial garantiza que el agente siempre tenga algo que hacer desde el primer paso.

---

## 4. Término del episodio

- **No hay estados terminales** (`terminated` siempre es `False`): la cocina no "se acaba".
- **Truncamiento** a los H = 200 pasos (`truncated = True`).

**Detalle importante para T4 y T5:** como el final es un truncamiento por tiempo y no un estado terminal real, al calcular los targets (DQN) o las ventajas (PPO) **hay que hacer bootstrap** con `V(s_H)`, en vez de asumir que el valor después del final es 0. Si se trata el truncamiento como un final real, el agente aprende que "el mundo se acaba" y cerca del final del episodio se vuelve miope. El paso de tiempo restante incluido en la observación también le ayuda a anticipar ese final.

---

## 5. Información adicional en `info` (para métricas y replays)

`step()` devuelve en `info` todo lo que el agente **no** ve, para usarlo en T7 (métricas) y T8 (replays):

- `true_cook`: los `c_k` reales.
- `true_rate`: los `r_k`.
- `orders`: todas las órdenes activas con su τ.
- `events`: lista de eventos de este paso (`served_ok`, `served_raw`, `served_burnt`, `expired`, `tasted`, `discarded`, `invalid`, `arrived`).

El agente nunca usa `info` para decidir; solo el código de evaluación lo lee.

---

## 6. Parámetros configurables (para las ablaciones de T6)

Todos son argumentos del constructor del ambiente, con los valores de este documento como defaults:

| Parámetro | Default | Uso en ablación |
|---|---|---|
| `sigma_obs` | 0.15 | Más ruido visual → probar vale más |
| `sigma_taste` | 0.02 | Peor información al probar |
| `taste_cost` | 0.2 | Probar más caro → ¿el agente prueba menos y se equivoca más? |
| `arrival_prob` | 0.10 | Más presión de tiempo |
| `horizon` | 200 | |
| `n_stations` | 2 | Se deja configurable, pero no es un eje de ablación prioritario |

El **frame stacking** no es un parámetro del ambiente: es un wrapper de Gymnasium (`FrameStackObservation`) aplicado encima.

---

## 7. Relación con las métricas de T7

| Métrica | Cómo se obtiene |
|---|---|
| Recompensa acumulada por episodio | Suma de las recompensas |
| Tasa de órdenes a tiempo | `served_ok / (served_ok + served_raw + served_burnt + expired)` |
| % de comida mal cocinada | `(served_raw + served_burnt) / servidos` |
| Pruebas por orden | `tasted / órdenes servidas` |
| Tiempo de entrega | Promedio de `τ₀ − τ` en los `served_ok` |
| Sample efficiency | Pasos de entrenamiento hasta superar el umbral de la heurística |
| Estabilidad | Desviación estándar e intervalo de confianza de lo anterior entre seeds |

---

## 8. Baselines previstos (T3)

- **Aleatoria:** muestrea acciones uniformemente de `Discrete(9)`. Sirve como cota inferior.
- **Heurística:**
  1. Si hay una estación libre y una orden en cola, empezar.
  2. Si una estación tiene `visual_cook ≥ 0.85` y no se ha probado en los últimos 2 pasos, probar.
  3. Si la última prueba está en [1.0, 1.25], servir. Si es ≥ 1.3, descartar.
  4. Si nada de lo anterior aplica, esperar.

  Es una política "sensata" escrita a mano. Si ni DQN ni PPO la superan, eso hay que discutirlo en el análisis.

**Criterio de calibración en T3:** la heurística debe superar claramente a la aleatoria, y la tasa de órdenes a tiempo de la heurística **no** debe estar cerca de 100 %. Si estuviera cerca, el ambiente sería demasiado fácil y no habría margen para que el RL mejore. Si alguna de las dos cosas falla, se ajustan los parámetros de la sección 3 y se documenta en la sección 10.

---

## 9. Alcance que se dejó fuera (posibles extensiones)

Recetas de varios pasos, clientes con paciencia variable, más de 2 estaciones y elegir qué orden empezar (en lugar de FIFO). Se dejaron fuera para que el presupuesto de cómputo alcance para los experimentos de T6. Si al final sobra tiempo, cualquiera de ellas se puede agregar sin cambiar la estructura del ambiente.

---

## 10. Registro de cambios de parámetros

| Fecha | Parámetro | Antes → después | Motivo |
|---|---|---|---|
| 2026-10-23 | (ninguno) | Sin cambios | La calibración de T3 (100 episodios) cumplió el criterio con los valores iniciales: la heurística obtuvo un retorno de 142.3 ± 31.4 contra −88.7 ± 14.7 de la aleatoria, y 78.5 % de órdenes a tiempo (no está cerca del 100 %, así que hay margen para el RL). |

### Detalles de implementación que precisan la especificación (T3)

- **Tres generadores aleatorios separados**, derivados de la seed del `reset`: uno para **órdenes** (llegadas, plato, plazo), otro para **cocción** (`r`, ruido del fuego) y otro para **observación** (ruido visual y de prueba). Además, en cada paso se sacan siempre los mismos números, ocurra lo que ocurra. Así, con la misma seed, **la secuencia de órdenes es idéntica sin importar las acciones del agente**, y DQN, PPO y los baselines se evalúan sobre exactamente las mismas órdenes. Con un solo generador, probar una vez más cambiaría todas las órdenes futuras.
- **Cola después de `discard`:** las llegadas se rechazan cuando la cola tiene `queue_size` órdenes, pero `discard` siempre devuelve la orden al frente. En el caso raro de descartar con la cola llena, queda una orden extra al final que no aparece en la observación. Como la cola es FIFO, esa orden es la última en salir, así que no afecta la próxima decisión del agente.
