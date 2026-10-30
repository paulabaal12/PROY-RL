# He(RL)l Kitchen v2: diseño experimental

**Proyecto Final — Reinforcement Learning, UVG.** Jose Marchena (22398), Paula Barillas (22764).

Este documento reemplaza al diseño v1 (2 estaciones, 2 platos, cola FIFO). Define el ambiente, la metodología, las métricas y los resultados que el proyecto debe producir. Es la especificación de la que se implementan T3 (ambiente y baselines) a T7 (análisis).

> **Convención.** Lo marcado como **(propuesta)** es un valor o decisión inicial sugerida, todavía no confirmada por el equipo o pendiente de calibrar en el Paso 0 (sección 4.2). Todo cambio posterior se registra en la sección 10.

---

## 1. Resumen

Una cocina simulada, estocástica y **parcialmente observable** donde un chef (el agente) hace **una acción por paso**: preparar ingredientes, poner platos al fuego, probar, servir, descartar o esperar. Llegan órdenes al azar de **tres platos con perfiles distintos**, cada una con una **hora de entrega** y, más tarde, una **hora en que el cliente se va**. El nivel real de cocción no se ve: solo una lectura visual con ruido, o una lectura casi exacta si el chef gasta un paso en probar.

Comparamos un algoritmo **off-policy (DQN)** contra uno **on-policy (PPO)** en su capacidad de aprender **manejo del tiempo**: decidir qué orden atender, cuándo adelantar preparación, cuándo informarse (probar) y cuándo actuar, bajo presión y con información incompleta.

### Qué cambia respecto a v1 y por qué

| Cambio | Motivo |
|---|---|
| Se agrega **preparación** (pasos previos a cocinar) | En v1, con las estaciones ocupadas el chef solo podía probar o esperar: probar casi no tenía costo de oportunidad y el chef pasaba mucho tiempo ocioso. En una cocina real el chef siempre está ocupado. La preparación representa en general todo el trabajo que no es cocinar (picar, sazonar, traer ingredientes) |
| **Tres platos** con perfiles opuestos | Que priorizar importe: un plato que llegó después puede necesitar empezar antes |
| **Hora de entrega ≠ hora de expiración** | En v1 la orden desaparecía al vencer el plazo. Si llegar tarde da lo mismo que no llegar, el agente aprende a abandonar órdenes difíciles, y abandonar no es una cualidad de un chef |
| Servir mal es **peor** que perder la orden | Entre entregar tarde y entregar comida cruda o quemada, preferimos lo primero |
| **Ventana visible de N órdenes** en lugar de FIFO estricto | El agente elige sobre cuál orden trabajar. N es el eje de ablación principal |
| **Randomización de dominio** | Cada episodio es una cocina distinta (más o menos órdenes, estaciones, ruido) para que el agente aprenda a adaptarse y no a memorizar un horario |
| Se elimina el bonus por velocidad | Entregar antes no da más recompensa; solo deja más tiempo libre |
| Se elimina frame stacking del plan | La observación ya incluye la memoria relevante (sección 3.5) y cada ablación extra multiplica el entrenamiento |

---

## 2. Objetivos

### 2.1 Objetivo general

Comparar DQN y PPO en un POMDP de manejo de tiempo y determinar cuál equilibra mejor **recolección de información** (probar), **velocidad** (entregar a tiempo) y **calidad** (comida bien cocida), y cómo cambia ese equilibrio con la **capacidad de priorizar** (tamaño N de la ventana visible).

### 2.2 Objetivos específicos

1. Implementar el ambiente v2 en Gymnasium, con estado oculto expuesto en `info` y pruebas automáticas.
2. Calibrar los valores de recompensa y los rangos de randomización con políticas escritas a mano, **sin entrenar** (Paso 0), y justificar cada valor con datos.
3. Entrenar DQN y PPO para cada N ∈ {1, 2, 4, 6} con 5 seeds cada uno.
4. Medir desempeño (retorno, entregas a tiempo, calidad) y comportamiento (pruebas, ociosidad, priorización) en seeds de prueba separadas y en escenarios fijos de distinta dificultad.
5. Determinar si los agentes aprenden a priorizar, comparando contra la política FIFO y contra el mismo agente con N = 1.

### 2.3 No-objetivos

- Construir la mejor política posible para la cocina: el objetivo es la **comparación**, no el récord.
- Diseñar baselines sofisticados. Los baselines son referencias, no competidores.
- Modelar una cocina realista en detalle (recetas de varios pasos, lavar platos, varios chefs).

### 2.4 Anti-objetivos (cosas que harían fallar el proyecto aunque el código funcione)

- Que priorizar **no importe**: si el agente con N = 1 rinde igual que con N = 6, la pregunta central no tiene respuesta. El Paso 0 debe verificar que el ambiente lo permite.
- Que **abandonar órdenes** sea rentable para el agente.
- Que la comparación DQN vs. PPO dependa de diferencias de implementación, presupuesto o tuning, y no del algoritmo.
- Que los resultados no se puedan reproducir con las seeds registradas.

---

## 3. El ambiente (POMDP)

La tupla es **(S, A, T, R, Ω, O, γ)** con distribución inicial **b₀**.

### 3.1 Platos

| Plato | Preparación (pasos, fija) | Velocidad de cocción `r` (oculta, se sortea por plato) | Pasos crudo → listo | Ventana "bien cocido" | Hora de entrega τ_due |
|---|---|---|---|---|---|
| Vegetales salteados | **7** | U(0.14, 0.20) | 5 a 7 | 1.5 a 2 pasos | U{25, …, 35} |
| Huevo | **4** | U(0.08, 0.14) | 8 a 13 | 2 a 4 pasos | U{20, …, 30} |
| Carne (steak) | **1** | U(0.04, 0.08) | 13 a 25 | 4 a 8 pasos | U{35, …, 50} |

Todos los valores son **(propuesta)**, a calibrar en el Paso 0.

- **Perfiles.** Los salteados son casi todo preparación y se queman rápido (ventana corta, piden vigilancia y pruebas). La carne es casi toda cocción, con ventana amplia (conviene ponerla al fuego pronto aunque haya llegado tarde). El huevo está en medio.
- **Calidad** según el nivel real de cocción `c`, igual para todos los platos: crudo si `c < 1.0`, **bien cocido** si `1.0 ≤ c < 1.3`, quemado si `c ≥ 1.3`. La diferencia entre platos sale de la velocidad `r`: un plato más rápido pasa menos pasos dentro de la ventana.
- **La velocidad `r` se sortea por plato** y nunca se observa. Si fuera fija, el agente podría memorizar tiempos sin mirar ni probar.
- **La preparación es determinista.** Un chef sabe cuánto le falta por picar; la incertidumbre del problema se concentra en la cocción y en las llegadas.

### 3.2 Estado oculto S

| Componente | Descripción |
|---|---|
| `t` | Paso actual, de 0 a H = 200 |
| Por estación k (máximo K_max = 3): | |
| `available_k` | Si la estación existe en esta cocina (depende de K, sección 3.7) |
| `dish_k`, `order_k` | Plato y orden que se está cocinando |
| `c_k` | **Nivel real de cocción (oculto)** |
| `r_k` | **Velocidad de cocción (oculta)** |
| `cook_steps_k` | Pasos que lleva cocinándose |
| `last_taste_k`, `steps_since_taste_k` | Resultado y antigüedad de la última prueba |
| Cola de órdenes pendientes | En **orden de llegada**, hasta Q_max = 10 **(propuesta)**. Cada orden: plato, pasos de preparación hechos, hora de entrega, tiempo de paciencia restante |

### 3.3 Órdenes, cola y ventana visible

- **Cola.** Las órdenes que todavía no están en una estación esperan en una cola ordenada por llegada. Si la cola está llena, la orden nueva no llega (sin penalización).
- **Ventana visible.** El agente ve y puede actuar sobre las **primeras N órdenes** de la cola (por orden de llegada). De las demás solo ve cuántas hay. Siguen perdiendo paciencia y pueden expirar sin que el agente pueda tocarlas: el costo de una ventana chica es real.
- **Con N = 1 el agente queda forzado a FIFO**: solo puede trabajar sobre la orden más antigua. Por eso la ablación sobre N mide directamente cuánto vale poder priorizar.
- Cuando una orden pasa a una estación, sale de la cola y libera su lugar en la ventana.
- **Hora de entrega y paciencia.** Cada orden tiene una hora de entrega τ_due (contada desde su llegada) y, después de ella, una **paciencia constante de P = 10 pasos (propuesta)**. Cuando se acaba la paciencia, el cliente se va: la orden expira, se quita de la cola o de su estación, y si estaba cocinándose la comida se bota. P es igual para todas las órdenes para que no sea otra variable que adivinar.

### 3.4 Acciones A

`Discrete(1 + 2N + 3·K_max)`. Con K_max = 3: **12 acciones con N = 1, 14 con N = 2, 18 con N = 4 y 22 con N = 6.**

| Acción | Válida si… | Efecto |
|---|---|---|
| `wait` | siempre | Nada |
| `prep_i` (i < N) | hay orden en el lugar i de la ventana y le falta preparación | Hace un paso de preparación de esa orden |
| `place_i` (i < N) | la orden i está completamente preparada y hay una estación disponible libre | La pone en la estación libre de menor índice, con `c = 0` |
| `taste_k` | la estación k está ocupada | Lectura casi exacta de `c_k` (sección 3.6) |
| `serve_k` | la estación k está ocupada | Entrega el plato, se evalúa la calidad con el `c_k` actual y se libera la estación |
| `discard_k` | la estación k está ocupada | Bota la comida. La orden vuelve a la cola en su lugar original (por llegada), **con la preparación por rehacer** y su misma hora de entrega |

- **`place_i` no elige estación.** Las estaciones son idénticas, así que elegir cuál solo agregaría acciones equivalentes. La observación de cada estación dice qué plato y qué orden tiene.
- **`serve_k` funciona en cualquier estación**: si un plato de una orden más reciente está listo, se puede servir aunque otras esperen.
- **Acciones inválidas** (incluidas las que apuntan a estaciones inexistentes cuando K < 3) se ejecutan como `wait` con −0.1. No se usa action masking, porque DQN en stable-baselines3 no lo soporta y dárselo solo a PPO haría injusta la comparación.

### 3.5 Recompensas R

| Evento | Recompensa | Justificación |
|---|---|---|
| Servir bien cocido **hasta la hora de entrega** | **+10** | Plano: entregar antes no da más, solo libera tiempo |
| Servir bien cocido con **d pasos de retraso** (1 ≤ d ≤ P) | **+2.5 · (1 − (d − 1)/P)** | Cae 75 % apenas pasa la hora de entrega y luego baja en línea recta hasta casi 0 (+0.25 en d = P). Siempre es positivo: terminar tarde siempre es mejor que abandonar |
| Orden **expirada** (se acaba la paciencia) | **−8** | Penalización alta: el cliente se fue |
| Servir **crudo o quemado** | **−10** | Lo peor: servir comida insegura es peor que perder la orden |
| Probar | **−0.2** (a calibrar en el Paso 0) | Costo explícito además del paso perdido |
| Acción inválida | −0.1 | Señal pequeña para aprender la estructura de las acciones |
| `wait`, `prep`, `place`, `discard` | 0 | `discard` ya cuesta: la preparación y la cocción perdidas |

Valores **(propuesta)**. Lo que sí está decidido es el **orden**: a tiempo > tarde > expirada > mal servida, y el castigo fuerte justo después de la hora de entrega.

**Por qué este orden elimina el abandono.** Mientras la orden no expire, servirla bien siempre da algo positivo, y dejarla expirar cuesta −8. Un agente que deja de trabajar en una orden atrasada pierde al menos 8 puntos. Y como servir mal (−10) es peor que expirar (−8), tampoco conviene "despachar" comida cruda para cerrar la orden.

### 3.6 Observación Ω y modelo O

Vector `Box` de **38 + 7N** números (66 con N = 4), normalizados aproximadamente a [0, 1], o a [−1, 1] donde aplica.

**Por estación (12 × K_max = 36):**

| Característica | Valor | ¿Ruidosa? |
|---|---|---|
| `available` | 1 si la estación existe en esta cocina | No |
| `occupied` | 0 / 1 | No |
| `dish_onehot` | 3 valores | No |
| `visual_cook` | `c + N(0, σ_obs²)`, ruido nuevo en cada paso; 0 si está vacía | **Sí** |
| `has_tasted` | 0 / 1 | No |
| `last_taste` | resultado de la última prueba | Poco ruido (σ_taste = 0.02) |
| `steps_since_taste` | `min(pasos, 20) / 20` | No |
| `cook_time` | `min(cook_steps, 50) / 50` | No |
| `time_to_due` | (τ_due − tiempo transcurrido) / 50; negativo si ya está atrasada (mínimo −P/50) | No |
| `patience_left` | paciencia restante / P (1 hasta la hora de entrega) | No |

**Por lugar de la ventana (7 × N):** `present`, `dish_onehot` (3), `prep_progress` (pasos hechos / pasos requeridos), `time_to_due`, `patience_left`.

**Global (2):** tiempo restante del episodio `(H − t)/H` y órdenes en cola fuera de la ventana `/ Q_max`.

- **Resultado de probar:** `taste = c' + N(0, σ_taste²)`, medido después de la cocción del paso. Un `serve` en el paso siguiente evalúa exactamente ese `c'` (más el ruido del fuego, muy pequeño).
- **Por qué es POMDP:** `c` solo se ve con ruido y `r` nunca se ve. La observación del paso no es un estado de Markov.
- **Memoria incluida en la observación:** la última prueba, los pasos desde esa prueba y el tiempo cocinando permiten estimar `r ≈ prueba / pasos cocinando` sin redes recurrentes. Por esto no se usa frame stacking (sección 9).
- **Los parámetros de la randomización no se observan** (excepto K, que se ve en `available`). El agente debe adaptarse por lo que observa, por ejemplo, notando lecturas visuales más inestables.

### 3.7 Dinámica T

Orden de eventos en cada paso:

1. **Acción del agente.** `serve` evalúa la calidad con el `c` de antes de cocinar este paso.
2. **Cocción.** En cada estación ocupada: `c ← max(0, c + r + ε)`, ε ~ N(0, 0.01²).
3. **Tiempo.** Todas las órdenes (en cola y en estaciones) avanzan un paso hacia su hora de entrega o consumen paciencia. Las que se quedan sin paciencia expiran.
4. **Llegadas.** Con probabilidad p (sección 3.8) llega una orden; el plato se elige uniformemente y τ_due según el plato.
5. **Observación** del nuevo estado.

**Generadores aleatorios separados**, derivados de la seed del `reset` (como en v1, más uno nuevo): **parámetros de la cocina** (randomización), **órdenes**, **cocción** y **observación**. Cada paso consume siempre la misma cantidad de números de cada generador. Así, con la misma seed, todas las políticas enfrentan la misma cocina y la misma secuencia de órdenes, sin importar lo que hagan.

### 3.8 Randomización de dominio

Al hacer `reset`, cada episodio sortea su cocina **(propuesta; los rangos se fijan en el Paso 0)**:

| Parámetro | Rango de entrenamiento | Qué representa |
|---|---|---|
| Número de estaciones K | {2, 3} | Capacidad de la cocina. K = 1 se evalúa como caso extremo fuera de distribución si el Paso 0 muestra que es jugable |
| Probabilidad base de llegada p | U(0.06, 0.16) | Día tranquilo vs. día lleno |
| Ruido visual σ_obs | U(0.05, 0.25) | Desde "a simple vista casi alcanza" hasta "la vista casi no sirve" |
| **Horas pico** | 0 a 2 por episodio, de 20 pasos, con llegada min(1, 2.5·p); el inicio se sortea uniformemente y no se traslapan | Situaciones de alta presión deliberadas, para que el agente no solo vea el caso promedio |

**Fijos (no se randomizan):** las recompensas, incluido el costo de probar (son la forma de evaluar al chef, no una propiedad de la cocina; si variaran sin que el agente lo vea, serían ruido en la recompensa), σ_taste = 0.02, el ruido del fuego, los pasos de preparación, P y H.

### 3.9 Término, descuento y estado inicial

- **No hay estados terminales.** Truncamiento a los H = 200 pasos. En DQN (targets) y PPO (ventajas) **se hace bootstrap con V(s_H)** en el truncamiento; tratarlo como final real vuelve miope al agente cerca del final.
- **γ = 0.99** (horizonte efectivo de unos 100 pasos, más que el ciclo de vida de cualquier orden: ≤ 50 + 10).
- **b₀:** estaciones vacías y una orden en la cola, con la cocina sorteada según la sección 3.8.

### 3.10 Información en `info`

`step()` devuelve todo lo que el agente no ve, para métricas (T7) y replays (T8): `true_cook`, `true_rate`, los parámetros de la cocina sorteados, todas las órdenes con sus tiempos y la lista de eventos del paso (`served_ok`, `served_late`, `served_raw`, `served_burnt`, `expired`, `tasted`, `prepped`, `placed`, `discarded`, `invalid`, `arrived`). Ninguna política usa `info` para decidir.

---

## 4. Metodología

### 4.1 Variables

| Tipo | Variable | Niveles |
|---|---|---|
| Independiente | Algoritmo | DQN, PPO |
| Independiente | Ventana visible N | 1, 2, 4, 6 (más el N por defecto del Paso 0, si no está en la lista) |
| Controlada | Ambiente | Randomización de la sección 3.8 en entrenamiento; escenarios fijos en evaluación |
| Controlada | Presupuesto | Mismos pasos de entrenamiento, misma red y mismo presupuesto de tuning para ambos algoritmos |
| Dependientes | Ver la sección 5 | Desempeño y comportamiento |

### 4.2 Paso 0: calibración sin entrenamiento

Antes de entrenar, se fijan los valores marcados como **(propuesta)** usando solo políticas escritas a mano (minutos de CPU):

1. **Costo de probar.** Una versión de la heurística que prueba cada k pasos (k = 1, …, 8) se corre para varios costos de probar (por ejemplo 0, 0.1, 0.2, 0.5, 1.0) y varios σ_obs. Se elige un costo con el que **el mejor k quede en medio**: ni "nunca probar" ni "probar siempre". Así, probar es una decisión real.
2. **Rango de σ_obs.** Se elige entre el valor con el que la heurística casi no necesita probar y el valor con el que la lectura visual ya no sirve.
3. **N por defecto.** Se corre la heurística y se mide cuántas órdenes hay en la cola en cada paso. El N por defecto es el que cubre la cola el 90 % del tiempo.
4. **Que priorizar importe.** Una heurística de verificación que puede elegir órdenes (por ejemplo, la de hora de entrega más próxima, adelantando preparación) debe superar claramente a la FIFO. Si no lo hace, se ajustan los platos o las horas de entrega. Esta heurística **solo se usa aquí**; no es un baseline del reporte.
5. **Dificultad.** La heurística FIFO debe superar claramente a la aleatoria y **no** estar cerca de 100 % de entregas a tiempo, para dejar margen al RL.

**Resultado:** una tabla y una figura por punto, y los valores elegidos registrados en la sección 10.

### 4.3 Baselines (no se entrenan)

- **Aleatoria:** acción uniforme. Cota inferior; demuestra que el ambiente no se resuelve solo.
- **Heurística FIFO ingenua:** trabaja solo sobre la orden más antigua (lugar 0): la prepara, la pone al fuego en cuanto hay estación libre, prueba cuando la lectura visual se acerca a lista, sirve si la prueba está en la ventana y descarta si está quemada. Funciona igual con cualquier N.

Evaluarlos cuesta segundos. Además, la comparación **agente con N = 1 vs. heurística FIFO** muestra cuánto aprende el RL dentro de la misma restricción.

### 4.4 Agentes

- **DQN** (off-policy) y **PPO** (on-policy), con la misma red MLP y el mismo preprocesamiento.
- La decisión **desde cero en PyTorch vs. stable-baselines3** se toma al inicio de T4 y aplica a ambos (pendiente).
- **Bootstrap en truncamiento** en ambos (sección 3.9).
- **Tuning:** una sola vez, con el N por defecto, con el mismo número de configuraciones para cada algoritmo, evaluando en las seeds de validación. Los hiperparámetros elegidos se reutilizan para todos los N.
- **Presupuesto de entrenamiento:** el mismo número de pasos de ambiente para todas las corridas. Se fija después de una corrida piloto que mida el tiempo por corrida **(pendiente)**.

### 4.5 Diseño experimental

- **Grilla:** 2 algoritmos × 4 valores de N × 5 seeds de entrenamiento = **40 corridas**, todas en el ambiente randomizado.
- **Tres conjuntos de seeds separados:**
  - **Entrenamiento:** 0 a 4.
  - **Validación:** 20000 + i. Se usan para el tuning y para elegir el mejor N de cada algoritmo.
  - **Prueba:** 10000 + i. Se usan solo para los números del reporte.
  Elegir el mejor N con las mismas seeds con que se reporta inflaría el resultado del ganador.
- **Checkpoint evaluado:** siempre el **final** de cada corrida (no el mejor en validación), para que la selección de checkpoint no favorezca a ningún algoritmo **(propuesta)**.
- **Comparación principal:** el mejor N de DQN vs. el mejor N de PPO (elegidos en validación, reportados en prueba), más la **curva completa de desempeño vs. N** de cada algoritmo, porque la interacción algoritmo × N es un resultado en sí.

### 4.6 Escenarios de evaluación

Cada agente entrenado (y cada baseline) se evalúa, sin reentrenar, en 100 episodios de cada escenario:

| Escenario | Configuración **(propuesta)** |
|---|---|
| Distribución de entrenamiento | Randomización completa |
| Fácil | K = 3, p = 0.06, σ_obs = 0.05, sin horas pico |
| Medio | K = 2, p = 0.10, σ_obs = 0.15, una hora pico |
| Difícil | K = 2, p = 0.16, σ_obs = 0.25, dos horas pico |
| Ruido bajo / ruido alto | Medio, con σ_obs en los extremos del rango |
| Fuera de distribución | K = 1 (si el Paso 0 lo valida) |

Esto da el análisis de sensibilidad (por ejemplo, si el mejor N cambia con el ruido) **sin entrenar nada extra**.

---

## 5. Métricas

**Desempeño:**

| Métrica | Definición |
|---|---|
| Retorno | Suma de recompensas del episodio |
| Tasa a tiempo | servidas bien a tiempo / órdenes cerradas (servidas + expiradas) |
| Tasa tarde | servidas bien tarde / órdenes cerradas |
| Tasa de expiración | expiradas / órdenes cerradas |
| Tasa mal servida | (crudas + quemadas) / servidas |
| Retraso medio | promedio de d en las servidas tarde |

**Comportamiento** (las que dicen *cómo* juega el agente):

| Métrica | Definición |
|---|---|
| Pruebas por orden | pruebas / órdenes servidas |
| Ociosidad | pasos con `wait` o acción inválida / pasos totales |
| Ociosidad evitable | pasos con `wait` cuando había alguna `prep_i` válida / pasos totales |
| Priorización | acciones `prep_i` o `place_i` con i > 0 / todas las `prep` y `place` |
| Preparación adelantada | `prep` hechas con todas las estaciones disponibles ocupadas / todas las `prep` |

**Entrenamiento:**

| Métrica | Definición |
|---|---|
| Sample efficiency | pasos de entrenamiento hasta superar a la heurística FIFO en validación |
| Estabilidad | media ± desviación estándar e IC 95 % (bootstrap) entre las 5 seeds |

Con 5 seeds la potencia estadística es baja. Se reportan intervalos de confianza y no se afirman diferencias cuyos intervalos se solapan ampliamente.

---

## 6. Resultados esperados (entregables)

Al terminar T7, el proyecto debe tener:

1. **Calibración (Paso 0):** retorno de la heurística vs. frecuencia de prueba k, por costo de probar y σ_obs; distribución del largo de la cola; tabla de valores elegidos.
2. **Curvas de entrenamiento de DQN y PPO para cada N:** retorno en validación vs. pasos, media y banda entre seeds, más las curvas propias de cada algoritmo (loss y epsilon en DQN; policy loss, value loss, entropía y KL en PPO).
3. **Desempeño vs. N:** retorno final (y tasa a tiempo) en función de N, una línea por algoritmo, con las heurísticas como líneas horizontales de referencia.
4. **Tabla principal:** mejor DQN vs. mejor PPO vs. heurística FIFO vs. aleatoria, en las seeds de prueba, con todas las métricas de la sección 5.
5. **Agente con N = 1 vs. heurística FIFO:** cuánto mejora el RL dentro de la misma restricción.
6. **Robustez:** tabla o heatmap de agente × escenario (sección 4.6).
7. **Comportamiento:** pruebas por orden, ociosidad, priorización y preparación adelantada vs. N, por algoritmo. Responde si los agentes aprendieron a priorizar y a mantenerse ocupados.
8. **Sample efficiency y estabilidad** de cada algoritmo.
9. **Replays (T8)** de DQN y PPO con la misma seed, para el visualizador.
10. **Conclusión escrita** (`docs/analysis.md`): cuál algoritmo equilibra mejor información, velocidad y calidad, y por qué.

### Hipótesis previas (a confirmar o refutar con los resultados)

Son expectativas razonables que los resultados deben poner a prueba, no conclusiones:

- **H1.** El desempeño mejora de N = 1 a N intermedios y luego se estanca o empeora, porque un espacio de acciones más grande es más difícil de aprender.
- **H2.** DQN es más eficiente en muestras (reutiliza experiencia con su replay buffer), y PPO es más estable entre seeds.
- **H3.** Ambos agentes prueban más en escenarios de ruido alto y más los salteados (ventana corta) que la carne.
- **H4.** Los agentes entrenados son menos ociosos que la heurística FIFO, porque adelantan preparación.

---

## 7. Impacto en el tablero de tareas

- **T2 (diseño):** se reabre; este documento lo reemplaza.
- **T3 (ambiente y baselines):** se reabre. `envs/kitchen_env.py`, `agents/baselines.py`, `experiments/evaluate.py` y sus tests se reescriben para v2. Se agrega el Paso 0 (`experiments/calibrate.py`) y los resultados de los baselines se regeneran.
- **T4 y T5:** sin cambios de alcance; entrenan sobre la grilla de N.
- **T6:** la ablación pasa a ser solo N. Se eliminan las ablaciones de costo de probar, ruido y frame stacking (ahora calibración, randomización y fuera de alcance, respectivamente).
- **T7:** usa las métricas de la sección 5 y produce los entregables de la sección 6.
- **T8 y T9:** el esquema de replays agrega preparación, cola y ventana.

---

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| El entrenamiento es demasiado lento para 40 corridas en CPU | Corrida piloto antes de fijar el presupuesto; correr seeds en paralelo; si no alcanza, reducir N a {1, 2, 4} |
| Ningún agente supera a la heurística FIFO | Es un resultado válido y se discute; el Paso 0 reduce el riesgo comprobando que existe margen |
| Priorizar no importa en el ambiente calibrado | Criterio 4 del Paso 0 |
| Las acciones inválidas dominan el aprendizaje con N grande | Penalización pequeña; medir la tasa de inválidas en las curvas |

---

## 9. Fuera de alcance (posibles extensiones)

- **Frame stacking y redes recurrentes:** la observación ya trae la memoria necesaria (última prueba, pasos desde la prueba, tiempo cocinando). Medir si memoria adicional ayuda es una pregunta lateral a DQN vs. PPO.
- **Lavar platos, traer ingredientes y otros recursos separados:** la preparación ya representa el trabajo que no es cocinar. Un recurso extra complicaría el análisis.
- **Action masking:** no se usa, para no favorecer a PPO.
- **Recetas de varios pasos, varios chefs, clientes con paciencia variable.**

---

## 10. Registro de cambios de parámetros

| Fecha | Parámetro | Antes → después | Motivo |
|---|---|---|---|
| 2026-10-23 | (v1) | Sin cambios | La calibración de v1 cumplió el criterio: heurística 142.3 ± 31.4 vs. aleatoria −88.7 ± 14.7; 78.5 % a tiempo |
| 2026-10-30 | Diseño completo | v1 → v2 | Preparación, 3 platos, entrega ≠ expiración, ventana visible, randomización de dominio (este documento) |
