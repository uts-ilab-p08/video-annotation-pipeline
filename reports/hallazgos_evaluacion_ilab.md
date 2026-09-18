# Hallazgos de Evaluación — iLab Proyecto #08 (CCTV Query System)
**Rol:** Evaluation Lead (Majo)
**Última actualización:** por completar cada vez que se agregue algo nuevo

---

## 1. Setup del pipeline de QA

**Fuente de datos (Supabase, proyecto "Annotations", id `uqofqjychtjsvfioebod`, schema `bronze`):**

| Tabla | Contenido | Filas |
|---|---|---|
| `bronze.videos` | 10 clips MEVA, todos 2018-03-05 | 10 |
| `bronze.events` | Eventos generados por el modelo del equipo (nombre corto + descripción en texto libre + frame inicio/fin) | 123 |
| `bronze.geometries` | Bounding boxes (pista de IoU — **sin empezar todavía**) | ~27,637 |
| `bronze.ground_truth` | Actividades reales de MEVA (auditadas), 23 tipos únicos, subidas por Majo desde archivos KPF locales | 161 |
| `bronze.matched_pairs` | Cruce evento-modelo ↔ referencia MEVA combinada, por solapamiento de frames | 123 |

**Dos pistas de evaluación** (confirmado en `Annotations_Dictionary.docx` del equipo):
1. **Texto** (`events.description` vs. actividad MEVA) → **LLM-as-judge (DeepEval/G-Eval)**
2. **Geometría** (bounding boxes) → **IoU** (intersection-over-union) — decisión ya tomada por el equipo, pista sin empezar

**Metodología pista de texto:** por cada evento del modelo, se buscan todas las actividades de MEVA cuyo rango de frames se solape con el suyo (mismo `camera_id` + `capture_start_local`), se ordenan cronológicamente, se convierten a oración vía diccionario de plantillas, y se combinan en un solo párrafo de referencia.

---

## 2. Hallazgos confirmados

### 2.1 Alucinación de detalles de escena
El modelo inventa elementos visuales que no existen en el video — ejemplos reales, cámara G336:
- *"passing by a central broccoli patch"*
- *"passing a circular garden bed"*

Mientras tanto, subdescribe la actividad real anotada (ej. MEVA dice `vehicle_makes_u_turn` + `vehicle_stops`; el modelo solo dice "se mueve").

### 2.2 Eventos sin respaldo en ground truth
De 123 eventos generados por el modelo, **44 (36%) no tienen ninguna actividad de MEVA que se solape en tiempo.**

**Por qué medimos distancia al frame más cercano:** que un evento no tenga solapamiento exacto no significa automáticamente que el modelo esté inventando — podía ser solo un desfase de unos pocos frames en cómo se cortó la ventana de tiempo. Para distinguir "esto es casi lo mismo, solo mal recortado" de "esto no tiene ninguna base real cerca", medimos qué tan lejos (en frames) está la actividad real de MEVA más cercana a cada evento sin match. Si la distancia es chica, es más probable que sea un problema técnico de recorte; si es grande, es más probable que el modelo esté generando actividad donde no pasaba nada.

Desglose por esa distancia:
- **~9 casos** — desfase chico (22-74 frames, ~2-3 segundos) → posible problema de recorte de ventana, no necesariamente alucinación.
- **~21 casos** — desfase grande (229 a 6,895 frames) → el modelo describe actividad en zonas donde MEVA no anotó nada cercano. Evidencia fuerte de sobre-generación/alucinación.
- Peor caso: G336, frame 8621-8744 (el mismo evento del "broccoli patch") — actividad real más cercana a **6,895 frames** de distancia.

### 2.3 Patrones de calidad por cámara

| Cámara | Eventos del modelo | Actividades MEVA reales | Patrón |
|---|---|---|---|
| **G505** | 17 | 4 | **Sobre-generación** — el modelo inventa actividad, concentra 6 de los peores 11 casos de "sin match lejano" |
| **G506** | 4 | 31 | **Sub-detección** — el modelo pierde la mayoría de la actividad real |
| **G423** (13:15) | 0 | 2 | **Falla total** — cero eventos generados pese a haber actividad real (ver 2.4) |
| G424 | 32 | 40 | Razonable |
| G340 | 20+10 | 24+9 | Razonable |
| G341 | 13 | 14 | Razonable |
| G336 | 13 | 14 | Razonable en cantidad, pero con alucinación de detalle (ver 2.1) |
| G328 | 9 | 19 | Sub-detección moderada |

### 2.4 G423 (13:15-13:20) — investigación de la falla total
El video tiene metadata normal (300 segundos, 9,000 frames, 1920×1080 — igual que sus vecinos) pero `object_count: 0`, `activity_count: 0`, `geometry_count: 0` — YOLO no detectó absolutamente nada en los 5 minutos completos. Comparando con el mismo G423 pero de las 13:20-13:25 (que sí tuvo 5 eventos y 149 geometrías), esto no era plausible como "no pasaba nada en la escena" — apuntaba a una **falla silenciosa en el paso de detección de objetos** para ese archivo específico (`data/raw/videos/2018-03-05.13-15-00.13-20-00.school.G423.r13.avi`).

**Confirmado manualmente:** Majo revisó el video directo en el navegador (vía `video_url` en Cloudflare) — sí hay actividad visible: una persona cruzando de un salón a otro, al fondo del cuadro, por unos segundos. Descarta por completo la hipótesis de "escena vacía." Es un bug confirmado en el detector/pipeline para este archivo específico, no un caso límite ambiguo.

### 2.5 Origen real de la alucinación del "broccoli patch"
No es el LLM de texto inventando de la nada — es un error **upstream**, en el detector de objetos (YOLO). En `bronze.geometries`, la clase `broccoli` aparece **719 veces**, con confianza promedio de **0.69** (hasta 0.82). YOLO viene entrenado en COCO (que tiene literalmente una clase "broccoli"), y probablemente confunde vegetación/arbustos de jardín en las escenas de MEVA con esa clase — de forma sistemática y consistente (no es ruido aislado; 719 veces con confianza relativamente alta es un sesgo repetido del modelo, no un error de una sola vez).

Esto es distinto al patrón de la sección 2.7 (`bird`, `chair`, `fire hydrant`, `motorcycle`) — esas sí son detecciones raras, de muy baja confianza (0.26-0.54) y pocas apariciones (n=1 cada una) — ruido puntual, no sesgo sistemático. `broccoli` es un problema del modelo confundiendo una clase real y frecuente en la escena (vegetación) con una categoría de COCO que no debería aparecer en un contexto de CCTV urbano/escolar.

**Implicación para la recomendación al equipo:** el fix no es ajustar el prompt del LLM de captioning — es revisar el detector de objetos. Para `broccoli` específicamente, valdría la pena revisar si se puede excluir esa clase por contexto (un CCTV de escuela/calle no debería detectar vegetales) o reentrenar/ajustar el detector para vegetación genérica en vez de una clase de comida tan específica.

Dato adicional: `truck` tiene confianza promedio notablemente más baja (0.339) que `car` (0.629) — el detector es menos confiable distinguiendo camiones.

### 2.6 Genericidad excesiva en las etiquetas del modelo
**"Vehicle movement"** (sin más detalle) representa **43 de 123 eventos (35%)** del output del modelo. Sumando todas las etiquetas genéricas de vehículo (Vehicle movement, Truck movement, Vehicle stationary, Vehicle movement detection, Vehicle parking, Vehicle appearance, Vehicle presence), son **66/123 (54%)**.

MEVA sí distingue con precisión (`vehicle_turns_left`, `vehicle_stops`, `vehicle_starts` son categorías separadas) — el modelo colapsa todo eso en una etiqueta vaga. Es un problema de **precisión por vaguedad**, distinto (y probablemente más frecuente) que la alucinación de detalles falsos.

**Matización importante:** la `description` completa (no solo el `event_name`) sí trae detalle — color del vehículo, número de etiqueta (`car #1423`), y dirección de movimiento en el cuadro ("de izquierda a derecha", "hacia el centro"). No es texto genérico copiado y pegado.

Pero **ninguna descripción revisada indica el tipo de maniobra** (giro, parada, arranque, vuelta en U) — que es justo lo que MEVA distingue como categorías separadas. El modelo describe **trayectoria visual en pantalla**, no la **semántica de la acción**. Desajuste de nivel de abstracción entre lo que genera el modelo y lo que MEVA (y probablemente el usuario final, al preguntar "¿quién giró aquí?") necesita.

**Recomendación:** dos caminos, no excluyentes —
- **Prompt-level:** darle al VLM la taxonomía de acciones de MEVA como categorías objetivo explícitas en el prompt (turn left/right, stop, start, U-turn), en vez de dejarlo describir libremente la trayectoria.
- **Post-proceso:** inferir el tipo de maniobra directo del tracking del bounding box (cambio de ángulo = giro, velocidad → 0 = parada) y combinarlo con el texto generado, en vez de depender 100% del LLM para clasificar la acción.

Consulta usada para verificar:
```sql
SELECT event_name, description FROM bronze.events WHERE event_name = 'Vehicle movement' LIMIT 8;
```
<!-- INSERTAR AQUÍ: screenshot de la consulta y sus resultados -->

### 2.7 Detecciones raras de baja confianza — manejo inconsistente y evento duplicado
Las clases raras del detector (`bird` 0.32, `chair` 0.38, `fire hydrant` 0.26, `motorcycle` 0.54 — todas de muy pocas apariciones, n=1, y muy por debajo de la confianza de `broccoli`, 0.692) reciben tratamiento inconsistente del lado del texto:
- `chair` y `motorcycle`: el LLM las ignora, describe otra cosa sin mencionarlas.
- `bird` (0.32): el texto dice **"dog"** en su lugar — ninguno de los dos confirmado.
- `fire hydrant` (0.26, la más baja de las cuatro): la MISMA detección (frame 8054, cámara G424, 268.47s) está vinculada a **13 eventos distintos y parcialmente contradictorios** para el mismo momento — algunos la afirman con seguridad, otros con duda ("possibly a fire hydrant or post"), la mayoría no la mencionan. Esto sugiere generación de eventos redundantes/casi-duplicados para el mismo objeto+frame, no solo un problema de precisión de texto.

**Consulta usada (para que Sebastián la corra y verifique):**
```sql
SELECT g.label, g.confidence, g.frame_index, g.timestamp_seconds, v.camera_id, v.scene, e.event_name, e.description
FROM bronze.geometries g
LEFT JOIN bronze.event_objects eo ON eo.object_id = g.object_id
LEFT JOIN bronze.events e ON e.event_id = eo.event_id
LEFT JOIN bronze.videos v ON v.video_id = e.video_id
WHERE g.label IN ('bird', 'chair', 'fire hydrant', 'motorcycle')
ORDER BY g.label;
```

**Resuelto — qué es realmente ese "fire hydrant":** no es un objeto separado. Es la **misma persona**, rastreada 2,284 veces en total por el tracker. De esas, **2,283 veces fue detectada correctamente como "person"**; **solo 1 vez, con confianza de apenas 0.26**, el detector la clasificó como "fire hydrant" — probablemente por su silueta alta y angosta (bounding box de 27×58 píxeles) combinada con una prenda roja que la persona lleva puesta. Esa única etiqueta ruidosa (1 de 2,284 detecciones) se filtró y quedó incluida en varias de las descripciones generadas, en vez de descartarse como el ruido estadístico que es. Esto también explica los "13 eventos": no son 13 objetos distintos, son múltiples descripciones candidatas generadas a lo largo del historial de tracking del MISMO objeto (la persona), sin deduplicar.

**Recomendación concreta y accionable para Sebastián:** al generar texto a partir del historial de un objeto, filtrar por `detection_count`/confianza antes de incluir una etiqueta en la descripción — una etiqueta con 1 detección y 0.26 de confianza, contra 2,283 detecciones consistentes de otra clase, debe descartarse como ruido, no colarse en el texto final. Además, agregar un paso de deduplicación/clustering espacio-temporal antes de la generación de texto, para que un mismo objeto/momento produzca una sola descripción canónica, no N eventos redundantes (esto probablemente explica también parte de la sobre-generación vista en G505).

### 2.8 Seguridad (para avisar a Sebastián)
Las 6 tablas del schema `bronze` **no tienen Row Level Security (RLS) activado** — cualquiera con la anon key podría leer/modificar los datos. Sin resolver todavía; pendiente de coordinar con el equipo antes de producción.

---

## 3. Pendiente / siguiente paso

- [ ] Correr DeepEval/G-Eval sobre las 123 filas de `bronze.matched_pairs` — **bloqueado por falta de API key** (OpenAI, Anthropic o Qwen) para el LLM juez.
- [ ] Empezar la pista de geometría/IoU (tabla `bronze.geometries`, sin tocar todavía).
- [ ] Validar si Qwen sirve como juez barato comparándolo contra GPT-4/Claude sobre el mismo lote.

---

*Este documento se actualiza a medida que avanza la evaluación. Última sesión de trabajo: matching por tiempo + hallazgos de alucinación confirmados con datos reales.*
