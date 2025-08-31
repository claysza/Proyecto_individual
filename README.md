# Faces vs Non-Faces — README (Colab/Jupyter)

Este README acompaña el notebook **`Faces_vs_Non_Faces (1).ipynb`** y explica:
- Ruta elegida y dataset (fuente/licencia).
- Cómo ejecutar en Google Colab o Jupyter (y GPU si aplica).
- Cómo entrenar y evaluar (PCA, LDA, ICA con clasificador de **centroides más cercanos**).
- Cómo generar la **tabla** y el **gráfico** de métricas.

---

## 1) Dataset, estructura de carpetas y rutas

**Rutas usadas en el notebook (Colab):**
```text
/content/Face-Recognition/datasets/faces
/content/Face-Recognition/datasets/nonfaces
```

**Estructura esperada:**

- `faces/` → una subcarpeta por persona (p. ej., `s1, s2, …`) con imágenes de **92×112** px en escala de grises.
- `nonfaces/` → carpetas con imágenes negativas (no caras).

> **Nota:** En el código, las imágenes de `nonfaces` se convierten a escala de grises y se redimensionan a **92×112**; las de `faces` se asumen ya en ese tamaño. Si tus caras no tienen 92×112, ajústalas o añade el mismo `resize` a la rama de `faces`.

**Fuente y licencia (rellena según tu caso):**  
- **Fuente dataset de caras:** _[ej.: ORL/AT&T Faces (92×112), u otro]_  
- **Fuente dataset no-caras:** _[indica tu fuente]_  
- **Licencia:** _[ej.: uso académico/educativo; revisa y cita la licencia original]_

---

## 2) Requisitos y cómo abrir el notebook

### Opción A — Google Colab (recomendado)
1. Sube las carpetas `faces/` y `nonfaces/` a tu Google Drive en `Face-Recognition/datasets/` **o** súbelas al almacenamiento de Colab y actualiza las rutas del notebook si cambian.
2. Abre `Faces_vs_Non_Faces (1).ipynb` en Colab.
3. (Opcional) **GPU**: no es estrictamente necesaria (se usan PCA/LDA/ICA y NumPy). Si quieres activarla: **Entorno de ejecución → Cambiar tipo de hardware → GPU**.
4. Instala dependencias si faltan (Colab ya trae la mayoría):
   ```bash
   pip install pillow scikit-learn pandas
   ```

### Opción B — Jupyter local
1. Crea un entorno (opcional) e instala:
   ```bash
   pip install numpy pillow matplotlib scikit-learn pandas
   ```
2. Coloca `faces/` y `nonfaces/` en `Face-Recognition/datasets/` (o ajusta las rutas en el notebook).
3. Abre el `.ipynb` con Jupyter Lab/Notebook y ejecuta las celdas en orden.

---

## 3) Entrenamiento y evaluación (pasos breves)

> El notebook implementa **PCA**, **LDA (binaria 0/1)** y **ICA**; proyecta los datos y clasifica con **centroides más cercanos** (coseno). Las métricas incluyen **Top-1**, **Top-5**, **Precisión macro**, **Recall macro**, y **matriz de confusión**.

1. **Cargar imágenes**
   - Las celdas llaman a:
     ```python
     faces, _ = load_images('/content/Face-Recognition/datasets/faces')
     non_faces, _ = load_images('/content/Face-Recognition/datasets/nonfaces')
     faces_labels = np.ones((len(faces),1))
     non_faces_labels = np.zeros((len(non_faces),1))
     ```
   - Asegúrate de que las rutas apunten a tus carpetas reales.

2. **Dividir en train/test**
   - Usa la función `split_data(...)` ya incluida. Ejemplo que aparece en el notebook:
     ```python
     train_data, train_labels, test_data, test_labels = split_data(
         faces, faces_labels, non_faces, non_faces_labels,
         non_faces_count=400,  # puedes ajustar
         alpha=0.5,            # proporción train (50%)
         non_face_precentage_in_train=1
     )
     ```
   - Parámetros clave:
     - `alpha`: proporción para entrenamiento (ej. `0.5`).
     - `non_faces_count`: cuántas no-caras considerar.
     - `non_face_precentage_in_train`: fracción de no-caras que van a train.

3. **PCA → proyección + clasificación**
   ```python
   space_pca, mean_pca = PCA(train_data, alpha=0.85)  # puedes variar alpha
   Xtr_pca = project(train_data, mean_pca, space_pca)
   Xte_pca = project(test_data,  mean_pca, space_pca)

   cent_pca, classes_pca = fit_nearest_centroid(Xtr_pca, train_labels)
   scores_pca = predict_scores(Xte_pca, cent_pca, classes_pca)
   y_pred_pca = classes_pca[np.argmax(scores_pca, axis=1)]
   ```

4. **LDA (binaria)** — similar a PCA pero con `LDA(...)`:
   ```python
   W_lda = LDA(train_data, train_labels, k=1)
   zero_mean = np.zeros(train_data.shape[1])
   Xtr_lda = project(train_data, zero_mean, W_lda)
   Xte_lda = project(test_data,  zero_mean, W_lda)
   cent_lda, classes_lda = fit_nearest_centroid(Xtr_lda, train_labels)
   scores_lda = predict_scores(Xte_lda, cent_lda, classes_lda)
   y_pred_lda = classes_lda[np.argmax(scores_lda, axis=1)]
   ```

5. **ICA** — igual flujo que PCA usando `ICA(...)`:
   ```python
   n_comp_ica = space_pca.shape[1]        # o fija un número
   P_ica, mean_ica = ICA(train_data, n_components=n_comp_ica, max_iter=400, tol=1e-5, random_state=0)
   Xtr_ica = project(train_data, mean_ica, P_ica)
   Xte_ica = project(test_data,  mean_ica, P_ica)
   cent_ica, classes_ica = fit_nearest_centroid(Xtr_ica, train_labels)
   scores_ica = predict_scores(Xte_ica, cent_ica, classes_ica)
   y_pred_ica = classes_ica[np.argmax(scores_ica, axis=1)]
   ```

6. **Métricas (por método)**
   ```python
   top1_pca = topk_accuracy(scores_pca, test_labels, classes_pca, k=1)
   top5_pca = topk_accuracy(scores_pca, test_labels, classes_pca, k=5)
   prec_pca, rec_pca, cm_pca, lab_pca = precision_recall_macro(test_labels, y_pred_pca)

   # Repite para LDA e ICA → top1_*, top5_*, prec_*, rec_*, cm_*
   ```

> **Tip reproducibilidad:** si quieres resultados estables, añade `np.random.seed(0)` antes de dividir datos.

---

## 4) Tabla y gráficos de métricas

### 4.1 Tabla rápida (PCA/LDA/ICA)
La celda **“Gráfica resumen de métricas por método”** construye un `DataFrame` **`res`** con **Top‑1**, **Top‑5**, **Precisión**, **Recall** y lo muestra formateado:
```python
res = pd.DataFrame({
    "Metodo":   ["PCA", "LDA", "ICA"],
    "Top-1":    [top1_pca, top1_lda, top1_ica],
    "Top-5":    [top5_pca, top5_lda, top5_ica],
    "Precision":[prec_pca, prec_lda, prec_ica],
    "Recall":   [rec_pca, rec_lda, rec_ica],
}).set_index("Metodo")

display(res.style.format("{:.4f}"))
```
**Guardar a CSV** (ya incluido):
```python
summary_df = res.reset_index().rename(columns={"index":"Metodo"})
summary_df.to_csv("resumen_train_test_metodos.csv", index=False)
print("Guardado: resumen_train_test_metodos.csv")
```

### 4.2 Gráfico comparativo (barras)
La celda crea un gráfico agrupado con **Top‑1**, **Precisión** y **Recall**:
```python
metrics_to_plot = ["Top-1", "Precision", "Recall"]
X = np.arange(len(res.index))
width = 0.25

plt.figure(figsize=(8,4))
for i, m in enumerate(metrics_to_plot):
    plt.bar(X + i*width, res[m].values, width, label=m)
plt.xticks(X + width, res.index)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Comparativa de métricas por método")
plt.legend()
plt.grid(True, axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
```
> **Guardar la figura** (opcional): añade antes de `plt.show()`  
> `plt.savefig("metricas_comparativa.png", dpi=150, bbox_inches="tight")`

### 4.3 Otros gráficos incluidos
- **Barras individuales** por métrica (Top‑1, Top‑5, Precisión, Recall) en una cuadrícula.
- **Curva Accuracy vs Umbral** (verificación de identidad) si defines pares y etiquetas; ajusta el `grid` y el `metric` en `accuracy_from_threshold(...)`.

---

## 5) Notas útiles
- El clasificador usa **similaridad coseno** entre vectores proyectados y **centroides** por clase.
- **LDA** está implementada para el caso **binario (0/1)**.
- El tamaño de imagen objetivo es **112×92** (se visualiza en eigenfaces reconfigurando vectores a `(112, 92)`). Asegúrate de mantener el mismo a lo largo del pipeline.
- Si cambias rutas o tamaños, revisa las funciones `load_images`, `PCA/ICA/LDA`, `project` y cualquier `reshape`/`resize`.

---

## 6) Créditos y licencia del dataset
- **Autor/es del notebook:** Clayanela Zambrano 
- **Dataset(s):** AT&T Database of Faces
- **Licencia:** Libre
