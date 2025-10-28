# K-Means (Clustering & Compresión de Imágenes)

Implementación desde cero del algoritmo **K-Means** en Python, aplicado tanto al **agrupamiento de puntos en 2D** como a la **compresión de imágenes RGB reduciendo la cantidad de colores** representativos. :contentReference[oaicite:0]{index=0}

---

## Objetivos del proyecto

- Agrupar datos bidimensionales y visualizar sus clústeres.
- Aplicar K-Means sobre imágenes para reducir la profundidad de color.
- Comparar visualmente la imagen original vs. la imagen comprimida.
- Comprender el funcionamiento interno del algoritmo (sin librerías externas tipo scikit-learn).

---

## Estructura del repositorio

| Archivo          | Descripción |
|------------------|-------------|
| `kMeans.py`      | Implementación principal del algoritmo K-Means. |
| `prueba.py`      | Script de demostración (clustering + compresión). |
| `ex7data2.txt`   | Dataset de prueba con puntos en 2D. |
| `bird_small.png` | Imagen utilizada para compresión. |

---

## Requisitos e instalación

Este proyecto requiere **Python 3.8+** y las siguientes dependencias:

```bash
pip install numpy matplotlib

