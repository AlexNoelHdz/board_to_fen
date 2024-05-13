<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">Clasificación de tableros de ajedrez digitales</h3>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://www.mit.edu/~amini/LICENSE.md)

</div>

---

<p align="center">
    <br> 
</p>

## 📝 Table of Contents

- [Acerca de](#about)
- [Estructura del proyecto](#project-structure)
- [Getting Started](#getting-started)
- [Uso](#usage)
- [Autores](#authors)

## 🧐 About <a name = "about"></a>

Modelo que clasifca imágenes de tableros digitales de ajedrez entrenado con datos de al menos 28 estilos de tableros y 32 estilos de piezas

## 🎋 Project structure <a name = "project-structure"></a>
data/train: En esta carpeta se colocan las imágenes de entrenamiento del modelo.
data/test:  En esta carpeta se colocan las imágenes de prueba del modelo.
Proyecto.ipynb: Desarrollo del proyecto

## 🏁 Getting Started <a name = "getting_started"></a>

Instrucciones para copiar y correr el modelo

### Prerequisites

```bash
git clone https://github.com/AlexNoelHdz/board_to_fen
cd CHESS_openings
pip install -r requirements.txt
```

## 🎈 Usage <a name="usage"></a>

Despues de clonar el repositorio, descargar o generar las imágenes de prueba y entrenamiento. Una opción es hacerlo de este enlace: [Datos](https://www.kaggle.com/datasets/koryakinp/chess-positions).

El modelo preentrenado esta  disponible para usarse como 'modelo_2024_05_13.h5'
Con las imágenes descargadas, se puede reentrenar el modelo con las configuraciones que se escojan. 
```
Corre manualmente: .\Proyecto.ipynb
```

## ✍️ Authors <a name = "authors"></a>

- [@AlexNoelHdz](https://github.com/AlexNoelHdz)

