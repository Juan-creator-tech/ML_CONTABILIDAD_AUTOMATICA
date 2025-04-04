
# 🤖 Automatización Contable mediante Machine Learning

## 📌 Descripción del Problema | Problem Description

### 🇪🇸 Español

El objetivo de este proyecto es automatizar la generación de asientos contables a partir de los movimientos bancarios mensuales de una empresa. Tradicionalmente, esta tarea es manual, repetitiva y sujeta a errores humanos. Aquí proponemos una solución basada en inteligencia artificial que aprende a transformar automáticamente cada movimiento bancario en un asiento contable completo, siguiendo el formato del diario contable histórico.

### 🇬🇧 English

The goal of this project is to automate the generation of accounting journal entries from monthly bank transactions. This task is traditionally performed manually and is time-consuming and error-prone. We propose a machine learning solution that learns to automatically translate each bank transaction into a complete accounting journal entry using the structure learned from historical records.

---

## 📁 Dataset

### 🇪🇸 Español

El conjunto de datos es **privado**, perteneciente a una empresa. Incluye:

- **Movimientos bancarios** (`Movimientos_bancarios.xls`): fecha, descripción, importe y metadatos adicionales.
- **Diario contable** (`diario_contable.xlsx`): histórico de asientos contables con todas sus líneas (cuenta, debe, haber...).
- **Plan contable** (`plan_contable.xlsx`): listado de cuentas contables válidas (se filtran aquellas con 10 dígitos).

> ⚠️ Todos los archivos se encuentran bajo la carpeta `/data_sample` y **han sido modificados para anonimizar y desvirtualizar la información real de la compañía**.

### 🇬🇧 English

The dataset is **private**, belonging to a company. It includes:

- **Bank transactions** (`Movimientos_bancarios.xls`): date, description, amount, and metadata.
- **Accounting journal** (`diario_contable.xlsx`): historical journal entries with all lines.
- **Chart of accounts** (`plan_contable.xlsx`): valid accounting accounts (only 10-digit codes are kept).

> ⚠️ All files are located in `/data_sample` and **have been altered to anonimize and devirtualize the real info of the company**.

---

## 💡 Solución Propuesta | Proposed Solution

### 🇪🇸 Español

El sistema se basa en una arquitectura encoder-decoder (`T5-small` de Hugging Face) entrenada para traducir cada movimiento bancario (preprocesado como texto) en un asiento contable estructurado (formato JSON). Para ello:

- Se emparejan automáticamente los movimientos con asientos existentes por **fecha e importe**.
- Se genera un dataset de pares `input_text → asiento contable`.
- Se entrena un modelo que minimiza la pérdida de traducción en el tiempo.
- Se validan los asientos generados asegurando que:
  - Las cuentas existen en el plan contable.
  - El asiento cuadra (Debe = Haber).
- Se genera un Excel con los nuevos asientos (`asientos_generados.xlsx`) y se actualiza el diario histórico.

Además, se ha desarrollado una interfaz gráfica (Tkinter) con fondo personalizado e interacción fluida.

### 🇬🇧 English

The system is based on an encoder-decoder architecture (`T5-small` from Hugging Face) trained to translate each bank transaction (as preprocessed text) into a structured accounting journal entry (in JSON). It:

- Automatically pairs transactions with journal entries by **date and amount**.
- Builds a dataset of `input_text → journal entry`.
- Trains a model to minimize translation loss.
- Validates generated entries by ensuring:
  - Accounts exist in the chart of accounts.
  - Entries are balanced (Debit = Credit).
- Produces an Excel with generated entries (`asientos_generados.xlsx`) and updates the historical journal.

A graphical interface (Tkinter) has also been developed with a custom background and smooth interaction.

---

## 🗂️ Estructura del Proyecto | Project Structure

```
📦 proyecto-contabilidad-ml
  ├──src
  |    ├── data_sample/                     # Datos privados (banco, contabilidad)
  |    │   ├── banco/
  |    │   ├── contabilidad/
  |    │ 
  |    ├── models/                  # Modelos entrenados (T5)
  |    │   └── trained_t5_model_best/
  |    |
  |    ├── result/                  # Resultados generados (pares, asientos)
  |    │   ├── asientos_generados.xlsx
  |    │   ├── diario_contable_actualizado.xlsx
  |    │   └── pares_entrenamiento.xlsx
  |    |
  |    ├── utils/
  |    |     └── useful_library.py  # Funciones reutilizables (preprocesado, exportación...)
  |    |
  |    ├── results_notebook/
  |    |    └── notebook_final.ipynb
  |    |
  |    ├── img/
  |    │   └── background.png       # Imagen de fondo de la app
  |
  └── README.md                # Este archivo
```

---

## 🧠 Requisitos y Librerías | Dependencies and Libraries

- `pandas`, `numpy`
- `torch`, `transformers`, `scikit-learn`
- `openpyxl`, `tkinter`, `Pillow`
- `sentencepiece` (para tokenización de T5)
- Python 3.10+ recomendado

---

## ✨ Autor

Desarrollado por **Juan Moreno Borrallo** como parte de una práctica avanzada de inteligencia artificial aplicada a procesos contables.

Developed by **Juan Moreno Borrallo** as an advanced practice of artificial intelligence applied to accounting processes.
