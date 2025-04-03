import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
import json
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk  # Requiere: pip install pillow
import os
import warnings
from IPython.display import Image, display
import shutil
import re

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# 1. Lectura y Preprocesamiento de los archivos Excel
# =============================================================================

def leer_datos():
    """
    Lee los archivos Excel del proyecto:
      - Movimientos bancarios
      - Plan contable
      - Registro de facturación
      - Diario contable
    """
    movimientos = pd.read_excel("../data_sample/bancos/Movimientos_bancarios_preprocesado.xlsx")
    plan_contable = pd.read_excel("../data_sample/contabilidad/plan_contable.xlsx")
    diario_contable = pd.read_excel("../data_sample/contabilidad/diario_contable.xlsx")
    return movimientos, plan_contable, diario_contable

def preprocesar_movimientos(movimientos):
    """
    Para cada movimiento bancario se genera una cadena de texto concatenando
    las columnas: Fecha, Movimiento, Importe y Más datos.
    """
    movimientos['input_text'] = (
        movimientos['Fecha'].astype(str) + " " +
        movimientos['Movimiento'].astype(str) + " " +
        movimientos['Importe'].astype(str) + " " +
        movimientos['Más datos'].astype(str)
    )
    return movimientos

# =============================================================================
# 2. Creación del Dataset de Entrenamiento a partir del Diario Contable
# =============================================================================

def generar_pares_movimiento_asiento(movimientos: pd.DataFrame, diario_contable: pd.DataFrame, tolerancia=0.01):
    """
    Relaciona automáticamente movimientos bancarios con asientos contables
    según coincidencia de fecha e importe (teniendo en cuenta signo).
    Devuelve una lista de pares (input_text del movimiento, asiento en formato JSON).
    """
    pares = []

    # Agrupar el diario por asiento
    diario_agrupado = diario_contable.groupby("Asiento")

    # Crear resumen por asiento
    resumen_asientos = []
    for asiento_id, grupo in diario_agrupado:
        try:
            fecha = pd.to_datetime(grupo["Fecha"].iloc[0], dayfirst=True)
        except Exception:
            continue  # saltar si la fecha no es válida
        total_debe = grupo["Debe"].sum()
        total_haber = grupo["Haber"].sum()
        resumen_asientos.append({
            "Asiento": asiento_id,
            "Fecha": fecha,
            "TotalDebe": total_debe,
            "TotalHaber": total_haber,
            "Lineas": grupo
        })

    # Recorrer movimientos bancarios
    for _, mov in movimientos.iterrows():
        try:
            mov_fecha = pd.to_datetime(mov["Fecha"], dayfirst=True)
            mov_importe = float(mov["Importe"])
        except Exception:
            continue

        mov_texto = mov["input_text"]
        importe_abs = abs(mov_importe)

        for resumen in resumen_asientos:
            misma_fecha = resumen["Fecha"].date() == mov_fecha.date()

            if mov_importe >= 0:
                match_importe = abs(resumen["TotalDebe"] - importe_abs) < tolerancia
            else:
                match_importe = abs(resumen["TotalHaber"] - importe_abs) < tolerancia

            if misma_fecha and match_importe:
                lineas_json = []
                for _, row in resumen["Lineas"].iterrows():
                    linea = {
                        "Cuenta": row["Cuenta"],
                        "Debe": row["Debe"],
                        "Haber": row["Haber"],
                        "Descripcion": row["Descripcion"]
                    }
                    lineas_json.append(linea)

                asiento_json = json.dumps({
                    "Asiento": resumen["Asiento"],
                    "Lineas": lineas_json
                }, ensure_ascii=False)

                pares.append((mov_texto, asiento_json))
                break  # solo un match por movimiento

    return pares

def exportar_pares_entrenamiento(pares, ruta_salida="../result/pares_entrenamiento.xlsx"):
    """
    Exporta los pares (input_text, asiento_json) a un archivo Excel para trazabilidad.
    """
    df = pd.DataFrame(pares, columns=["InputText", "AsientoJSON"])
    df.to_excel(ruta_salida, index=False)
    print(f"✅ Pares de entrenamiento exportados a: {ruta_salida}")

# =============================================================================
# 3. Definición del Dataset para PyTorch
# =============================================================================

class AsientosDataset(Dataset):
    """
    Dataset personalizado que recibe una lista de pares (input_text, target_text)
    y utiliza el tokenizer de T5 para codificar los textos de entrada y salida.
    """
    def __init__(self, pairs, tokenizer, max_input_length=512, max_output_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]
        input_encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }

# =============================================================================
# 5. Generación e Inferencia de Asientos Contables
# =============================================================================

def generar_asiento(model, tokenizer, input_text, max_length=512):
    """
    Genera un asiento contable a partir de un texto de entrada usando el modelo T5.
    Intenta corregir el formato si la salida no es un JSON válido.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Tokenizar entrada
    input_encoding = tokenizer.encode_plus(
        input_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = input_encoding["input_ids"].to(device)
    attention_mask = input_encoding["attention_mask"].to(device)

    # Generar texto
    generated_ids = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   max_length=max_length)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Intentar parsear directamente como JSON
    try:
        return json.dumps(json.loads(generated_text), ensure_ascii=False)
    except json.JSONDecodeError:
        pass  # Intentamos repararlo abajo

    # === Intento de reparación rápida del JSON ===
    repaired = generated_text.strip()

    # Si falta la apertura o cierre
    if not repaired.startswith("{"):
        repaired = "{" + repaired
    if not repaired.endswith("}"):
        repaired += "}"

    # Quitar comas colgantes (muy comunes)
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)

    # Reparar comillas simples por comillas dobles (si las hay)
    repaired = repaired.replace("'", '"')

    # Quitar líneas vacías
    repaired = "\n".join([line for line in repaired.splitlines() if line.strip()])

    # Reintentar parsear
    try:
        return json.dumps(json.loads(repaired), ensure_ascii=False)
    except json.JSONDecodeError:
        log_message("⚠️ No se pudo convertir la salida en JSON válido.")
        log_message(f"Salida generada (sin parsear):\n{generated_text}")
        return None

# =============================================================================
# 6. Validación de los Asientos Generados
# =============================================================================

def validar_asiento(asiento_str, plan_contable):
    """
    Valida que el asiento generado:
      - Sea un JSON válido.
      - Las cuentas generadas existan en el plan contable (solo las de 10 dígitos).
      - El asiento cuadre (la suma del Debe es igual a la del Haber).
    """
    try:
        asiento = json.loads(asiento_str)
    except json.JSONDecodeError:
        log_message("Error: La salida generada no es un JSON válido.")
        return False

    cuentas_validas = plan_contable[plan_contable["Cuenta"].apply(lambda x: len(str(x)) == 10)]["Cuenta"].astype(str).tolist()
    suma_debe = 0
    suma_haber = 0

    for linea in asiento.get("Lineas", []):
        cuenta = str(linea.get("Cuenta", ""))
        if cuenta not in cuentas_validas:
            log_message(f"Cuenta {cuenta} no es válida según el plan contable.")
            return False
        try:
            debe = float(linea.get("Debe", 0))
            haber = float(linea.get("Haber", 0))
        except ValueError:
            log_message("Error en la conversión de valores de Debe o Haber.")
            return False
        suma_debe += debe
        suma_haber += haber

    if abs(suma_debe - suma_haber) > 1e-2:
        log_message("El asiento no cuadra: suma_debe != suma_haber")
        return False

    return True

# =============================================================================
# 7. Exportación de Asientos a Excel y Actualización del Diario Contable
# =============================================================================

def exportar_asientos(asientos_generados, diario_contable):
    """
    Exporta los asientos generados a un archivo Excel con el formato del diario contable
    y actualiza el diario concatenando los nuevos asientos.
    
    Parámetro `asientos_generados`:
    Lista de tuplas (asiento_json, fecha_movimiento, input_text)
    """
    nuevas_lineas = []

    for asiento_str, fecha, input_text in asientos_generados:
        asiento = json.loads(asiento_str)
        asiento_id = asiento.get("Asiento", "Nuevo")

        for linea in asiento.get("Lineas", []):
            fila = {
                "Punteo": 0,
                "Cuenta": linea.get("Cuenta"),
                "Fecha": fecha,  # ← Aquí se asigna la fecha del movimiento original
                "Asiento": asiento_id,
                "Documento": "",
                "Descripcion": linea.get("Descripcion"),
                "Diario": 1,
                "Canal": 1,
                "Moneda": 0,
                "Debe": linea.get("Debe"),
                "Haber": linea.get("Haber"),
                "Cambio": 1,
                "Input": input_text  # ← nueva columna de trazabilidad
            }
            nuevas_lineas.append(fila)

    df_nuevos_asientos = pd.DataFrame(nuevas_lineas)
    df_nuevos_asientos.to_excel("../result/asientos_generados.xlsx", index=False)

    diario_actualizado = pd.concat([diario_contable, df_nuevos_asientos], ignore_index=True)
    diario_actualizado.to_excel("../result/diario_contable_actualizado.xlsx", index=False)

    log_message("✅ Export completed: 'asientos_generados.xlsx' and 'diario_contable_actualizado.xlsx' saved.")
    return diario_actualizado

# =============================================================================
# Variables Globales para el Modelo y Tokenizer
# =============================================================================

trained_model = None
trained_tokenizer = None

# =============================================================================
# Elementos de la GUI y funciones de actualización de log y progreso
# =============================================================================

def log_message(message):
    """Inserta un mensaje en el widget de log y lo imprime en consola."""
    log_text.insert(tk.END, message + "\n")
    log_text.see(tk.END)
    print(message)

def actualizar_porcentaje_label(label, texto):
    """Actualiza el texto de una etiqueta desde un hilo externo."""
    root.after(0, lambda: label.config(text=texto))

def actualizar_barra(barra, valor, tipo="train"):
    root.after(0, lambda: barra.config(value=valor))
    texto = f"{tipo.capitalize()}: {int(valor)}%"
    if tipo == "train" and 'porcentaje_entrenamiento' in globals():
        porcentaje_entrenamiento.set(texto)
    elif tipo == "process" and 'porcentaje_procesamiento' in globals():
        porcentaje_procesamiento.set(texto)


# =============================================================================
# Funciones para Integración con la GUI
# =============================================================================

def procesar_movimientos_gui():
    """
    Función invocada desde la interfaz para procesar los movimientos bancarios.
    Se leen los datos, se preprocesan y se utiliza el modelo entrenado para generar
    los asientos contables, actualizando la barra de progreso conforme se procesan.
    """
    global trained_model, trained_tokenizer
    try:
        if trained_model is None or trained_tokenizer is None:
            log_message("El modelo no ha sido entrenado. Reentrena el modelo antes de procesar movimientos.")
            messagebox.showwarning("Modelo no entrenado", "Reentrena el modelo antes de procesar movimientos.")
            return
        log_message("Iniciando procesamiento de movimientos bancarios...")
        actualizar_barra(progress_bar_processing, 0)
        ruta_movimientos = seleccionar_movimientos()
        if not ruta_movimientos:
            log_message("❌ No se seleccionó archivo de movimientos.")
            messagebox.showwarning("Archivo no seleccionado", "Debes seleccionar un archivo de movimientos bancarios.")
            return

        # Leer archivo seleccionado
        movimientos = pd.read_excel(ruta_movimientos)
        movimientos = preprocesar_movimientos(movimientos)
        _, plan_contable, diario_contable = leer_datos()
        total_movimientos = len(movimientos)
        asientos_generados = []  # Lista de tuplas: (asiento_json, fecha)

        for index, row in movimientos.iterrows():
            input_text = row["input_text"]
            fecha = row["Fecha"]  # ← Capturamos la fecha del movimiento original
            # ⬇️ Nueva línea para depurar la entrada
            log_message(f"Input text movimiento {index}:\n{input_text}")
            asiento_generado = generar_asiento(trained_model, trained_tokenizer, input_text)
            
            log_message(f"Asiento generado para movimiento {index}:")
            log_message(asiento_generado)

            if isinstance(asiento_generado, dict) and validar_asiento(json.dumps(asiento_generado), plan_contable):
                asientos_generados.append((json.dumps(asiento_generado), fecha, input_text))
            else:
                log_message("Asiento descartado por no pasar la validación.")
            # Actualizar la barra de progreso de procesamiento
            porcentaje = ((index + 1) / total_movimientos) * 100
            actualizar_barra(progress_bar_processing, porcentaje, tipo="process")
        if asientos_generados:
            exportar_asientos(asientos_generados, diario_contable)
            log_message("Procesamiento de movimientos completado.")
            messagebox.showinfo("Procesamiento", "Se han procesado los movimientos y exportado los asientos.")
        else:
            log_message("No se generaron asientos válidos.")
            messagebox.showwarning("Sin asientos", "No se generaron asientos válidos a partir de los movimientos.")
        actualizar_barra(progress_bar_processing, 100)
    except Exception as e:
        log_message(f"Error en el procesamiento: {str(e)}")
        messagebox.showerror("Error", f"Error en el procesamiento: {str(e)}")

def run_threaded(func):
    """Ejecuta la función pasada en un hilo separado para no bloquear la GUI."""
    threading.Thread(target=func).start()

def cargar_modelo_entrenado():
    global trained_model, trained_tokenizer
    try:
        if os.path.isdir("../models/trained_t5_model_best"):
            trained_model = T5ForConditionalGeneration.from_pretrained("../models/trained_t5_model_best")
            trained_tokenizer = T5Tokenizer.from_pretrained("../models/trained_t5_model_best")
            estado_modelo_texto.set("✅ Modelo cargado correctamente")
            estado_modelo_label.config(fg="green")
            estado_modelo_label.place(relx=0.5, rely=0.375, anchor="n")  # ¡Importante!
        else:
            estado_modelo_texto.set("⚠️ Modelo no encontrado")
            estado_modelo_label.config(fg="red")
            estado_modelo_label.place(relx=0.5, rely=0.375, anchor="n")
    except Exception as e:
        estado_modelo_texto.set("❌ Error al cargar modelo")
        estado_modelo_label.config(fg="red")
        estado_modelo_label.place(relx=0.5, rely=0.375, anchor="n")
        print(f"Error al cargar modelo: {e}")

def seleccionar_movimientos():
    """
    Abre un diálogo para que el usuario seleccione el archivo de movimientos bancarios.
    Devuelve la ruta seleccionada o None si se cancela.
    """
    ruta = filedialog.askopenfilename(
        title="Seleccionar archivo de movimientos bancarios",
        filetypes=[("Excel files", "*.xls *.xlsx")]
    )
    return ruta

def lanzar_app_procesamiento(model_path):
    """
    Carga el modelo y lanza la interfaz solo para procesar movimientos bancarios.
    """
    global trained_model, trained_tokenizer
    trained_model = T5ForConditionalGeneration.from_pretrained(model_path)
    trained_tokenizer = T5Tokenizer.from_pretrained(model_path)
    estado_modelo_texto.set("✅ Modelo cargado correctamente")
    root.mainloop()

# =============================================================================
# Interfaz Gráfica con Tkinter y fondo de imagen
# =============================================================================

import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Programa de automatización de asientos contables")
root.geometry("800x600")

# Variables para mostrar el porcentaje dinámico de progreso
porcentaje_entrenamiento = tk.StringVar(value="Entrenamiento: 0%")
porcentaje_procesamiento = tk.StringVar(value="Procesamiento: 0%")

# === Imagen de fondo ===
try:
    global bg_photo
    bg_image = Image.open("../img/background.png")
    bg_photo = ImageTk.PhotoImage(bg_image)
    background_label = tk.Label(root, image=bg_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print("No se pudo cargar la imagen de fondo:", e)

# === Botón: Procesar movimientos ===
btn_procesar = tk.Button(root, text="💼 Procesar Movimientos", width=25,
                         command=lambda: run_threaded(procesar_movimientos_gui))
btn_procesar.place(relx=0.5, rely=0.24, anchor="n")

# === Etiqueta de progreso procesamiento ===
lbl_process = tk.Label(root, textvariable=porcentaje_procesamiento,
                       font=("Arial", 10, "bold"), fg="white", bg=root["bg"])
lbl_process.place(relx=0.5, rely=0.29, anchor="n")

progress_bar_processing = ttk.Progressbar(root, orient="horizontal", mode="determinate", maximum=100, length=400)
progress_bar_processing.place(relx=0.5, rely=0.33, anchor="n")

# === Log de mensajes ===
log_text = scrolledtext.ScrolledText(root, width=95, height=12, font=("Courier", 9))
log_text.place(relx=0.5, rely=0.42, anchor="n")

# === Etiqueta de estado del modelo (inicialmente oculta o azul) ===
estado_modelo_texto = tk.StringVar(value="")
estado_modelo_label = tk.Label(root, textvariable=estado_modelo_texto,
                               font=("Arial", 11, "bold"), fg="red", bg=root["bg"])
estado_modelo_label.place(relx=0.5, rely=0.375, anchor="n")

# === Botón: Limpiar log ===
def limpiar_log():
    log_text.delete("1.0", tk.END)

btn_limpiar = tk.Button(root, text="🧹 Limpiar Log", width=20, command=limpiar_log)
btn_limpiar.place(relx=0.35, rely=0.91)

# === Botón: Salir ===
btn_salir = tk.Button(root, text="🚪 Salir", width=20, command=root.destroy)
btn_salir.place(relx=0.60, rely=0.91)

# Carga el modelo entrenado y lanza la app
lanzar_app_procesamiento(model_path="../models/trained_t5_model_best")