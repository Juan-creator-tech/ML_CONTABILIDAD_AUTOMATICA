import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
import json
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import ttk
from PIL import Image, ImageTk  # Requiere: pip install pillow

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
    movimientos = pd.read_excel("../data/banco/Movimientos_bancarios.xls")
    plan_contable = pd.read_excel("../data/contabilidad/plan_contable.xlsx")
    registro_facturacion = pd.read_excel("../data/facturacion/Registro_facturacion.xlsx")
    diario_contable = pd.read_excel("../data/contabilidad/diario_contable.xlsx")
    return movimientos, plan_contable, registro_facturacion, diario_contable

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

def crear_dataset_asientos(diario_contable):
    """
    A partir del diario contable se agrupan las líneas que pertenecen a un mismo asiento.
    Se genera un par (input, output) en el que:
      - Input: concatenación de las descripciones de cada línea del asiento.
      - Output: representación estructurada del asiento en formato JSON.
    """
    dataset_pairs = []
    for asiento, group in diario_contable.groupby("Asiento"):
        input_desc = " ".join(group["Descripcion"].astype(str).tolist())
        lineas = []
        for _, row in group.iterrows():
            linea = {
                "Cuenta": row["Cuenta"],
                "Debe": row["Debe"],
                "Haber": row["Haber"],
                "Descripcion": row["Descripcion"]
            }
            lineas.append(linea)
        salida = {"Asiento": asiento, "Lineas": lineas}
        salida_str = json.dumps(salida, ensure_ascii=False)
        dataset_pairs.append((input_desc, salida_str))
    return dataset_pairs

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
# 4. Entrenamiento del Modelo Encoder-Decoder (T5)
# =============================================================================

def entrenar_modelo(dataset, model, tokenizer, epochs=3, batch_size=4, learning_rate=5e-5, progress_callback=None):
    """
    Entrena el modelo T5 usando el dataset de asientos. Se utiliza un loop de
    entrenamiento básico para fines demostrativos.
    El parámetro progress_callback(epoch, total_epochs) permite actualizar una barra de progreso.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        log_message(f"Epoch {epoch+1}/{epochs}")
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            log_message(f"  Loss: {loss.item():.4f}")
        if progress_callback:
            progress_callback(epoch+1, epochs)
    return model

# =============================================================================
# 5. Generación e Inferencia de Asientos Contables
# =============================================================================

def generar_asiento(model, tokenizer, input_text, max_length=512):
    """
    A partir de la descripción de un movimiento bancario, se utiliza el modelo
    entrenado para generar el asiento contable correspondiente (en formato JSON).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    input_encoding = tokenizer.encode_plus(
        input_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = input_encoding["input_ids"].to(device)
    attention_mask = input_encoding["attention_mask"].to(device)
    generated_ids = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   max_length=max_length)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

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
    """
    nuevas_lineas = []
    for asiento_str in asientos_generados:
        asiento = json.loads(asiento_str)
        asiento_id = asiento.get("Asiento", "Nuevo")
        for linea in asiento.get("Lineas", []):
            fila = {
                "Punteo": 0,
                "Cuenta": linea.get("Cuenta"),
                "Fecha": "",  # Aquí se puede asignar la fecha correspondiente
                "Asiento": asiento_id,
                "Documento": "",
                "Descripcion": linea.get("Descripcion"),
                "Diario": 1,
                "Canal": 1,
                "Moneda": 0,
                "Debe": linea.get("Debe"),
                "Haber": linea.get("Haber"),
                "Cambio": 1
            }
            nuevas_lineas.append(fila)
    df_nuevos_asientos = pd.DataFrame(nuevas_lineas)
    df_nuevos_asientos.to_excel("../result/asientos_generados.xlsx", index=False)
    diario_actualizado = pd.concat([diario_contable, df_nuevos_asientos], ignore_index=True)
    diario_actualizado.to_excel("../result/diario_contable_actualizado.xlsx", index=False)
    log_message("Exportación completada: se han guardado 'asientos_generados.xlsx' y 'diario_contable_actualizado.xlsx'.")
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

def actualizar_barra(barra, valor):
    """Actualiza la barra de progreso de forma segura en el hilo principal."""
    barra["value"] = valor
    root.update_idletasks()

# =============================================================================
# Funciones para Integración con la GUI
# =============================================================================

def reentrenar_modelo_gui():
    """
    Función invocada desde la interfaz para reentrenar el modelo.
    Se carga el diario contable, se genera el dataset, se inicializa el tokenizer y modelo,
    y se entrena el modelo. Se actualiza la barra de progreso del entrenamiento.
    """
    global trained_model, trained_tokenizer
    try:
        log_message("Iniciando reentrenamiento del modelo...")
        # Reiniciar barra de progreso de entrenamiento
        actualizar_barra(progress_bar_training, 0)
        movimientos, plan_contable, registro_facturacion, diario_contable = leer_datos()
        dataset_pairs = crear_dataset_asientos(diario_contable)
        if not dataset_pairs:
            log_message("No se han generado pares de entrenamiento. Verifica el diario contable.")
            return
        log_message("Ejemplo de par de entrenamiento:")
        log_message("Entrada: " + dataset_pairs[0][0])
        log_message("Salida: " + dataset_pairs[0][1])
        trained_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        asientos_dataset = AsientosDataset(dataset_pairs, trained_tokenizer)
        # Función callback para actualizar la barra de progreso
        def progress_callback(epoch, total):
            porcentaje = (epoch / total) * 100
            actualizar_barra(progress_bar_training, porcentaje)
        trained_model = entrenar_modelo(asientos_dataset, model, trained_tokenizer, epochs=3, batch_size=2, progress_callback=progress_callback)
        log_message("Reentrenamiento completado correctamente.")
        messagebox.showinfo("Modelo", "El modelo se ha reentrenado correctamente.")
        actualizar_barra(progress_bar_training, 100)
    except Exception as e:
        log_message(f"Error en el reentrenamiento: {str(e)}")
        messagebox.showerror("Error", f"Error en el reentrenamiento: {str(e)}")

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
        movimientos, plan_contable, registro_facturacion, diario_contable = leer_datos()
        movimientos = preprocesar_movimientos(movimientos)
        total_movimientos = len(movimientos)
        asientos_generados = []
        for index, row in movimientos.iterrows():
            input_text = row["input_text"]
            asiento_generado = generar_asiento(trained_model, trained_tokenizer, input_text)
            log_message(f"Asiento generado para movimiento {index}:")
            log_message(asiento_generado)
            if validar_asiento(asiento_generado, plan_contable):
                asientos_generados.append(asiento_generado)
            else:
                log_message("Asiento descartado por no pasar la validación.")
            # Actualizar la barra de progreso de procesamiento
            porcentaje = ((index + 1) / total_movimientos) * 100
            actualizar_barra(progress_bar_processing, porcentaje)
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

# =============================================================================
# Interfaz Gráfica con Tkinter y fondo de imagen
# =============================================================================

root = tk.Tk()
root.title("Automatización Contable y ML")
root.geometry("800x600")

# Cargar la imagen de fondo (asegúrate de tener "background.png" en el directorio)
try:
    bg_image = Image.open("../img/background.png")
    bg_image = bg_image.resize((800, 600), Image.ANTIALIAS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    background_label = tk.Label(root, image=bg_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print("No se pudo cargar la imagen de fondo:", e)

# Crear un frame semi-transparente para los controles (opcional)
control_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
control_frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.9)

# Botón para reentrenar el modelo
btn_reentrenar = tk.Button(control_frame, text="Reentrenar Modelo", width=25,
                           command=lambda: run_threaded(reentrenar_modelo_gui))
btn_reentrenar.pack(pady=10)

# Barra de progreso para el entrenamiento
lbl_train = tk.Label(control_frame, text="Progreso Entrenamiento:")
lbl_train.pack()
progress_bar_training = ttk.Progressbar(control_frame, orient="horizontal", mode="determinate", maximum=100, length=400)
progress_bar_training.pack(pady=5)

# Botón para procesar movimientos bancarios
btn_procesar = tk.Button(control_frame, text="Procesar Movimientos Bancarios", width=25,
                         command=lambda: run_threaded(procesar_movimientos_gui))
btn_procesar.pack(pady=10)

# Barra de progreso para el procesamiento
lbl_process = tk.Label(control_frame, text="Progreso Procesamiento:")
lbl_process.pack()
progress_bar_processing = ttk.Progressbar(control_frame, orient="horizontal", mode="determinate", maximum=100, length=400)
progress_bar_processing.pack(pady=5)

# Botón para limpiar el log
def limpiar_log():
    log_text.delete("1.0", tk.END)

btn_limpiar = tk.Button(control_frame, text="Limpiar Log", width=25, command=limpiar_log)
btn_limpiar.pack(pady=5)

# Área de log para visualizar mensajes
log_text = scrolledtext.ScrolledText(control_frame, width=80, height=15)
log_text.pack(pady=10)

# Botón de salida
btn_salir = tk.Button(control_frame, text="Salir", width=25, command=root.quit)
btn_salir.pack(pady=10)

# Ejecutar la aplicación
root.mainloop()
