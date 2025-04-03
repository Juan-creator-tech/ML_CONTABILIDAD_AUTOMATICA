import pandas as pd
import os
import re

carpeta_base = r"G:\Mi unidad\OPERACIONES CABIEDES\ADMINISTRACIÓN COMÚN\CONTABILIDAD\REGISTRO CONTABLE"
archivo_salida = r"D:\Git\ML_CONTABILIDAD_AUTOMATICA\src\data_sample\contabilidad\diario_contable.xlsx"

archivos_encontrados = []
for root, dirs, files in os.walk(carpeta_base):
    for filename in files:
        if re.match(r"^\d{4}_BUENO\.xlsx$", filename):
            archivos_encontrados.append(os.path.join(root, filename))

archivos_encontrados.sort()

if not archivos_encontrados:
    raise FileNotFoundError("❌ No se encontraron archivos con el patrón esperado.")

dataframes = []
columnas_base = None

for i, ruta in enumerate(archivos_encontrados):
    try:
        # Leer solo las primeras 12 columnas
        df = pd.read_excel(ruta, header=0, usecols=range(12))

        if i == 0:
            columnas_base = df.columns
            dataframes.append(df)
        else:
            df.columns = columnas_base  # Igualamos columnas a las del primero
            dataframes.append(df)
    except Exception as e:
        print(f"⚠️ Error leyendo {ruta}: {e}")

# Concatenar todos los DataFrames
diario_unificado = pd.concat(dataframes, ignore_index=True)

# Guardar el archivo final
diario_unificado.to_excel(archivo_salida, index=False)
print(f"✅ Archivo '{archivo_salida}' generado con {len(diario_unificado)} líneas desde {len(dataframes)} archivos.")
