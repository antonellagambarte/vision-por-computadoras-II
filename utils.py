import gdown
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import random
import cv2
import numpy as np
import hashlib

def get_color_proportions(image_path):
    """
    Cálcula la proporción de píxeles verdes y amarillos en una imagen.
    """
    try:
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            # Maneja la falla en la carga de una imagen
            return 0.0, 0.0

        # Cambiar el espacio de color de BGR a HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Definimos un rango de amarillo
        # Es subjetivo pero puede refinarse si se encuentra útil la información
        lower_yellow = np.array([20, 100, 100])  # Hue, Saturation, Value
        upper_yellow = np.array([40, 255, 255])

        # Rangos para verde
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Creamos la máscara para los dos colores
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Calculo el total de pixeles no-negros
        # obtamos por esta opción porque trabajemos sobre las imagenes segmentadas
        # Un pixel se considera no-negro cuando cuando la suma de los canales BGR no dan 0
        non_black_pixels_mask = (img[:,:,0] != 0) | (img[:,:,1] != 0) | (img[:,:,2] != 0)
        total_non_black_pixels = np.sum(non_black_pixels_mask)

        if total_non_black_pixels == 0:
            return 0.0, 0.0

        # Calculo el número de pixeles de cada color en la región no-negro
        yellow_pixels = np.sum(mask_yellow[non_black_pixels_mask] > 0)
        green_pixels = np.sum(mask_green[non_black_pixels_mask] > 0)

        # Calcúlo la proporción
        yellow_proportion = yellow_pixels / total_non_black_pixels
        green_proportion = green_pixels / total_non_black_pixels

        return yellow_proportion, green_proportion
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return 0.0, 0.0

def plot_rgb_distribution(image_path, title="Distribución de Canales RGB"):
    """
    Carga una imagen, separa sus canales RGB y genera histogramas para la intensidad de cada canal.

    Args:
        image_path (str): Ruta al archivo de imagen.
        title (str): Título para el gráfico de los histogramas.
    """
    if image_path is None:
        print("No se puede procesar la imagen: la ruta es nula.")
        return

    try:
        img = Image.open(image_path).convert("RGB") # Asegurarse de que la imagen sea RGB
        img_array = np.array(img)

        # Separar canales RGB
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]

        # Crear histogramas
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)

        axes[0].hist(r_channel.flatten(), bins=256, color='red', alpha=0.7)
        axes[0].set_title('Canal Rojo')
        axes[0].set_xlabel('Intensidad')
        axes[0].set_ylabel('Frecuencia')

        axes[1].hist(g_channel.flatten(), bins=256, color='green', alpha=0.7)
        axes[1].set_title('Canal Verde')
        axes[1].set_xlabel('Intensidad')
        axes[1].set_ylabel('Frecuencia')

        axes[2].hist(b_channel.flatten(), bins=256, color='blue', alpha=0.7)
        axes[2].set_title('Canal Azul')
        axes[2].set_xlabel('Intensidad')
        axes[2].set_ylabel('Frecuencia')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar layout para título
        plt.show()

        # Mostrar la imagen
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Imagen de muestra: {image_path.split('/')[-1]}")
        plt.axis('off')
        plt.show()

    except FileNotFoundError:
        print(f"Error: La imagen no se encontró en la ruta: {image_path}")
    except Exception as e:
        print(f"Ocurrió un error al procesar la imagen: {e}")

def plot_green_yellow_distribution(image_path, title="Distribución de Colores Verde y Amarillo"):
    """
    Carga una imagen, identifica píxeles verdes y amarillos y genera histogramas
    que representan su distribución de intensidad.

    Args:
        image_path (str): Ruta al archivo de imagen.
        title (str): Título para el gráfico de los histogramas.
    """
    if image_path is None:
        print("No se puede procesar la imagen: la ruta es nula.")
        return

    try:
        # --- Cargar imagen ---
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Convertir a HSV para detectar mejor colores
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # --- Definir rangos de color ---
        # Verde (aprox. 35° a 85° en H)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        # Amarillo (aprox. 20° a 35° en H)
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([35, 255, 255])

        # --- Máscaras ---
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # --- Intensidades ---
        green_intensity = hsv[:, :, 2][mask_green > 0]
        yellow_intensity = hsv[:, :, 2][mask_yellow > 0]

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=16)

        axes[0].hist(green_intensity, bins=50, color='green', alpha=0.7)
        axes[0].set_title('Distribución de Verde')
        axes[0].set_xlabel('Intensidad (V en HSV)')
        axes[0].set_ylabel('Frecuencia')

        axes[1].hist(yellow_intensity, bins=50, color='gold', alpha=0.7)
        axes[1].set_title('Distribución de Amarillo')
        axes[1].set_xlabel('Intensidad (V en HSV)')
        axes[1].set_ylabel('Frecuencia')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Mostrar imagen original con los píxeles verdes y amarillos resaltados
        highlighted = img_array.copy()
        highlighted[mask_green > 0] = [0, 255, 0]
        highlighted[mask_yellow > 0] = [255, 255, 0]

        plt.figure(figsize=(6, 6))
        plt.imshow(highlighted)
        plt.title("Verde y Amarillo detectados")
        plt.axis('off')
        plt.show()

    except FileNotFoundError:
        print(f"Error: La imagen no se encontró en la ruta: {image_path}")
    except Exception as e:
        print(f"Ocurrió un error al procesar la imagen: {e}")
