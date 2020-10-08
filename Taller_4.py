'''
Taller 4 --> Segmentación por color y Transformaciones geométricas
Leidy Carolina Pulido Feo
Eliana Andrea Romero Leon
'''

#Librerías
import cv2
import numpy as np
import sys
import os
import math

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
#import matplotlib.pyplot as plt


def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

coordenadas_imagen1 = []
coordenadas_imagen2 = []

def click_imagen1(event, x, y, flags, params):
    if event == cv2.EVENT_RBUTTONDOWN:
        coordenadas_imagen1.append([x, y])

def click_imagen2(event, x, y, flags, params):
    if event == cv2.EVENT_RBUTTONDOWN:
        coordenadas_imagen2.append([x, y])

if __name__ == '__main__':
    seleccion_punto = int(input("Escriba \n 1 --> si desea implementar el primer punto \n 2 --> si desea implementar el segundo punto \n "))
    if seleccion_punto == 1:
        """
        Primer Punto --> Segmentación por color
        """
        path = str(input("Ingrese la dirección de ubicación de la imagen \n "))   # J:/Proc.Imagenes/Imagenes/bandera.png
        select = int(input("Escriba \n 0 --> Implementar Método Kmeans \n 1 --> Implementar Método GMM \n "))

        path_file = os.path.join(path)
        image = cv2.imread(path_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #Reordenar colores para poder observar en matplotlib

        sumas = np.zeros(10, float)
        n_colors = 0   # Grupo de color deseado
        method = ['Kmeans', 'GMM']

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        image = np.array(image, dtype=np.float64) / 255     #Normalizar la imagen entre 0 y 1. Es importante para calcular las distancias

        # Load Image and transform to a 2D numpy array.
        rows, cols, ch = image.shape    # Dimensiones de la imagen
        assert ch == 3  # Declaración que supone verdadera, de lo contrario genera un mensaje de error
        image_array = np.reshape(image, (rows * cols, ch))  # Da una nueva forma a la imagen, se coloca un gran vector de donde cada pixel tiene su correspondencia de color

        print("Fitting model on a small sub-sample of the data")    #Ajuste del modelo en una pequeña submuestra de datos
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:10000]   # Selecciona 10000 pixeles de la imagen (entre más pixeles, más tiempo)

        for i_cluster in range(10):
            n_colors = i_cluster + 1
            if method[select] == 'GMM':
                model = GMM(n_components=n_colors).fit(image_array_sample)  # Método GMM - Encuentra n número de gausianas basada en los pixeles deseados
            else:
                model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)     # Método Kmeans
            print("done in %0.3fs." % (time() - t0))

            t0 = time()
            if method[select] == 'GMM':
                print("Predicting color indices on the full image (GMM)")
                labels = model.predict(image_array)  # Se hallan las etiquetas correspondiente a cada imagen
                centers = model.means_  # Recupera los centros de las gaussianas.
            else:
                print("Predicting color indices on the full image (Kmeans)")
                labels = model.predict(image_array)  # Se hallan las etiquetas correspondiente a cada imagen
                centers = model.cluster_centers_  # Recupera los centros de los cluster
            print("done in %0.3fs." % (time() - t0))

            # SUMA DE DISTANCIAS INTRA-CLUSTER

            dist = np.zeros(n_colors, float)
            for index_dis in range(0, image_array.shape[0]):
                dist[labels[index_dis]] += np.sqrt(((image_array[index_dis, 0] - centers[labels[index_dis], 0])**2) +
                                                   ((image_array[index_dis, 1] - centers[labels[index_dis], 1])**2) +
                                                   ((image_array[index_dis, 2] - centers[labels[index_dis], 2])**2))
            sumas[i_cluster] = np.sum(dist)

        #Gráfica suma de distancias intra-cluster vs n_color
        plt.figure(1)
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (sumas))
        plt.xlabel('Número de cluster')
        plt.ylabel('Distancias Intra-cluster')
        if  select == 0:
            plt.title('Distancias Intra-cluster - Método Kmeans')
        else:
            plt.title('Distancias Intra-cluster - Método GMM')

        # Display all results, alongside original image
        plt.figure(2)
        plt.clf()
        plt.axis('off')
        plt.title('Original image')
        plt.imshow(image)

        plt.figure(3)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} colors, method={})'.format(n_colors, method[select]))
        plt.imshow(recreate_image(centers, labels, rows, cols))
        plt.show()

    else:
        """
        Segundo Punto --> Transformaciones geométricas
        """
        path = str(input("Ingrese la dirección de ubicación de la primera imagen \n "))  # J:/Proc.Imagenes/Imagenes/lena.png
        path_file = os.path.join(path)
        image_1 = cv2.imread(path_file)
        cv2.namedWindow("ImagenLena")   #Nombre de la primera ventana de eventos de click

        cv2.setMouseCallback('ImagenLena', click_imagen1)

        cv2.imshow('ImagenLena', image_1)
        cv2.waitKey(0)
        print(coordenadas_imagen1)

        path = str(input("Ingrese la dirección de ubicación de la segunda imagen \n "))  # J:/Proc.Imagenes/Imagenes/lena_warped.png
        path_file = os.path.join(path)
        image_2 = cv2.imread(path_file)
        cv2.namedWindow("ImagenLena_warped")  # Nombre de la segunda ventana de eventos de click

        cv2.setMouseCallback('ImagenLena_warped', click_imagen2)

        cv2.imshow('ImagenLena_warped', image_2)
        cv2.waitKey(0)
        print(coordenadas_imagen2)

        # affine
        coordenadas_imagen1 = np.float32(coordenadas_imagen1)
        coordenadas_imagen2 = np.float32(coordenadas_imagen2)
        M_affine = cv2.getAffineTransform(coordenadas_imagen1, coordenadas_imagen2)  # Matriz afín.
        image_affine = cv2.warpAffine(image_1, M_affine, image_1.shape[:2])  # Transformación afín.
        print(M_affine)

        #COMPUTO DE PARÁMETROS

        #Parámetros de escala
        sx = np.sqrt((M_affine[0, 0]**2) + (M_affine[1, 0]**2))
        sy = np.sqrt((M_affine[0, 1]**2) + (M_affine[1, 1]**2))

        #Parámetro de rotación
        theta = -np.arctan(M_affine[1, 0]/M_affine[0, 0])

        #Parámetro de traslación
        tx = ((M_affine[0, 2] * np.cos(theta)) - (M_affine[1, 2] * np.sin(theta))) / sx
        ty = ((M_affine[0, 2] * np.sin(theta)) + (M_affine[1, 2] * np.cos(theta))) / sy

        #Transformación de similitud
        # similarity - Transformación que realiza rotación, escalamiento y rotación.
        M_sim = np.float32([[sx * np.cos(theta), -np.sin(theta), tx],
                            [np.sin(theta), sy * np.cos(theta), ty]])  # Matriz de similitud.
        image_similarity = cv2.warpAffine(image_1, M_sim, image_1.shape[:2])  # Transformación de similitud.
        print(M_sim)

        cv2.imshow("Image_affine", image_affine)    #Visualización de imagen afin
        cv2.imshow("Image_similarity", image_similarity)    #Visualización  de imagen de similitud
        cv2.waitKey(0)

        #Cálculo de error

        M_sim_puntos = np.append(M_sim, np.array([[0,0,1]]), axis = 0)  #Matriz de operaciones, basada en la matriz de similitud
        puntos_trans = coordenadas_imagen1.transpose()      #Transponer el vector de coordenadas de la imagen 1
        Matriz_puntos = np.append(puntos_trans, np.array([[1,1,1]]), axis = 0)  #Matriz homogenea de coordenadas
        Transform = np.matmul(M_sim_puntos, Matriz_puntos)  #Multiplicación de las matrices
        Transform_final = Transform[:-1, :].transpose()
        error1 = np.linalg.norm(Transform_final - coordenadas_imagen2, axis = 1)    #Cálculo de la norma del error
        print("Error = ", error1)
