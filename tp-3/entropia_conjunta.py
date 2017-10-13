import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import welch
import scipy as sp
import pandas as pd
import seaborn
import numpy as np
from matplotlib.colors import LogNorm
import math
import itertools
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

Samples = 201
Duration = 0.8

DELTA = (0,4)
THETA = (4,8)
ALPHA = (8,13)
BETA = (13,30)
GAMMA = (30,45) 

electrodes = [4,12,13,14,19,20,21,22,25,26,27,28,29,31,32,33,34,35]
SUJETOS_P = ["/Users/lmatayoshi/Desktop/EEG/P" + str(0) + str(i) + ".mat" for i in range(1,10)]
SUJETOS_P.append("/Users/lmatayoshi/Desktop/EEG/P10.mat")

SUJETOS_S = ["/Users/lmatayoshi/Desktop/EEG/S" + str(0) + str(i) + ".mat" for i in range(1,10)]
SUJETOS_S.append("/Users/lmatayoshi/Desktop/EEG/S10.mat")

def calculate_bin_max_min(matrix_sujeto):
    cant_electrodos = matrix_sujeto.shape[1]
    x25 = np.zeros(cant_electrodos)
    x75 = np.zeros(cant_electrodos)
    maximum = np.zeros(cant_electrodos)
    minimum = np.zeros(cant_electrodos)
    
    for i in range(0,cant_electrodos):
        electrodo_actual = np.copy(matrix_sujeto[:, i, :])
        np.sort(electrodo_actual, axis=None)

        maximum[i] = electrodo_actual.max()
        minimum[i] = electrodo_actual.min()
        x75[i] = np.percentile(electrodo_actual, 75)
        x25[i] = np.percentile(electrodo_actual, 25)
        
    return np.max(maximum), np.min(minimum), np.mean(x75), np.mean(x25)

def calculate_tbin(maximum, minimum, x75, x25, matrix_shape):
    n_instances = matrix_shape[0] * matrix_shape[2]
    return 2 * (x75-x25) / (math.pow(n_instances,1.0/3))

# numpy arange step
def calculate_probabilities(electrodo_matrix, max_val, min_val, t_bin):
    bins = np.arange(min_val, max_val, t_bin)
    acum = np.zeros(bins.shape[0] - 1)
    for i in range(0, electrodo_matrix.shape[0]):
        hist, _ = np.histogram(electrodo_matrix[i,:], bins=bins)
        acum = acum + hist
    return (acum / float(electrodo_matrix.shape[0] * electrodo_matrix.shape[1]))

def calculate_entropia(proba_x, cant_epochs, cant_muestras):
    res=0
    for i in range(0,len(proba_x)):
        if proba_x[i] > 0:
            res = res + (proba_x[i]/(cant_epochs*cant_muestras))*math.log((proba_x[i]/(cant_epochs*cant_muestras)),10)
    return -res

def entropias_electrodos_por_sujeto(filename):
    sujeto = sio.loadmat(filename)
    matrix_sujeto = sujeto['data']
    cant_epochs = matrix_sujeto.shape[0]
    #cant_electrodos = matrix_sujeto.shape[1]
    cant_muestras = matrix_sujeto.shape[2] 
    
    max_value, min_value, x75, x25 = calculate_bin_max_min(matrix_sujeto)
    t_bin = calculate_tbin(max_value, min_value, x75, x25, matrix_sujeto.shape)
    
    global electrodes

    entropia_por_electrodo = []
    for i in electrodes:
        electrodo_i_matrix = matrix_sujeto[:, i, :]
        #t_bin = calculate_tbin(maximum_values[i], minimum_values[i], x75_values[i], x25_values[i])
        probabilities = calculate_probabilities(electrodo_i_matrix, max_value, min_value, t_bin)
        entropia_por_electrodo.append(calculate_entropia(probabilities, cant_epochs, cant_muestras))
    return entropia_por_electrodo

def calculate_entropias_group():
    entropias_group_mean = np.zeros((2,10))
    entropias_group_std = np.zeros((2,10))
    for i in range(0,len(SUJETOS_S)):
        res_P = entropias_electrodos_por_sujeto(SUJETOS_P[i])
        entropias_group_mean[0][i]=np.mean(res_P)
        entropias_group_std[0][i]=np.std(res_P)
        res_S = entropias_electrodos_por_sujeto(SUJETOS_S[i])
        entropias_group_mean[1][i]=np.mean(res_S)
        entropias_group_std[1][i]=np.std(res_S)
    return entropias_group_mean, entropias_group_std

def calculate_serie(electrodo_matrix, electrodo_serie, min_val, t_bin, cant_bins):
    tam_fila = electrodo_matrix.shape[0] #cant_epochs
    tam_columna = electrodo_matrix.shape[1] #cant_muestras

    vec = np.zeros(cant_bins)
    a = 0
    
    for i in range(0,tam_fila):
        for j in range(0,tam_columna):
            bin_index = int((electrodo_matrix[i][j]-min_val)/t_bin)
            vec[bin_index]=vec[bin_index]+1
            electrodo_serie[i][j] = ord('A')+bin_index
            a=a+1

def probabilidad_conjunta(matrix_serie_1,matrix_serie_2, cant_bins):
    
    mat_PConjunta = np.zeros((cant_bins,cant_bins))

    tam_fila_serie = matrix_serie_1.shape[0]
    tam_columna_serie = matrix_serie_1.shape[1]

    size_serie = tam_fila_serie*tam_columna_serie
    
    for i in range(0,tam_fila_serie):
        for j in range(0,tam_columna_serie):
            index_f = int(matrix_serie_1[i][j]-ord('A'))
            index_c = int(matrix_serie_2[i][j]-ord('A'))
            mat_PConjunta[index_f][index_c]=mat_PConjunta[index_f][index_c]+1
            
    for i in range(0,cant_bins):
        for j in range(0,cant_bins):
            mat_PConjunta[i][j]=mat_PConjunta[i][j]/size_serie
            
    return mat_PConjunta

def calculate_entropia_conjunta(matrix_serie_1,matrix_serie_2,cant_bins):

    mat_PConjunta = probabilidad_conjunta(matrix_serie_1,matrix_serie_2,cant_bins)
    
    res=0
    for h in range(0,mat_PConjunta.shape[0]):
        for k in range(0,mat_PConjunta.shape[1]):
            if(mat_PConjunta[h][k]!=0):
                res=res+(mat_PConjunta[h][k])*math.log(mat_PConjunta[h][k])
    return -res

electrodes_grupo_1 = [31, 32, 35]
electrodes_grupo_2 = [33, 26, 19, 27, 21, 20]
electrodes_grupo_3 = [34, 28, 29, 22]
electrodes_grupo_4 = [12, 13, 14, 4, 5]

electrode_groups = [electrodes_grupo_1, electrodes_grupo_2, electrodes_grupo_3, electrodes_grupo_4]

for sujeto in SUJETOS_S:
	p01 = sio.loadmat(sujeto)
	matrix_p01 = p01['data']
	#epoch x electrodos x muestras

	#Shapes
	cant_epochs = matrix_p01.shape[0]
	cant_electrodos = matrix_p01.shape[1]
	cant_muestras = matrix_p01.shape[2]

	#Bins
	max_value, min_value, x75, x25 = calculate_bin_max_min(matrix_p01)
	t_bin = calculate_tbin(max_value, min_value, x75, x25, matrix_p01.shape)
	cant_bins = int((max_value-min_value)/t_bin)+1

	#Matrices
	array_electrodos_matrix = []
	array_electrodos_serie = []

	for g in electrode_groups:
	    array_electrodos_matrix.append(np.mean(matrix_p01[:, g, :], axis=1))

	for i in range(0,len(electrode_groups)):
	    array_electrodos_serie.append(np.zeros((cant_epochs, cant_muestras)))

	#calculamos las series
	for i in range(0,len(electrode_groups)):
	    calculate_serie(array_electrodos_matrix[i],array_electrodos_serie[i], min_value, t_bin, cant_bins)

	#electrodos tomados de a 2
	electrodos_tomados_de_a_2 = list(itertools.combinations(range(0,len(electrode_groups)), 2))
	electrodos_tomados_de_a_2 = [list(e) for e in electrodos_tomados_de_a_2]

	# El calculo de la entropia conjunta para la combinatoria tarda bastante en ejecutarse (5-10 min)
	array_entropia_conjunta = []
	for par in electrodos_tomados_de_a_2:
	    array_entropia_conjunta.append(calculate_entropia_conjunta(array_electrodos_serie[par[0]],array_electrodos_serie[par[1]],cant_bins))

	# array_entropia_conjunta_P = np.copy(array_entropia_conjunta)
	print sujeto + "\n"
	print "mean: " + str(np.mean(array_entropia_conjunta)) + "\n"
	print "std: " + str(np.std(array_entropia_conjunta))
	print "-------------------------------------------------" + "\n"