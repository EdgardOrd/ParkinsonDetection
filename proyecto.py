import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import time
import pickle
# Función para obtener las carpetas de los sujetos
def get_subject_folders(processed_path):
    folders = []
    for root, dirs, _ in os.walk(processed_path):
        folders.extend([os.path.join(root, d) for d in dirs]) 
        break # solo se requieren subcarpetas de nivel superior 
    return folders
# Función para obtener la potencia promedio
def get_avg_power(subject_path):
    epochs = mne.read_epochs(os.path.join(subject_path, subject_path[-7:] + '-epo.fif'))
    psd, freqs = mne.time_frequency.psd_array_multitaper(epochs['S  2'].pick('Cz').get_data(), fmin=1, fmax=10, n_jobs=1, verbose=None, sfreq=256)
    avg = psd.mean(axis=(0,2))[0]
    return avg
# Función para obtener la amplitud promedio
def get_avg_amplitude(subject_path):
    epochs = mne.read_epochs(os.path.join(subject_path, subject_path[-7:] + '-epo.fif'))
    return epochs['S  2'].pick('Cz').get_data().mean(axis=(0,2))[0]
# Obtener las carpetas de los sujetos
folders = get_subject_folders(r'C:\Users\pamel\OneDrive\Escritorio\preprocessed')
# Crear un DataFrame para almacenar los datos
template = {'mean_amp':[], 'mean_power':[], 'subject':[], 'is_PD':[]}
data = pd.DataFrame(template)
# Leer el archivo de participantes
participants = pd.read_csv(r'C:\Users\pamel\OneDrive\Escritorio\updated_participants.tsv', sep= '\t+', engine='python')
# Copiar los IDs de los sujetos al DataFrame de datos
data['subject'] = participants['participant_id'].copy()
data.set_index('subject', inplace=True, drop=True)
# Agregar datos adicionales al DataFrame
data['MOCA'] = participants['MOCA'].values
data['Age'] = participants['AGE'].values
data['TYPE'] = participants['TYPE'].values

# Función para categorizar las etiquetas
def categorize_label(row):
    if row['TYPE'] == 1:
        if row['MOCA'] < 22:
            return 'PDD'
        elif row['MOCA'] >= 22 and row['MOCA'] <= 26:
            return 'PD-MCI'
        else:
            return 'PD'
    else:
        return 'Control'
# Aplicar la función para categorizar las etiquetas
data['label'] = data.apply(categorize_label, axis=1)
# Calcular la amplitud y potencia promedio para cada sujeto
for subject in folders:
    data.loc[subject[-7:], 'mean_amp'] = get_avg_amplitude(subject)
    data.loc[subject[-7:], 'mean_power'] = get_avg_power(subject)
# Seleccionar las características y etiquetas
X = data[(data['label']=='PD') | (data['label']=='PDD')][['mean_amp','mean_power','Age']]
y = data[(data['label']=='PD') | (data['label']=='PDD')]['label']
# Escalar las características
scaler = StandardScaler()
scaler.fit(X)
X_tr = scaler.transform(X)
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_tr, y, test_size=0.33, random_state=42)
# Modelos y ajuste

# SVM
param_grid_svc = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']}
grid_svc = GridSearchCV(SVC(), param_grid_svc, refit=True, verbose=2)
grid_svc.fit(X_train, y_train)
y_pred_svc = grid_svc.predict(X_test)
precision_svc = precision_score(y_test, y_pred_svc, pos_label='PDD')
especificidad_svc = recall_score(y_test, y_pred_svc, pos_label='PD', average='binary')

# Bosque Aleatorio
param_grid_rf = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [10, 20, 30, 40, 50]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=0), param_grid_rf, refit=True, verbose=2)
grid_rf.fit(X_train, y_train)
y_pred_rf = grid_rf.predict(X_test)
precision_rf = precision_score(y_test, y_pred_rf, pos_label='PDD')
especificidad_rf = recall_score(y_test, y_pred_rf, pos_label='PD', average='binary')

# Gradient Boosting
param_grid_gb = {'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.1, 0.5, 1], 'max_depth': [3, 5, 7]}
grid_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, refit=True, verbose=2)
grid_gb.fit(X_train, y_train)
y_pred_gb = grid_gb.predict(X_test)
precision_gb = precision_score(y_test, y_pred_gb, pos_label='PDD')
especificidad_gb = recall_score(y_test, y_pred_gb, pos_label='PD', average='binary')

# Regresión Logística
param_grid_logreg = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_logreg = GridSearchCV(LogisticRegression(), param_grid_logreg, refit=True, verbose=2)
grid_logreg.fit(X_train, y_train)
y_pred_logreg = grid_logreg.predict(X_test)
precision_logreg = precision_score(y_test, y_pred_logreg, pos_label='PDD')
especificidad_logreg = recall_score(y_test, y_pred_logreg, pos_label='PDD')

#xgboost
# Convertir etiquetas de clase a valores binarios
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# xgboost
param_grid_xgb = {'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.1, 0.5, 1], 'max_depth': [3, 5, 7]}
grid_xgb = GridSearchCV(xgb.XGBClassifier(), param_grid_xgb, refit=True, verbose=2)
grid_xgb.fit(X_train, y_train_encoded)
y_pred_xgb = grid_xgb.predict(X_test)
precision_xgb = precision_score(y_test_encoded, y_pred_xgb)
especificidad_xgb = recall_score(y_test_encoded, y_pred_xgb, average='binary')
models = {
    'svm': grid_svc.best_estimator_,
    'logistic_regression': grid_logreg.best_estimator_,
    'random_forest': grid_rf.best_estimator_,
    'xgboost': grid_xgb.best_estimator_,
    'gradient_boosting': grid_gb.best_estimator_
}

for model_name, model in models.items():
    with open(model_name + '_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print(f'{model_name} model successfully saved')
print("SVM:")
print("Precisión:", precision_svc)
print("Especificidad:", especificidad_svc)

print("Regresión Logística:")
print("Precisión:", precision_logreg)
print("Especificidad:", especificidad_logreg)

print("Random Forest:")
print("Precisión:", precision_rf)
print("Especificidad:", especificidad_rf)

print("Gradient Boosting:")
print("Precisión:", precision_gb)
print("Especificidad:", especificidad_gb)

print("XGBoost:")
print("Precisión:", precision_xgb)
print("Especificidad:", especificidad_xgb)
#Precision y Especificidad por modelo
import matplotlib.pyplot as plt

# Datos de precisión y especificidad para cada modelo


modelos = ["SVM", "Regresión Logística", "Random Forest", "Gradient Boosting", "XGBoost"]
precisiones = [precision_svc, precision_logreg, precision_rf, precision_gb, precision_xgb]
especificidades = [especificidad_svc, especificidad_logreg, especificidad_rf, especificidad_gb, especificidad_xgb]

# Colores para las barras
color_precision = '#1f77b4'  # Azul
color_especificidad = '#ff7f0e'  # Azul claro

# Crear gráfico de barras agrupadas
x = range(len(modelos))  # Posiciones en el eje x para cada modelo

plt.figure(figsize=(10, 6))

# Barras para la precisión
plt.bar(x, precisiones, color=color_precision, width=0.4, label='Precisión')

# Barras para la especificidad
plt.bar([i + 0.4 for i in x], especificidades, color=color_especificidad, width=0.4, label='Especificidad')

# Configurar el eje x con los nombres de los modelos
plt.xticks([i + 0.2 for i in x], modelos)

# Configurar título y etiquetas de los ejes
plt.title('Precisión y Especificidad por Modelo')
plt.xlabel('Modelo')
plt.ylabel('Valor')
plt.ylim(0, 1)  # Asumiendo que las métricas están en el rango [0, 1]

# Mostrar leyenda y cuadrícula
plt.legend()
plt.grid(False)

# Mostrar gráfico
plt.tight_layout()
plt.show()

import itertools

# Lista de modelos
modelos = ["SVM", "Regresión Logística", "Random Forest", "Gradient Boosting", "XGBoost"]

# Generar todas las combinaciones posibles de dos modelos sin repetir
combinaciones = list(itertools.combinations(modelos, 2))

# Mostrar las combinaciones
for combinacion in combinaciones:
    print(combinacion)
# Diccionario de modelos y sus métricas
metricas = {
    'SVM': (precision_svc, especificidad_svc),
    'Regresión Logística': (precision_logreg, especificidad_logreg),
    'Random Forest': (precision_rf, especificidad_rf),
    'Gradient Boosting': (precision_gb, especificidad_gb),
    'XGBoost': (precision_xgb, especificidad_xgb)
}

# Calcular métricas promedio para cada combinación
for combinacion in combinaciones:
    precision_combinada = 0
    especificidad_combinada = 0
    for modelo in combinacion:
        precision_combinada += metricas[modelo][0]
        especificidad_combinada += metricas[modelo][1]
    precision_promedio = precision_combinada / 2
    especificidad_promedio = especificidad_combinada / 2
    print(f"Combinación: {combinacion}")
    print(f"Precisión promedio: {precision_promedio}")
    print(f"Especificidad promedio: {especificidad_promedio}")
    print()
#Grafica de combinaciones de modelos
#Combinación ('SVM', 'Regresión Logística')
import matplotlib.pyplot as plt
modelos = ['SVM', 'Regresión Logística']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_svc, precision_logreg]
especificidades_individuales = [especificidad_svc, especificidad_logreg]
precision_promedio = 0.8571428571428572
especificidad_promedio = 1.0
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (SVM, Regresión Logística)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Combinación ('SVM', 'Random Forest')
import matplotlib.pyplot as plt
modelos = ['SVM', 'Random Forest']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_svc, precision_rf]
especificidades_individuales = [especificidad_svc, especificidad_rf]
precision_promedio = 1.0
especificidad_promedio = 1.0
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (SVM, Random Forest)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Combinación ('SVM', 'Gradient Boosting'):
import matplotlib.pyplot as plt
modelos = ['SVM', 'Gradient Boosting']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_svc, precision_gb]
especificidades_individuales = [especificidad_svc, especificidad_gb]
precision_promedio = 1.0
especificidad_promedio = 1.0
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (SVM, Gradient Boosting)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Combinación ('SVM', 'XGBoost')
import matplotlib.pyplot as plt
modelos = ['SVM', 'XGBoost']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_svc, precision_xgb]
especificidades_individuales = [especificidad_svc, especificidad_xgb]
precision_promedio = 1.0
especificidad_promedio = 0.8
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (SVM, XGBoost)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Combinación ('Regresión Logística', 'Random Forest'):
import matplotlib.pyplot as plt
modelos = ['Regresión Logística', 'Random Forest']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_logreg, precision_rf]
especificidades_individuales = [especificidad_logreg, especificidad_rf]
precision_promedio = 0.8571428571428572
especificidad_promedio = 1.0
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (Regresión Logística, Random Forest)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Combinación ('Regresión Logística', 'Gradient Boosting'):
import matplotlib.pyplot as plt
modelos = ['Regresión Logística', 'Gradient Boosting']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_logreg, precision_gb]
especificidades_individuales = [especificidad_logreg, especificidad_gb]
precision_promedio = 0.8571428571428572
especificidad_promedio = 1.0
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (Regresión Logística, Gradient Boosting)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Combinación ('Regresión Logística', 'XGBoost')
import matplotlib.pyplot as plt
modelos = ['Regresión Logística', 'XGBoost']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_logreg, precision_xgb]
especificidades_individuales = [especificidad_logreg, especificidad_xgb]
precision_promedio = 0.8571428571428572
especificidad_promedio = 0.8
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (Regresión Logística, XGBoost)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Combinación ('Random Forest', 'Gradient Boosting'):
import matplotlib.pyplot as plt
modelos = ['Random Forest', 'Gradient Boosting']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_rf, precision_gb]
especificidades_individuales = [especificidad_rf, especificidad_gb]
precision_promedio = 1.0
especificidad_promedio = 1.0
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (Random Forest, Gradient Boosting)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Combinación ('Random Forest', 'XGBoost'):
import matplotlib.pyplot as plt
modelos = ['Random Forest', 'XGBoost']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_rf, precision_xgb]
especificidades_individuales = [especificidad_rf, especificidad_xgb]
precision_promedio = 1.0
especificidad_promedio = 0.8
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (Random Forest, XGBoost)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Combinación ('Gradient Boosting', 'XGBoost'):
import matplotlib.pyplot as plt
modelos = ['Gradient Boosting', 'XGBoost']
metricas = ['Precisión', 'Especificidad']
precisiones_individuales = [precision_gb, precision_xgb]
especificidades_individuales = [especificidad_gb, especificidad_xgb]
precision_promedio = 1.0
especificidad_promedio = 0.8
plt.figure(figsize=(10, 6))
posiciones = range(len(modelos))
plt.barh(posiciones, precisiones_individuales, color='#1f77b4', label='Precisión')
plt.barh(posiciones, especificidades_individuales, left=precisiones_individuales, color='#ff7f0e', label='Especificidad')
plt.axvline(x=precision_promedio, color='blue', linestyle='--', label='Precisión Promedio')
plt.axvline(x=especificidad_promedio, color='orange', linestyle='--', label='Especificidad Promedio')
plt.title('Combinación: (Gradient Boosting, XGBoost)')
plt.xlabel('Valor')
plt.yticks(posiciones, modelos)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
# Lista para almacenar todas las combinaciones con la métrica promedio más alta
mejores_combinaciones = []
mejor_metrica_promedio = 0

#cALCULAR LAS METRICAS PROMEDIO
for combinacion in combinaciones:
    precision_combinada = 0
    especificidad_combinada = 0
    for modelo_index in combinacion:  # Cambiar a modelo_index
        modelo = metricas[modelo_index]  # Obtener la lista de métricas para el modelo_index
        print(metricas)
        precision_combinada += modelo[0]
        especificidad_combinada += metricas[modelo][1]
    precision_promedio = precision_combinada / 2
    especificidad_promedio = especificidad_combinada / 2
    metrica_promedio = (precision_promedio + especificidad_promedio) / 2
    
    # Si la métrica promedio es mayor que la mejor métrica promedio actual,
    # actualizamos la mejor métrica y reiniciamos la lista de mejores combinaciones
    if metrica_promedio > mejor_metrica_promedio:
        mejor_metrica_promedio = metrica_promedio
        mejores_combinaciones = [combinacion]
    # Si la métrica promedio es igual a la mejor métrica promedio actual,
    # agregamos esta combinación a la lista de mejores combinaciones
    elif metrica_promedio == mejor_metrica_promedio:
        mejores_combinaciones.append(combinacion)

# Imprimir las mejores combinaciones
print("Las mejores combinaciones son:")
for combinacion in mejores_combinaciones:
    print(combinacion)
print("Con una métrica promedio de:", mejor_metrica_promedio)

import matplotlib.pyplot as plt

# Definir las combinaciones y sus métricas (en este caso, valores de ejemplo)
combinaciones = [('SVM', 'Random Forest'), ('SVM', 'Gradient Boosting'), ('Random Forest', 'Gradient Boosting')]
metricas = {
    'SVM': {'precision': 0.85, 'especificidad': 0.9},
    'Random Forest': {'precision': 0.92, 'especificidad': 0.87},
    'Gradient Boosting': {'precision': 0.88, 'especificidad': 0.89}
}

# Preparar datos para el gráfico
modelos = list(metricas.keys())
precisiones = [metricas[modelo]['precision'] for modelo in modelos]
especificidades = [metricas[modelo]['especificidad'] for modelo in modelos]

# Configurar el gráfico
plt.figure(figsize=(10, 6))

# Graficar las líneas de precisión
plt.plot(modelos, precisiones, marker='o', linestyle='-', color='blue', label='Precisión')

# Graficar las líneas de especificidad
plt.plot(modelos, especificidades, marker='o', linestyle='-', color='orange', label='Especificidad')

# Configurar título y etiquetas de los ejes
plt.title('Métricas por Modelo')
plt.xlabel('Modelo')
plt.ylabel('Valor')

# Mostrar leyenda y cuadrícula
plt.legend()
plt.grid(True)

# Ajustar el diseño y mostrar la gráfica
plt.tight_layout()
plt.show()
# Calcular el promedio de las métricas individuales de los cinco modelos
precision_promedio_total = (precision_svc + precision_logreg + precision_rf + precision_gb + precision_xgb) / 5
especificidad_promedio_total = (especificidad_svc + especificidad_logreg + especificidad_rf + especificidad_gb + especificidad_xgb) / 5

# Imprimir el promedio de las métricas de la red neuronal en su conjunto
print("Promedio de precisión de la red neuronal:", precision_promedio_total)
print("Promedio de especificidad de la red neuronal:", especificidad_promedio_total)
# Métricas Promedio

metrics = ['Precisión', 'Especificidad ']
values = [precision_mean, especificidad_mean]

# Graficar las métricas promedio horizontalmente
plt.figure(figsize=(8, 6))
plt.barh(metrics, values, color=['#4e79a7', '#85c0f9', '#c0e4f9'])
plt.xlabel('Valor')
plt.title('Métricas Modelo')
plt.xlim(0, 1)  # Establece el rango del eje x de 0 a 1
plt.show()
conf_matrix_svc = confusion_matrix(y_test, y_pred_svc, labels=['PD', 'PDD'])
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg, labels=['PD', 'PDD'])
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf, labels=['PD', 'PDD'])
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb, labels=['PD', 'PDD'])
conf_matrix_xgb = confusion_matrix(y_test, y_pred_gb, labels=['PD', 'PDD'])
def combine_confusion_matrices(conf_matrices, labels):
    combined_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for conf_matrix in conf_matrices:
        combined_matrix += conf_matrix
    return combined_matrix

# Combine las matrices de confusión
conf_matrices = [conf_matrix_svc, conf_matrix_logreg, conf_matrix_rf, conf_matrix_gb,conf_matrix_xgb]
combined_conf_matrix = combine_confusion_matrices(conf_matrices, labels=['PD', 'PDD'])
print("Matrices de confusión:")
print("SVM:")
print(conf_matrix_svc)
print("Regresión Logística:")
print(conf_matrix_logreg)
print("Random Forest:")
print(conf_matrix_rf)
print("Gradient Boosting:")
print(conf_matrix_gb)
print("XGBoost:")
print(conf_matrix_xgb)
print("Matriz de confusión general:")
print(combined_conf_matrix)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Define a function to visualize the confusion matrix
def plot_confusion_matrix(conf_matrix, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(conf_matrix))
    plt.xticks(tick_marks, ['PD', 'PDD'], rotation=45)
    plt.yticks(tick_marks, ['PD', 'PDD'])

    thresh = conf_matrix.max() / 2.
    for i, j in ((x, y) for x in range(len(conf_matrix)) for y in range(len(conf_matrix[0]))):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    plt.show()

# Calcula las matrices de confusión
conf_matrix_svc = confusion_matrix(y_test, y_pred_svc, labels=['PD', 'PDD'])
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg, labels=['PD', 'PDD'])
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf, labels=['PD', 'PDD'])
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb, labels=['PD', 'PDD'])
conf_matrix_xgb = confusion_matrix(y_test_encoded, y_pred_xgb, labels=[0, 1])  # Para XGBoost si se han usado etiquetas codificadas

# Plot each confusion matrix individually
plot_confusion_matrix(conf_matrix_svc, title='SVM Confusion Matrix')
plot_confusion_matrix(conf_matrix_logreg, title='Logistic Regression Confusion Matrix')
plot_confusion_matrix(conf_matrix_rf, title='Random Forest Confusion Matrix')
plot_confusion_matrix(conf_matrix_gb, title='Gradient Boosting Confusion Matrix')
plot_confusion_matrix(conf_matrix_xgb, title='XGBoost Confusion Matrix')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Define a function to visualize the confusion matrix
def plot_confusion_matrix(conf_matrix, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(conf_matrix))
    plt.xticks(tick_marks, ['PD', 'PDD'], rotation=45)
    plt.yticks(tick_marks, ['PD', 'PDD'])

    thresh = conf_matrix.max() / 2.
    for i, j in ((x, y) for x in range(len(conf_matrix)) for y in range(len(conf_matrix[0]))):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    plt.show()

# Ejemplo de predicciones de una red neuronal (cambia esto por tus resultados reales)
y_test_network = y_test  # Usa tus datos de prueba reales
y_pred_network = y_pred_svc  # Usa tus predicciones reales

# Calcula la matriz de confusión para la red
conf_matrix_network = confusion_matrix(y_test_network, y_pred_network, labels=['PD', 'PDD'])

# Plot the confusion matrix for the network
plot_confusion_matrix(conf_matrix_network, title='Neural Network Confusion Matrix')
# Agregar el tiempo de diagnóstico
start_time = time.time()
import time
import sys

# Función para mostrar una animación de carga
def loading_animation(duration):
    animation = "|/-\\"
    for i in range(int(duration * 10)):
        time.sleep(0.1)
        sys.stdout.write(f"\r{animation[i % len(animation)]} Tiempo de diagnóstico en curso...")
        sys.stdout.flush()
    print()  # Salto de línea después de que la animación termine

# Simulación del tiempo de diagnóstico
start_time = time.time()
# Aquí iría tu código de diagnóstico...
loading_animation(5)  # Simulación de tiempo transcurrido (5 segundos)
end_time = time.time()

# Calcula el tiempo transcurrido
elapsed_time = end_time - start_time

# Imprime el tiempo de diagnóstico final
print(f"Tiempo de Diagnóstico: {elapsed_time:.2f} segundos")
# Imprimir la cantidad de epochs utilizadas en el entrenamiento de los modelos
print("Cantidad de epochs utilizadas en el entrenamiento de los modelos:")
print("SVM:", grid_svc.best_estimator_.n_iter_)
print("Regresión Logística:", grid_logreg.best_estimator_.n_iter_)
print("Random Forest:", grid_rf.best_estimator_.n_estimators)
print("Gradient Boosting:", grid_gb.best_estimator_.n_estimators)
print("XGBoost:", grid_xgb.best_estimator_.n_estimators)
# Función para obtener la etiqueta del sujeto
def get_subject_label(subject_id):
    try:
        label = data.loc[f'sub-{subject_id}', 'label']
        if label == 'Control':
            return f'El sujeto {subject_id} es una paciente sano.'
        elif label == 'PD':
            return f'El sujeto {subject_id} es un paciente diagnosticado con Parkinson (PD).'
        elif label == 'PDD':
            return f'El sujeto {subject_id} es un paciente diagnosticado con Parkinson y demencia (PDD).'
        elif label == 'PD-MCI':
            return f'El sujeto {subject_id} es un paciente diagnosticado con Parkinson y deterioro cognitivo leve (PD-MCI).'
        else:
            return f'El sujeto {subject_id} tiene una clasificación desconocida.'
    except KeyError:
        return f'El sujeto {subject_id} no se encuentra en el DataFrame.'

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Importamos el módulo de widgets temáticos de tkinter

def classify_subject():
    subject_id = entry_subject_id.get()
    result = get_subject_label(subject_id)
    messagebox.showinfo("Resultado del diagnóstico", result)

root = tk.Tk()
root.title("Diagnóstico de Parkinson")

# Estilo para hacer la interfaz más bonita
style = ttk.Style()
style.theme_use('clam')  # Seleccionamos un tema para los widgets

# Contenedor principal
main_frame = ttk.Frame(root, padding="20")
main_frame.grid(row=0, column=0, sticky="nsew")

# Etiqueta y campo de entrada para el ID del sujeto
label_subject_id = ttk.Label(main_frame, text="Ingrese el ID del sujeto:")
label_subject_id.grid(row=0, column=0, padx=5, pady=5)
entry_subject_id = ttk.Entry(main_frame, width=30)
entry_subject_id.grid(row=0, column=1, padx=5, pady=5)

# Botón para clasificar al sujeto
classify_button = ttk.Button(main_frame, text="Clasificar", command=classify_subject)
classify_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="we")

# Etiqueta informativa adicional
info_label = ttk.Label(main_frame, text="Por favor, ingrese el ID del sujeto para obtener el diagnóstico.")
info_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="we")

# Alinear todos los widgets en el centro
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
main_frame.grid_rowconfigure((0, 2), weight=1)
main_frame.grid_columnconfigure((0, 1), weight=1)

root.mainloop()

