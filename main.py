import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Cargar el conjunto de datos heart.csv
dataFilePath = 'heart.csv'
dataFrame = pd.read_csv(dataFilePath)

# Reemplazar los valores de la variable objetivo con 1 y 0
dataFrame['target'] = dataFrame['target'].replace({'Female': 1, 'Male': 0})

# Gráfico de barras mostrando la cantidad de personas saludables y no saludables
ax = dataFrame['target'].value_counts().plot(kind='bar')
plt.title('Distribución de personas sanas y no sanas')
plt.xlabel('Objetivo')
plt.ylabel('Cantidad')
plt.show()

# Características y objetivo
y = dataFrame['target'].values
x = dataFrame.drop(["target", "age"], axis=1)

# División en conjuntos de entrenamiento y prueba
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

# Escalado de las características
scalerMM = MinMaxScaler()
scalerS = StandardScaler()

xtrainMM = scalerMM.fit_transform(xtrain)
xtestMM = scalerMM.transform(xtest)

xtrainS = scalerS.fit_transform(xtrain)
xtestS = scalerS.transform(xtest)

# Entrenamiento del modelo con datos escalados por MinMax
modelo = MLPClassifier(alpha=1, max_iter=1000, random_state=42)
modelo.fit(xtrainMM, ytrain)

# Evaluación con datos escalados por MinMax
print('Escala MinMax:')
print('Precisión en entrenamiento:', modelo.score(xtrainMM, ytrain))
print('Precisión en prueba:', modelo.score(xtestMM, ytest))
ytest_pred = modelo.predict(xtestMM)
print('Reporte de clasificación:\n', classification_report(ytest, ytest_pred))
cm = confusion_matrix(ytest, ytest_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión - Escala MinMax')
plt.show()

# Entrenamiento del modelo con datos escalados por StandardScaler
modelo = MLPClassifier(alpha=1, max_iter=1000, random_state=42)
modelo.fit(xtrainS, ytrain)

# Evaluación con datos escalados por StandardScaler
print('Escala StandardScaler:')
print('Precisión en entrenamiento:', modelo.score(xtrainS, ytrain))
print('Precisión en prueba:', modelo.score(xtestS, ytest))
ytest_pred = modelo.predict(xtestS)
print('Reporte de clasificación:\n', classification_report(ytest, ytest_pred))
cm = confusion_matrix(ytest, ytest_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión - Escala StandardScaler')
plt.show()
