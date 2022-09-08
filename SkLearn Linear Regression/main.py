import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Importamos la información a utilizar
columns = ["Date", "Precip", "MaxTemp", "MinTemp", "MeanTemp"]
df = pd.read_csv('Summary of Weather.csv', names=columns)
#df.dropna()

# Definimos los valores de las variables independientes y la variable dependiente
X1 = df["MinTemp"]
X2 = df["MeanTemp"]
X = df[["MinTemp","MeanTemp"]]
y = df["MaxTemp"]

# Se crea un conjunto para train y uno para test para cada set de datos. Se toma el 30% de los datos para train
# Random state nos permite pasar una semilla para la aleatoreidad
#en la separación
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=1)

# Creamos el modelo como regresión lineal. Fit intercept nos permite indicar si se debe calcular el intercepto.
model = LinearRegression(fit_intercept = True)

# Entrenamos el modelo
model.fit(X_train, y_train)

# Obtenemos los coeficientes del atributo coef_ y el intercepto del atributo intercept
print ("MaxTemp = ", model.coef_[0], "*(MinTemp) +", model.coef_[1], "*(MeanTemp) + ", model.intercept_) 

# Usamos el modelo para predecir sobre nuestro set de pruebas 
predecir = model.predict(X_test)

# model evaluation
print('MSE de las predicciones : ', mean_squared_error(y_test, predecir))
