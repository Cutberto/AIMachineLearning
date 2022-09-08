import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mse(n, y, y_hat):
    square_diff_sum = 0
    #Here we iterate through the substraction of each element of both lists (y and yhat)
    for diff in (y - y_hat):
        square_diff_sum += diff**2
    #after we obtained the summatory, we apply the rest of the formula
    return ((1 / n) * square_diff_sum)


# Importamos la información a utilizar
columns = ["Date", "Precip", "MaxTemp", "MinTemp", "MeanTemp"]
df = pd.read_csv('Summary of Weather.csv', names=columns)
#df.dropna()

# Definimos los valores de las variables independientes y la variable dependiente
X1 = df["MinTemp"]
X2 = df["MeanTemp"]
X = df[["MinTemp", "MeanTemp"]]
y = df["MaxTemp"]

# Se crea un conjunto para train y uno para test para cada set de datos. Se toma el 30% de los datos para train
# Random state nos permite pasar una semilla para la aleatoreidad
#en la separación
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=1)

# Creamos el modelo como regresión lineal. Fit intercept nos permite indicar si se debe calcular el intercepto.
model = LinearRegression(fit_intercept=True)

# Entrenamos el modelo
model.fit(X_train, y_train)

# Obtenemos los coeficientes del atributo coef_ y el intercepto del atributo intercept
print("MaxTemp = ", model.coef_[0], "*(MinTemp) +", model.coef_[1],
      "*(MeanTemp) + ", model.intercept_)

# Usamos el modelo para predecir sobre nuestro set de pruebas
#predecir = model.predict(X_test)

# Ahora haremos las predicciones a mano usando el dataset
y_p = []
n = 0

print ("Iniciando predicciones...")
for temperature in range (0, len(y_test)):
  coef_1 = model.coef_[0]* (X_test["MinTemp"].tolist()[temperature])
  coef_2 = model.coef_[1]* (X_test["MeanTemp"].tolist()[temperature])
  y_hat = coef_1 + coef_2 + model.intercept_
  y_p.append(y_hat)
  print ("n=", n  , "MinTemp = " , X_test["MinTemp"].tolist()[temperature] , "MeanTemp",  (X_test["MeanTemp"].tolist()[temperature])   , "y_hat=",  y_hat)
  n+=1

                   
print("El MSE de la predicción es ", mse(n, y_test, y_p))

