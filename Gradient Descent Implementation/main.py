import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# We read our dataset and drop na values
columns = ["Date","Precip","MaxTemp","MinTemp","MeanTemp"]
df = pd.read_csv('Summary of Weather.csv',names = columns)
df.dropna()

#Function to compute the MSE
# Params: 
# n = size of the sample
# y = list with every real value
# y_hat = list with every predicted value 
# returns: MSE 
def mean_square_error(n,y,y_hat):
  square_diff_sum = 0
  #Here we iterate through the substraction of each element of both lists (y and y_hat)
  #For each difference, we power it to 2 and add it to our summatory
  for diff in (y - y_hat):
    square_diff_sum += diff ** 2
  #after we obtained the summatory, we apply the rest of the formula
  return ((1/n) * square_diff_sum )


# This function applies the gradient descent algorithm for linear regression
# It supports two variables.
# Params:
# x1: Series with the first independent variable
# x2: Series with the second independent variable
# alpha: Learning rate (hyperparameter)
# epochs: the amount of epochs or iterations to execute 
def multivariable_linear_reg_gd(x1, x2, y,alpha, epochs):
  # We define our initial values
  m1_current = 0
  m2_current = 0
  b_current = 0
  n = len(x1)
  # We create two arrays that will be used to plot the error change related to epochs
  errors = []
  epochs_array = np.array(range(0, epochs))
  # Here, the GD algorithm starts. The epochs define how many times it will be executed.
  for i in range (epochs):
    # First we compute the value of the current predictions according to the current values
    y_hat = (m1_current  *x1) + (m2_current * x2) + b_current 
    # Then we compute the MSE that is produced with our current parameters
    mse = mean_square_error(n,y,y_hat)
    print ("Epoch #", i+1, ", current function: y = ", m1_current, "* x1 + ", m2_current, "*x2 +", b_current, ", Error:", mse)
    # Now we apply the derivative part of the algorithm, useful for computing our next values
    bder = -(2/n) * sum ( y - y_hat)
    m1der = -(2/n) * sum (x1*( y - y_hat))
    m2der = -(2/n) * sum (x2*( y - y_hat))
    # Now we use the obtained derivative values to compute the new current values, which will be used in the next iteration. 
    # Here is where we use the value of the learning rate
    m1_current = m1_current - m1der * alpha
    m2_current = m2_current - m2der * alpha
    b_current = b_current - bder * alpha
    # We add the errors obtained in this epoch to the errors list that will be plotted at the end
    errors.append(mse)
  # After finishing the training, we will plot the errors that we stored previously
  plt.plot(epochs_array, errors, color = "blue", marker = "x", label = "MSE")
  plt.legend()
  plt.title("Error magnitude compared to the epoch number")
  plt.xlabel("Epochs")
  plt.ylabel("Mean Square Error Magnitude")
  plt.show()


# Parameters for two variable linear regression
X1 = df["MinTemp"]  
X2 = df["MeanTemp"]
Y = df["MaxTemp"]
# I found a = 0.0001 to be good with the 2 x's data =)
alpha = 0.0001
# The amount of epochs to be executed. 
epochs = 2000
# And now we execute our linear regression 
multivariable_linear_reg_gd(X1,X2, Y, alpha, epochs)


