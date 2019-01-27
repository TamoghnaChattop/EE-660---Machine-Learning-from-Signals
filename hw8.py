# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 08:38:45 2018

@author: tchat
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
x1 = np.random.uniform(-1,1,1000)
x2 = np.random.uniform(-1,1,1000)
y1 = x1**2
y2 = x2**2

a = x1+x2
b = -x1*x2

# Plot g_bar vs f
for i in range(0,1000):
    x = [x1[i],x2[i]]
    y = [y1[i],y2[i]]
    plt.plot(x,y,'g')
    
x_value = np.arange(-1,1,0.01)
y_value = x_value**2
plt.plot(x_value,y_value,'r')
plt.title('Plot of g_(x) and f(x) together')
plt.xlabel('Input values between -1 and 1')
plt.ylabel('Output values of the function')

# Calculate g_bar
x = np.random.uniform(-1,1,1000)

a_gbar = np.mean(x1+x2)
b_gbar = -np.mean(x1)*np.mean(x2)
g_bar = a_gbar *x + b_gbar

plt.plot(x,g_bar, 'b')

# Calculate bias
f_x = x**2
bias = np.mean((g_bar - f_x)**2)
print('Value of Bias is : ', bias)

# Calculate variance
g_x = a*x + b
var = np.mean((g_x - g_bar)**2)
print('Value of Variance is : ', var)

# Calculate Eout
eout = np.mean((g_x - f_x)**2)
print('Value of Eout is : ', eout)
print('Value of E[Eout] is : ', np.mean(eout))
    

