import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import sys

N = 100 #number of steps

#makes the vector f(x)
func = lambda x: 100*np.exp(-10*x)
x = np.linspace(0,1,N+2) #values on the 1. axis
f = np.zeros(N)
f[:] = func(x[1:N+1])





def solveequation(function):
	N = len(function)	
	f = function.copy()
	coeff = np.zeros(N)
	solution = np.zeros(N)
	h = 1/N #step size

	for i in range(N):
		coeff[i] = (i+2)/(i+1)

	#print(coeff) #coeffisients are ok
	for i in range(N-1):
		f[i+1] = function[i+1]+i*f[i]/(i+1)
	f=f*(h**2)

	solution[N-1] = f[N-1]/coeff[N-1]
	
	for i in range(2,N-1):
		k = N-i
		solution[k] = (f[k]+solution[k+1])/coeff[k]
				

	return(solution)
solution = solveequation(f)

#solution vector u
func = lambda x: 1-(1-np.exp(-10))*x-np.exp(-10*x)
g = np.linspace(0,1,N+2)
u = np.zeros(N)
u[:] = func(g[1:N+1])

plt.plot(g[1:N+1], u, 'r',g[1:N+1], solution, 'b')
plt.legend(["Exact","Numerical"])
plt.show()

#print(solution)
#plt.plot(solution)
#plt.show()


