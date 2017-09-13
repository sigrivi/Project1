import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time

N = 10 #here you can choose the number of grid points
g = np.linspace(0,1,N+2) #values on the 1. axis

#makes the vector f(x)
func = lambda x: 100*np.exp(-10*x)
f = np.zeros(N)
f[:] = func(g[1:N+1])

#solution vector u
func = lambda x: 1-(1-np.exp(-10))*x-np.exp(-10*x)
u = np.zeros(N)
u[:] = func(g[1:N+1])


def solveequation(function):
	N = len(function)	
	f = function.copy()
	coeff = np.zeros(N)
	solution = np.zeros(N)
	h = 1/N #step size

	for i in range(N):
		coeff[i] = (i+2)/(i+1)

	for i in range(N-1):
		f[i+1] = function[i+1]+f[i]/coeff[i]
	f=f*(h**2)

	solution[N-1] = f[N-1]/coeff[N-1]

	for i in range(2,N+1):#originally, i used range(2,N-1), this caused the two first elements of solution to be zero.
		k = N-i
		solution[k] = (f[k]+solution[k+1])/coeff[k]
		
	return(solution)

time1=time.time()
v = solveequation(f) #numerical solution
time2=time.time()
print("time:",(time2-time1)*1000)

print(v)


plt.plot(g[1:N+1], u, 'r',g[1:N+1], v, 'b')
plt.legend(["Exact","Numerical"])
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('the exact and numerical solution of u²(x)/dx²=100*exp(-10)')
plt.show()

