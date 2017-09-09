import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import sys


N = 10 #here you can choose the number of grid points
h = 1/N
g = np.linspace(0,1,N+2) #values on the 1. axis

#makes the vector f(x)
func = lambda x: 100*np.exp(-10*x)
f = np.zeros(N)
f[:] = func(g[1:N+1])

#solution vector u
func = lambda x: 1-(1-np.exp(-10))*x-np.exp(-10*x)
u = np.zeros(N)
u[:] = func(g[1:N+1])

print(u)

#Uses the vectors a,b,c to build the tridiagonal matrix.
def buildmatrix(N):

	a=np.ones(N-1)*(-1)
	b=np.ones(N)*2
	c=np.ones(N-1)*(-1)
	A=np.zeros((N,N))

	for i in range(N-1):
		A[i+1,i]=a[i]
	for i in range(N):
		A[i,i]=b[i]
	for i in range(N-1):
		A[i,i+1]=c[i]
	return(A)

D = buildmatrix(N)
E = D.copy()

v = linalg.solve(E,f*h**2)

plt.plot(g[1:N+1], u, 'r',g[1:N+1], v, 'b')
plt.legend(["Exact","Numerical"])
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('the exact and numerical solution of u²(x)/dx²=100*exp(-10)')
plt.show()

