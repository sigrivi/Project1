import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import sys
#makes the vectors of the tridiagonal matrix. You choose the size of the vectors.
def selectsize(size):
	a=np.ones(size-1)
	b=np.ones(size)
	c=np.ones(size-1)
	return(a*(-1),b*2,c*(-1))

#Uses the vectors to build the tridiagonal matrix.
def buildmatrix(vectora, vectorb, vectorc):
	A=np.zeros((len(vectorb),len(vectorb)))
	for i in range(len(vectora)):
		A[i+1,i]=vectora[i]
	for i in range(len(vectorb)):
		A[i,i]=vectorb[i]
	for i in range(len(vectorc)):
		A[i,i+1]=vectorc[i]
	return(A)

N = 40
a,b,c=selectsize(N)
D=buildmatrix(a,b,c)
E=D.copy()

#makes the vector f(x)
func = lambda x: 100*np.exp(-10*x)
a = np.linspace(0,1,N+2) #values on the 1. axis
f = np.zeros(N)
f[:] = func(a[1:N+1])

for jjj in range(E.shape[0]-1):

	ff=E[jjj+1,jjj]/E[jjj,jjj]
	E[jjj+1,:]=E[jjj+1,:]-E[jjj,:]*ff
print("tridiagonal matrix after first step of reduction:")
print(E)

for jjj in reversed(range(1,E.shape[0])):
	ff=E[jjj-1,jjj]/E[jjj,jjj]
	E[jjj-1,:]=E[jjj-1,:]-E[jjj,:]*ff
#print("tridiagonal matrix after reduction:")
#print(E)

# intention of this function: solve the equation Ev=q
def solveequation(E,q):
	v=q.copy()
	h=1/q.size
	v=v*h**2
	
	for jjj in range(E.shape[0]-1):
		ff=E[jjj+1,jjj]/E[jjj,jjj]
		E[jjj+1,:]=E[jjj+1,:]-E[jjj,:]*ff
		v[jjj+1]=v[jjj+1]-v[jjj]*ff
	print(E)
	for jjj in reversed(range(1,E.shape[0])):
		ff=E[jjj-1,jjj]/E[jjj,jjj]
		E[jjj-1,:]=E[jjj-1,:]-E[jjj,:]*ff
		v[jjj-1]=v[jjj-1]-v[jjj]*ff
	
	for jjj in range(E.shape[0]):
		v[jjj]=v[jjj]/E[jjj,jjj]

	return v

solution = solveequation(E,f)

#solution vector u
func = lambda x: 1-(1-np.exp(-10))*x-np.exp(-10*x)
g = np.linspace(0,1,N+2)
u = np.zeros(N)
u[:] = func(g[1:N+1])

plt.plot(g[1:N+1], u, 'r',g[1:N+1], solution, 'b')
plt.legend(["Exact","Numerical"])
plt.show()
