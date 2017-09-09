import numpy
import matplotlib.pyplot as plt
from scipy import linalg
import sys

#Uses the vectors a,b,c to build the tridiagonal matrix.
def buildmatrix(N):

	a=numpy.ones(N-1)*(-1)
	b=numpy.ones(N)*2
	c=numpy.ones(N-1)*(-1)
	A=numpy.zeros((N,N))

	for i in range(N-1):
		A[i+1,i]=a[i]
	for i in range(N):
		A[i,i]=b[i]
	for i in range(N-1):
		A[i,i+1]=c[i]
	return(A)

N = 100  #here you can choose the number of grid points
D=buildmatrix(N)
E=D.copy()


g = numpy.linspace(0,1,N+2) #values on the 1. axis

#makes the vector f(x)=100*exp(-10x)
func = lambda x: 100*numpy.exp(-10*x)
f = numpy.zeros(N)
f[:] = func(g[1:N+1])

# exact solution vector u
func = lambda x: 1-(1-numpy.exp(-10))*x-numpy.exp(-10*x)
u = numpy.zeros(N)
u[:] = func(g[1:N+1])


# intention of this function: solve the equation Ev=q
def solveequation(E,q):
	v=q.copy()
	h=1/q.size
	v=v*h**2
	
	#reduses E to an upper triangular matrix:
	for jjj in range(E.shape[0]-1): 
		ff=E[jjj+1,jjj]/E[jjj,jjj]
		E[jjj+1,:]=E[jjj+1,:]-E[jjj,:]*ff
		v[jjj+1]=v[jjj+1]-v[jjj]*ff
	
	#reduces E to a diagonal matrix:
	for jjj in reversed(range(1,E.shape[0])):
		ff=E[jjj-1,jjj]/E[jjj,jjj]
		E[jjj-1,:]=E[jjj-1,:]-E[jjj,:]*ff
		v[jjj-1]=v[jjj-1]-v[jjj]*ff
	
	#the solution vector v:
	for jjj in range(E.shape[0]):
		v[jjj]=v[jjj]/E[jjj,jjj]
	
	return v

#to test the equation solver:
v=solveequation(E,f)

plt.plot(g[1:N+1], u, 'r',g[1:N+1], v, 'b')
plt.legend(["Exact","Numerical"])
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('the exact and numerical solution of u²(x)/dx²=100*exp(-10)')
plt.show()


