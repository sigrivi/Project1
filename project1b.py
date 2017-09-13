import numpy
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time

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

N = 10  #here you can choose the number of grid points
D=buildmatrix(N)
E=D.copy()
print(D)

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
	for i in range(E.shape[0]-1): 
		ff=E[i+1,i]/E[i,i]
		E[i+1,:]=E[i+1,:]-E[i,:]*ff
		v[i+1]=v[i+1]-v[i]*ff
	
	#reduces E to a diagonal matrix:
	for i in reversed(range(1,E.shape[0])):
		ff=E[i-1,i]/E[i,i]
		E[i-1,:]=E[i-1,:]-E[i,:]*ff
		v[i-1]=v[i-1]-v[i]*ff
	
	#the solution vector v:
	for i in range(E.shape[0]):
		v[i]=v[i]/E[i,i]
	
	return v


time1=time.time()
v=solveequation(E,f) #numerical solution
print(v)
time2=time.time()
print("time:",(time2-time1)*1000)


plt.plot(g[1:N+1], u, 'r',g[1:N+1], v, 'b')
plt.legend(["Exact","Numerical"])
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('solution of u²(x)/dx²=100*exp(-10), 1000 grid points')
plt.savefig('1000gridpoints.png',dpi=225)

plt.show()


