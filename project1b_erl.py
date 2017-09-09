
import numpy as np
import numpy
import matplotlib.pyplot as plt




def selectsize(size):
	a=numpy.ones(size-1)
	b=numpy.ones(size)
	c=numpy.ones(size-1)
	return(a*(-1),b*2,c*(-1))

#Uses the vectors to build the tridiagonal matrix.
def buildmatrix(vectora, vectorb, vectorc):
	A=numpy.zeros((len(vectorb),len(vectorb)))
	for i in range(len(vectora)):
		A[i+1,i]=vectora[i]
	for i in range(len(vectorb)):
		A[i,i]=vectorb[i]
	for i in range(len(vectorc)):
		A[i,i+1]=vectorc[i]
	return(A)


def solveequation(E,q):
	v=q.copy()
	h=1/q.size
	v=v*h**2
	
	for jjj in range(E.shape[0]-1):
		ff=E[jjj+1,jjj]/E[jjj,jjj]
		E[jjj+1,:]=E[jjj+1,:]-E[jjj,:]*ff
		v[jjj+1]=v[jjj+1]-v[jjj]*ff
	#print(E)
	for jjj in reversed(range(1,E.shape[0])):
		ff=E[jjj-1,jjj]/E[jjj,jjj]
		E[jjj-1,:]=E[jjj-1,:]-E[jjj,:]*ff
		v[jjj-1]=v[jjj-1]-v[jjj]*ff
	
	for jjj in range(E.shape[0]):
		v[jjj]=v[jjj]/E[jjj,jjj]

	return v


N = 10 #number of steps
a,b,c=selectsize(N)
D=buildmatrix(a,b,c)

func = lambda x: 100*np.exp(-10*x)
exact_func = lambda x: 1-(1-numpy.exp(-10))*x-numpy.exp(-10*x)

x = np.linspace(0,1,N+2) #values on the 1. axis
f = np.zeros(N)
f[:] = func(x[1:N+1])
g=exact_func(x[1:N+1])

v=solveequation(D,f)
print(v)


plt.plot(x[1:N+1], g, 'r',x[1:N+1], v, 'b')
plt.legend(["Exact","Numerical"])
plt.show()




