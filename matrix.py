import numpy
import matplotlib.pyplot as plt
from scipy import linalg
import sys
#makes the vektors of the tridiagonal matrix. You choose the size of the vectors.
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

N = 100
a,b,c=selectsize(N)
D=buildmatrix(a,b,c)

#print(D)


E=D.copy()
F=D.copy()
G=D.copy()
#first attempt on row reduction. Does not work, it does not modyfy all the elements of each row:
#for j in range(D.shape[0]-1):
#	for i in range(D.shape[0]):
#		ff=D[j,j]
#		D[j+1,i]=D[j+1,i]-D[j,i]*D[j+1,j]/ff
#		print(D)

#makes the vector f(x)
func = lambda x: 100*numpy.exp(-10*x)
a = numpy.linspace(0,1,N+2) #values on the 1. axis
f = numpy.zeros(N)

print (a)
print (a[1:N])
print (len(a))
print (len(a[1:N]))

f[:] = func(a[1:N+1])


#print(f)

#print("tridiagonal matrix before reduction:")
#print(E)

#second attempt on row reduction. works well:

for jjj in range(E.shape[0]-1):

	ff=E[jjj+1,jjj]/E[jjj,jjj]
	E[jjj+1,:]=E[jjj+1,:]-E[jjj,:]*ff
#print("tridiagonal matrix after first step of reduction:")
#print(E)

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

def solveequationAlternative(E,q):
	v=q.copy()
	h=1.0/q.size
	v=v*h**2
	N=E.shape[0]
	#print(v)
	for jjj in range(N-1):
		ff=-1/E[jjj,jjj]
		E[jjj+1,:]=E[jjj+1,:]-E[jjj,:]*ff
		v[jjj+1]=v[jjj+1]-v[jjj]*ff
		#print(v)
	
	print (E)
	solution = numpy.zeros(N)
	solution[N-1] = v[N-1]/E[N-1,N-1]
	
	#for k in range(2,N+1):
		#print (j)
		#j = N-k	
	for j in reversed(range(2,N+1)):	
		solution[j] = (v[j]-E[j,j+1]*solution[j+1])/-1
	return solution

#to test the equation solver:
v=solveequation(F,f)


#solution vector u
func = lambda x: 1-(1-numpy.exp(-10))*x-numpy.exp(-10*x)
g = numpy.linspace(0,1,N+2)
u = numpy.zeros(N)
u[:] = func(g[1:N+1])

plt.plot(g[1:N+1], u, 'r',g[1:N+1], v, 'b')
plt.legend(["Exact","Numerical"])
plt.show()

def uppertriangular(q): #n is the dimension
	N=len(q)
	A=numpy.zeros((N,N))
	A[0,0]=2
	for i in range(N-1):
		A[i+1,i+1]=2-1/A[i,i]
		q[i+1]=q[i+1]-q[i]/A[i,i]
		A[i,i+1]=-1
	solution=numpy.zeros(N)
	solution[N]=q[N]
	for i in range(1,N):
		k=N-i
		solution[k-1]=A[k-1]*q[k-1]-q[k]
	print(q)	
	return(A)

#def uppertriangular(N): #N is the dimension
#	A=numpy.zeros((N,N))
#	A[0,0]=2
#	for i in range(N):
#		A[i,i]=(i+2)/(i+1)
#	
#	for i in range(N-1):
##		A[i,i+1]=-1
#	return(A)

Q=uppertriangular(f)
print(Q)


