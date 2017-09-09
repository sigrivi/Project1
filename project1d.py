import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import sys

N = 10 #here you can choose the number of grid points
g = np.linspace(0,1,N+2)

def makevectorf(N): #N is size
	func = lambda x: 100*np.exp(-10*x)
	g = np.linspace(0,1,N+2) #values on the 1. axis
	f = np.zeros(N)
	f[:] = func(g[1:N+1])
	return(f)

def makeexactsolution(N):
	func = lambda x: 1-(1-np.exp(-10))*x-np.exp(-10*x)
	g = np.linspace(0,1,N+2)
	u = np.zeros(N)
	u[:] = func(g[1:N+1])
	return(u)

def solveequation(function):
	N = len(function)	
	f = function.copy()
	f = f.astype('float32')
	coeff = np.zeros(N,dtype='float32')
	solution = np.zeros(N,dtype='float32')
	h = np.float32(1/N) #step size

	for i in range(N):
		coeff[i] = (i+2)/(i+1)

	for i in range(N-1):
		
		f[i+1] = function[i+1]+f[i]/coeff[i]
	f=f*(h**2)

	solution[N-1] = f[N-1]/coeff[N-1]
	
	for i in range(2,N+1):
		k = N-i
		solution[k] = (f[k]+solution[k+1])/coeff[k]
		
	return(solution)

#vecf = makevectorf(N)
#v = solveequation(vecf)
#u = makeexactsolution(N)



def calculateerror(v,u): #v is the numerical solution, u is the exact solution
	N = len(u)
	#error = np.zeros(N)
	error = np.log10(np.abs((u[1:N-1]-v[1:N-1])/u[1:N-1]))
	#for i in range(N-1):
		#print("v[i]-u[i]",(v[i]-u[i]))
		#print("u[i]",u[i])
		#print("v[i]-u[i])/u[i]", (v[i]-u[i])/u[i])
		#error[i] = np.log10( abs( (v[i]-u[i])/u[i] ) )
	#plt.plot(error)	
	#plt.show()

	maxerror = np.mean(error[1:N-1]) #end elements have error equal zero
	return(maxerror)
#error = calculateerror(v, u)
#print(error)


def ploterror(maxN): #maxN is the power of 10
	e = np.zeros(maxN)
	g = np.zeros(maxN)

	for i in range(maxN):
		N=10**(i+1)
		f = makevectorf(N)
		v = solveequation(f)
		u = makeexactsolution(N)
		print(len(v), len(u))
		e[i] = calculateerror(v,u)
		print(e[i])
		g[i] = -(i+1) # g = log10(1/N) 
	plt.plot(g, e, 'r', g, 2*g, 'b')
	plt.xlabel("log10(h)")
	plt.ylabel("log10(error)")
	plt.title("log-log plot of relative error as a function of step size")
	plt.savefig('errorfig.png',dpi=225)
	plt.show()

	return(e)
	
e = ploterror(6)

