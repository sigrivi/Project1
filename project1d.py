import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time

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
	#u = u.astype('float32')
	return(u)

def solveequation(function):
	N = len(function)	
	f = function.copy()
	#f = f.astype('float32')
	#coeff = np.zeros(N,dtype='float32')
	coeff = np.zeros(N)
	#solution = np.zeros(N,dtype='float32')
	solution = np.zeros(N)
	#h = np.float32(1/N) #step size
	h=1/N
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
#time1=time.time()
#v = solveequation(vecf)
#time2=time.time()
#print("time:",(time2-time1)*1000)
#u = makeexactsolution(N)



def calculateerror(v,u): #v is the numerical solution, u is the exact solution
	N = len(u)
	
	error = np.log10(np.abs((u[1:N-1]-v[1:N-1])/u[1:N-1]))
	
	g = np.linspace(0,1,N+2)	
	#plt.plot(g[2:N-100],error[0:N-102])	
	#plt.xlabel("x")
	#plt.ylabel("log10(error)")
	#plt.title("Relative error, double precision, N=1000")
	#plt.savefig('relativeerror_double.png',dpi=225)
	#plt.show()

	maxerror = np.amax(error[1:N-1]) #end elements have error equal zero
	meanerror = np.mean(error[1:N-1])
	return(maxerror, meanerror)
#error = calculateerror(v, u)
#print(error[1])


def ploterror(maxN): #maxN is the power of 10
	e = np.zeros(maxN)
	em = np.zeros(maxN)
	g = np.zeros(maxN)

	for i in range(maxN):
		N=10**(i+1)
		f = makevectorf(N)
		v = solveequation(f)
		u = makeexactsolution(N)
		print(len(v), len(u))
		e[i] = calculateerror(v,u)[0]
		em[i] = calculateerror(v,u)[1]
		print(e[i])
		g[i] = -(i+1) # g = log10(1/N) 
	#plt.plot(g, e, 'r', g, 2*g, 'b')
	plt.plot(g,e,'b',g,em,'r')
	plt.xlabel("log10(h)")
	plt.ylabel("log10(error)")
	plt.title("log-log plot of relative error as a function of step size")
	plt.savefig('errorfig.png',dpi=225)
	plt.show()

	return(e)
	
e = ploterror(6)
#print(e)

