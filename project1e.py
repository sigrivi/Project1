import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time

from project1b import buildmatrix
from project1c import solveequation

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


D = buildmatrix(N) #builds tridiagonal matrix
E = D.copy()

time1=time.time()
v1 = linalg.solve(E,f*h**2)
time2=time.time()
print("time:",(time2-time1)*1000)

print(v1)

v2 = solveequation(f)
diff = v1-v2

plt.plot(g[1:N+1], diff)
#plt.plot(g[1:N+1], v1, 'r',g[1:N+1], v2, 'b')
#plt.legend(["Solution from LU decomposition","My numerical solution"])
plt.xlabel('x')
plt.ylabel('v1-v2')
plt.title('LU solution (v1) - row reduction solution(v2)')
#plt.savefig('difference.png',dpi=225)
plt.show()

