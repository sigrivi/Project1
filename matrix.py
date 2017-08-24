import numpy
print ("hello world")

def selectsize(size):
	a=numpy.ones(size-1)
	b=numpy.ones(size)
	c=numpy.ones(size-1)
	return(a*(-1),b*2,c*(-1))
selectsize(5)
def buildmatrix(vectora, vectorb, vectorc):
	A=numpy.zeros((len(vectorb),len(vectorb)))
	for i in range(len(vectora)):
		A[i+1,i]=vectora[i]
	for i in range(len(vectorb)):
		A[i,i]=vectorb[i]
	for i in range(len(vectorc)):
		A[i,i+1]=vectorc[i]
	print(A)
a,b,c=selectsize(10)
buildmatrix(a,b,c)

