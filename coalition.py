import pandas,numpy,math
import sys
from sklearn import metrics
from skfeature.function.similarity_based import fisher_score


#======================================================================
#DEFINE MUTUAL_INFO HERE

#======================================================================

infile=sys.argv[1]
dataframe=pandas.read_csv(infile)
(a,b)= numpy.shape(dataframe)
print (a)
print (b)
feature = dataframe.values[:,0:b-1]
target = dataframe.values[:,b-1]

data=feature
n=b
omega=3
storedval=numpy.zeros(shape=(n,n))
for i in range(0,n-1):
	for j in range(0,n-1):
		var1=data[:,i]
		var2=data[:,j]
		#this mutual_info_score : I have ti chane this
		#info=metrics.mutual_info_score(var1,var2,contingency=None)
		info=metrics.normalized_mutual_info_score(var1,var2)
		#info=criteria(var1,var2)
		storedval[i][j]=info
		#print(i,j,info)

delta=numpy.repeat(0,n)
coalitioncutoff=0.5
for i in range(0,n-1):
	newdata=numpy.delete(data,i,axis=1)
	(_,m)=numpy.shape(newdata)
	print(i,"new data dimension ",m)
	for flag in range(1,(1<<m)-1 ):
		if bin(flag).count("1")<=omega :
			#print(flag,bin(flag))
			independent=0
			dependent=0
			for j in range(0,m):
				if flag & (1<<j) :
					#print(j)
					var1=data[:,i]
					var2=newdata[:,j]
					temp=j
					if j>=i :
						temp+=1
					info=storedval[i][temp]
					#info=metrics.mutual_info_score(var1,var2,contingency=None)
					if info < 0.4:
						independent= independent +1
					else :
						dependent=dependent+1
						
			p=1.0
			if dependent != 0:
				p=independent/dependent
			#print(independent,dependent,p)
			#print(p)
			if p>= coalitioncutoff:
				delta[i]=delta[i]+1
	#print(delta[i])
	val=delta[i]/(1<<omega)
	delta[i]=float(val)
	print(delta[i])

#==============feature importance================
