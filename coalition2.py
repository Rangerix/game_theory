import pandas,numpy,math
import sys
from sklearn import metrics
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#======================================================================
#DEFINE MUTUAL_INFO HERE	
#======================================================================
def choose(n,k):
	"""
	A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
	"""
	if 0 <= k <= n:
		ntok = 1
		ktok = 1
		for t in range(1, min(k, n - k) + 1):
			ntok *= n
			ktok *= t
			n -= 1
		return ntok // ktok
	else:
		return 0
#=======================================================================
infile=sys.argv[1]
dataframe=pandas.read_csv(infile)
(a,b)= numpy.shape(dataframe)
print (a)
print (b)
feature = dataframe.values[:,0:b-1]
target = dataframe.values[:,b-1]

data=feature
n=b-1
omega=3

storedval=numpy.full((n,n),-2.0) #using the fact that normalized_mutual_info_score lies b/w 0 and 1
for i in range(n):
	print(i)
	var1=data[:,i]
	for j in range(n):
		#this mutual_info_score : we can use any other criteria here
		if storedval[j][i]== -2.0:
			var2=data[:,j]
			info=metrics.normalized_mutual_info_score(var1,var2)
		else :
			info=storedval[j][i]
		storedval[i][j]=info
		#print(i,j,info)
print("\n\npre processed score criteria")

m=n-1
#-------------------------
allout=[]
alloutset=set()
size=m
num=numpy.repeat(0,size)
#alloutset=set(tuple(i) for i in allout)

print("\n\npre processed iteration")

#------------------------
delta=numpy.repeat(0.0,n)
w_one=numpy.repeat(0,n)

coalitioncutoff=0.5
for i in range(n):
	newdata=numpy.delete(data,i,axis=1)
	#(_,m)=numpy.shape(newdata)
	#print(i)
	#print(i,"new data dimension ",m)
	for j in range(m):
		#print(flag,bin(flag))
		#print(j)
		#var1=data[:,i]
		#var2=newdata[:,j]
		temp=j
		if j>=i :
			temp+=1
		info=storedval[i][temp]
		#print(independent,dependent,p)
		#print(p)
		if info<0.4:
			w_one[i]+=1
	w_two_i=choose(w_one[i],2)+ (m- w_one[i])*w_one[i]
	w_three_i=choose(w_one[i],3)+choose(w_one[i],2)*(m - w_one[i])
	delta[i]=w_one[i]+w_three_i+w_two_i
	#print(i,w_one[i],delta[i])
	val=delta[i]/(1<<omega)
	delta[i]=float(val)
	#print(delta[i])
print("delta done")
print(w_one)
#print(delta)
#------------------calculate feature importance--------------------
featureimportance=numpy.repeat(0.0,n)
var2=target
for i in range(0,n-1):
	var1=data[:,i]
	filtervalue=metrics.normalized_mutual_info_score(var1,var2) #here we can use any other filter
	selectionvalue=filtervalue*delta[i]
	featureimportance[i]=selectionvalue
#print(featureimportance)
sortedfeatures=(-featureimportance).argsort()
print(sortedfeatures)
print("ranking done ")


#test accuracy==========================================================================
ranking=sortedfeatures
data=feature
for loopval in range(5,b-1,5):
	temp=ranking[0:loopval]
	#print(temp)
	datanew=data[:,temp]
	
	cross=10
	maxacc=0
	for _ in range(0,cross):
		test_size=(1/cross)
		X_train, X_test, y_train, y_test = train_test_split(datanew, target,stratify=target ,test_size=test_size)
		
		clf=RandomForestClassifier()
		clf.fit(X_train,y_train)
		val=clf.score(X_test,y_test)
		if val>maxacc:
			maxacc=val
			#if we want to save the test and train data
			#savetrainX=X_train
			#savetrainY=y_train
			#savetestX=X_test
			#savetestY=y_test
	print(loopval,maxacc)
