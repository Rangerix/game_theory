import pandas,numpy,math
import sys
from sklearn import metrics
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#======================================================================
#DEFINE MUTUAL_INFO HERE	
#======================================================================
def choose(n, k):
	"""
	A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
	"""
	if 0 <= k <= n:
		ntok = 1
		ktok = 1
		for t in xrange(1, min(k, n - k) + 1):
			ntok *= n
			ktok *= t
			n -= 1
		return ntok // ktok
	else:
		return 0
#=======================================================================

def gen_iteration(num,start,end,count):
	if count>=0:
		print(num)
		num1=[]
		num1=num.copy()
		#allout.append(num1)
		setbit=[]
		for i in range(size):
			if num1[i]==1:
				setbit.append(i)
		t=tuple(setbit)
		alloutset.add(t)
		#print(allout)
		#print(num)
	if count==0:
		#print(num)
		return
	if start>end:
		return
	if end-start+1<count:
		return
	#this bit is not set ;
	gen_iteration(num,start+1,end,count)
	#this bit is set
	num[start]=1
	gen_iteration(num,start+1,end,count-1)
	num[start]=0

def gen_omega3(num,start,end,count):
	#omega = 1 
	for i in range(start,end+1):
		t=tuple([i])
		print(t)
		alloutset.add(t)
	#omega=2
	for i in range(start,end):
		for j in range(i+1,end+1):
			t=tuple([i,j])
			print(t)
			alloutset.add(t)
	#omega = 3
	for i in range(start,end-1):
		for j in range(i+1,end):
			for k in range(j+1,end+1):
				t=tuple([i,j,k])
				print(t)
				alloutset.add(t)

#======================================================================
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
	for j in range(n):
		var1=data[:,i]
		var2=data[:,j]
		#this mutual_info_score : we can use any other criteria here
		if storedval[j][i]== -2.0:
			info=metrics.normalized_mutual_info_score(var1,var2)
		else :
			info=storedval[j][i]
		storedval[i][j]=info
		print(i,j,info)
print("\n\npre processed score criteria")

m=n-1
#-------------------------
allout=[]
alloutset=set()
size=m
num=numpy.repeat(0,size)
gen_omega3(num,0,size-1,omega)
#alloutset=set(tuple(i) for i in allout)

print("\n\npre processed iteration")

#------------------------
delta=numpy.repeat(0.0,n)
w_one=numpy.repeat(0,n)

coalitioncutoff=0.5
for i in range(n):
	newdata=numpy.delete(data,i,axis=1)
	#(_,m)=numpy.shape(newdata)
	print(i)
	#print(i,"new data dimension ",m)
	for flag in alloutset:
		#print(flag,bin(flag))
		independent=0
		dependent=0
		'''
		for j in range(0,m):
			if flag[j]==1 :
				#print(j)
				#var1=data[:,i]
				#var2=newdata[:,j]
				temp=j
				if j>=i :
					temp+=1
				info=storedval[i][temp]
				#info=metrics.normalized_mutual_info_score(var1,var2)
				if info < 0.4:
					independent= independent +1
				else :
					dependent=dependent+1
		'''
		for j in flag:
			#print(j)
			#var1=data[:,i]
			#var2=newdata[:,j]
			temp=j
			if j>=i :
				temp+=1
			info=storedval[i][temp]
			#info=metrics.normalized_mutual_info_score(var1,var2)
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
	print(delta[i])
	#val=delta[i]/(1<<omega)
	#delta[i]=float(val)
	#print(delta[i])
print("delta done")
print(delta)
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
for loopval in range(1,b-1,2):
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
