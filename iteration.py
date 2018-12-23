import numpy
#======================

def func(num,start,end,count):
	if count>=0:
		#print(num)
		num1=[]
		num1=num.copy()
		allout.append(num1)
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
	func(num,start+1,end,count)
	#this bit is set
	num[start]=1
	func(num,start+1,end,count-1)
	num[start]=0

#=========================
size=5
num1=numpy.repeat(0,size)
omega=3
allout=[]
func(num1,0,size-1,omega)
#for i in allout:
#	print(i)
alloutset=set(tuple(i) for i in allout)
for flag in alloutset:
	print(flag)
	for i in range(size):
		if flag[i]==1:
			print(i,end=' ')
	print()