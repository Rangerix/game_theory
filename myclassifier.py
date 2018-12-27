import pandas
import numpy
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  


infile=sys.argv[1]
dataframe=pandas.read_csv(infile)

(a,b)= numpy.shape(dataframe)
print (a)
print (b)
X = dataframe.values[:,0:b-1]
y = dataframe.values[:,b-1]
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.1)


clf=RandomForestClassifier(n_estimators=300)
clf.fit(X_train,y_train)
val=clf.score(X_test,y_test)
print(val)
'''
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(X_train, y_train)
val=classifier.score(X_test,y_test)
print(val)
'''