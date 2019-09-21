import numpy as np 
import load 
import meta
from sklearn.neighbors import KNeighborsClassifier

def knn(features):
    X_train,X_test,Y_train,Y_test = load.load()
    delcols = []
    for i in range(meta.dim):
        if(features[i]==0):
            delcols.append(i)
    np.delete(X_train,delcols,axis=1)
    np.delete(X_test,delcols,axis=1)
    knn = KNeighborsClassifier(n_neighbors=meta.dim)
    knn.fit(X_train,Y_train)
    return knn.score(X_test,Y_test)    