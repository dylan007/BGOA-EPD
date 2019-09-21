import numpy as np
import meta
import knn

def init():
    dims = meta.dim
    ns = meta.ns
    return np.random.randint(2,size=(ns,dims))

def fitness(agents):
    F = []
    for i in range(agents.shape[0]):
        score = knn.knn(agents[i])
        R = np.count_nonzero(agents[i]==1)
        N = np.shape(agents[i])[0]
        f = meta.alpha * score + meta.beta * (R/N)
        F.append(f)
    return np.asarray(F)