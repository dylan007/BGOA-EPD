import meta
import numpy as np
import sklearn
import helper
import sys
import math
import knn

def social(x):
    x = math.fabs(x)
    return meta.f * math.exp((0-x)/meta.l) - math.exp(0-x)

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def bgoa():
    agents = helper.init()
    fitness = helper.fitness(agents)
    inds = fitness.argsort()
    agents = agents[inds]
    target = agents[0]
    maxscore = helper.fitness(np.asarray([target]))
    dist = np.zeros((meta.ns,meta.ns))
    for l in range(meta.L):
        print("Iteration : ",l+1)
        c = meta.cmax - l * (meta.cmax-meta.cmin)/meta.L
        dmin = sys.float_info.max    
        dmax = 0
        for i in range(meta.ns):
            for j in range(meta.ns):
                if j==i:
                    continue
                dist[i][j] = dist[j][i] = np.linalg.norm(agents[i]-agents[j])
                dmax = max(dist[i][j],dmax)
                dmin = min(dist[i][j],dmin)
        dist = np.interp(dist,(dmin,dmax),(1,4))
        dT = np.zeros((meta.ns,meta.dim))
        for i in range(meta.ns):
            upd = np.zeros(meta.dim)
            for d in range(meta.dim):
                for j in range(meta.ns):
                    if(i==j):
                        continue
                    upd[d] += social(math.fabs(agents[j][d] - agents[i][d])) * (agents[j][d] - agents[i][d]) / dist[i][j]
            upd = upd * (c*c*0.5)
            T = np.array([sigmoid(x) for x in upd])
            dT[i] = T
        # print("Updating Vectors")
        for i in range(meta.ns):
            for j in range(meta.dim):
                r = np.random.rand(1)
                if(r>=dT[i][j]):
                    agents[i][j] = 0
                else:
                    agents[i][j] = 1
        fitness = helper.fitness(agents)
        inds = fitness.argsort()
        score = helper.fitness(np.asarray([agents[inds[0]]]))
        if(score > maxscore):
            target = agents[inds[0]]
            maxscore = score
        print(maxscore)
    return target