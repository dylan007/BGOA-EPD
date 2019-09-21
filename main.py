import numpy as np
import load 
import knn
import helper
import bgoa
import meta

if __name__ == "__main__":
    target = bgoa.bgoa()
    bgoa_score = knn.knn(target)
    naive_score = knn.knn(np.ones(meta.dim))
    print("BGOA Classifier: " + str(bgoa_score) + "\nNaive Classifier: " + str(naive_score))