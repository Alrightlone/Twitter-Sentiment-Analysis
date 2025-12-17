#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
from tqdm import tqdm  

def main():
    print("loading cooccurrence matrix")
    with open("cooc.pkl", "rb") as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for ix, jy, n in tqdm(zip(cooc.row, cooc.col, cooc.data), total=cooc.nnz, mininterval=1.0):
            
            # 1. calculate weight f(n)
            weight = (n / nmax) ** alpha if n < nmax else 1
            
            # 2. calculate current prediction error
            log_n = np.log(n)
            dot_product = np.dot(xs[ix], ys[jy])
            diff = dot_product - log_n
            
            # 3. calculate gradient
            gradient = 2 * weight * diff
            
            # 4. update vectors (SGD step)
            grad_x = gradient * ys[jy]
            grad_y = gradient * xs[ix]
            
            xs[ix] -= eta * grad_x
            ys[jy] -= eta * grad_y

    print("Saving embeddings...")
    np.save("embeddings", xs)
    print("Done!")


if __name__ == "__main__":
    main()
