# Helpful functions 
import numpy as np
import random
def sparse_generator(m,n,L=1):
    """
    Function that generates sparse matrix with n rows and m columns, each row having at most L nonzero entries
    input:
    m: number of rows
    n: number of columns
    L: number of nonzero entries per row
    output:
    random sparse matrix A
    """
    A = np.zeros((m,n))
    for row in range(m): 
        new_value = [2*random.random()-1 for j in range(L)] # random numbers in [-1,1]
        index = random.sample(range(n),L) # indexes of the nonzero values
        for (v,i) in zip(new_value, index):
            A[row,i] = v 
    return A

def naive_mult_np(A): # function that depends on numpy (because of dot product)
    """
    Function that computes sparse matrix multiplication
    input: 
    A: sparse matrix
    output:
    A^T A: sparse matrix product
    """
    return A.T@A

def naive_mult(A): # function that doesnt depend on numpy
    """
    Function that computes sparse matrix multiplication
    input: 
    A: sparse matrix
    output:
    A^T A: sparse matrix product
    """
    (m,n) = A.shape
    AProd = sum( [A[i][0]*A[i] for i in range(len(A))] )
    for r in range(1,n):
        AProd = np.vstack((AProd,sum( [A[i][r]*A[i] for i in range(len(A))] ) ))
    return AProd