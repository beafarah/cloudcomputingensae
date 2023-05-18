# Helpful functions for the implementation using Python and Numpy
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
        new_value = [2*random.random()-1 for j in range(L)] # random numbers in [-1,1]. we'll have at most L nonzero entries in each row
        index = random.sample(range(n),L) # indexes of the nonzero values
        for (v,i) in zip(new_value, index):
            A[row,i] = v 
    return A



def norm(A):
    """
    Function that computes the norm 2 of each column of a matrix
    input:
    A: matrix
    output:
    norm: array of norms of each column
    """
    norm = (np.square(A).sum(axis=0))**(1/2) 
    return norm