import numpy as np
import random

def NaiveMapper(ri):
    """
    Mapper function to naive matrix multiplication
    input: 
    ri: arrays of a row i of the sparse matrix A
    output:
    AProd: matrix with dot products between pairs of the row ri
    """
    AProd =  [ri[i]*ri for i in range(len(ri))]  # we walk through the rows

    return np.matrix(AProd)


def NaiveReducer(col_maps):
    """
    Reducer function to naive matrix multiplication
    input:
    col_maps: list of mappings
    output:
    result_mat = matrix with the sum of mappings
    """
    m = len(col_maps)
    (nrow, ncol) = col_maps[0].shape
    result_mat = np.zeros((nrow, ncol) )
    for lm in col_maps:
        result_mat+=np.matrix(lm)
    return result_mat


def NaiveProd(A):
    """
    Function that computes A^T A in a naive way
    input:
    A: sparse matrix
    output: matrix product A^T A
    """
    (m,n) = A.shape
    rows = [A[i] for i in range(m) ]
    col_maps = [NaiveMapper(rows[j]) for j in range(m)]
    return NaiveReducer(col_maps)


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

def DIMSUMMApper(A, ind, gamma=0.5, nb_it = 1000):
    """
    Efficient Mapper
    input:
    A: sparse matrix
    ind: index of the row that we wish to compute (ri is the i-th row of the matrix, so ind = i)
    gamma: hyperparameter
    nb_it: number of iterations 
    output:
    mean of all the matrices computed
    """
    results = []
    it = 0
    while it<nb_it:
        it+=1
        (m,n) = A.shape 
        ri = A[ind]
        Anorm = norm(A) # norm of columns of A
        AProd = np.zeros((n,n))
        for ci in range(n):
            for cj in range(n):
                random_probs = random.random()  # generate random probabilities
                prob =  min(1,gamma/(Anorm[ci] * Anorm[cj]))
                if prob>random_probs:
                    AProd[ci,cj] += ri[ci]*ri[cj]
                    
        results.append(AProd)
    
    results = np.array(results)
    return results.sum(axis=0)/(len(results))


def DIMSUMReducer(A, col_maps, gamma = 0.5):
    """
    Efficient Reducer
    input:
    A: sparse matrix
    col_maps: list of matrices from the mapper function (for each row of A)
    gamma: hyperparameter
    output:
    result_mat: matrix of size (n,n)
    """
    (nrow, ncol) = col_maps[0].shape
    result_mat = np.zeros((nrow, ncol) )
    Anorm = norm(A) 
    result_mat = sum(col_maps)
    #for lm in col_maps:
        #result_mat+=np.matrix(lm)
    for i in range(nrow):
        for j in range(ncol):
            if gamma> Anorm[i]*Anorm[j]:
                result_mat[i][j]=1/(Anorm[i]*Anorm[j]) * result_mat[i][j]
            else:
                result_mat[i][j]= (1/gamma) * result_mat[i][j]
    return result_mat


def DIMSUMProd(A, gamma = 0.5, nb_it = 1500):
    """
    Function that efficiently computes A^T A for sparse matrix
    input:
    A: sparse matrix
    gamma: hyperparameter
    nb_it: number of iterations for the mapper function
    output:
    prod: matrix equal to A^T A
    """
    (m,n) = A.shape
    col_maps = [DIMSUMMApper(A,j, gamma, nb_it) for j in range(m)]
    prod = DIMSUMReducer(A, col_maps)
    prod[np.isnan(prod)] = 0
    return prod