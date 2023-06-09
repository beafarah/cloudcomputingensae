import numpy as np
import scipy as sc
import scipy.sparse as sp
import random
from collections import defaultdict
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix
from numpy import linalg as LA
from collections import Counter


def generate_sparse_matrix_dict_repr(m, n, num_nonzero):
    dict_repr = dict()
    i, j = np.random.choice(m, num_nonzero, replace=True), np.random.choice(
        n, num_nonzero, replace=True
    )
    for x, y in zip(i, j):
        dict_repr[(x, y)] = 1
    return dict_repr
	
def dot_product_with_dict_repr(dict_repr, m, n):
    multiplied_dict_repr = dict()
    for i in range(n):
        dict_col_i = {x: v for (x, y), v in dict_repr.items() if y == i}
        dict_col_i = defaultdict(int, dict_col_i)
        if dict_col_i:
            for j in range(m):
                result = 0
                dict_col_j = {
                    x: v
                    for (x, y), v in dict_repr.items()
                    if y == j and x in dict_col_i.keys()
                }
                dict_col_j = defaultdict(int, dict_col_j)
                if dict_col_j:
                    for x, v in dict_col_i.items():
                        result += v * dict_col_j[x]
                    if result != 0:
                        multiplied_dict_repr[(i, j)] = result
    return multiplied_dict_repr
	
def dict_repr_to_sp_csc(dict_rep, m, n):
    row = np.array([])
    col = np.array([])
    data = np.array([])

    for (x, y), v in dict_rep.items():
        row = np.append(row, x)
        col = np.append(col, y)
        data = np.append(data, v)
    return sp.csc_matrix((data, (row, col)), shape=(m, n))
	
# https://stackoverflow.com/questions/45881580/pyspark-rdd-sparse-matrix-multiplication-from-scala-to-python
def coordinateMatrixMultiply(leftmatrix, rightmatrix):
    left = leftmatrix.entries.map(lambda e: (e.j, (e.i, e.value)))
    right = rightmatrix.entries.map(lambda e: (e.i, (e.j, e.value)))
    productEntries = (
        left.join(right)
        .map(lambda e: ((e[1][0][0], e[1][1][0]), (e[1][0][1] * e[1][1][1])))
        .reduceByKey(lambda x, y: x + y)
        .map(lambda e: (*e[0], e[1]))
    )
    return productEntries
	
def list_repr_to_sp_csc(list_rep, m, n):
    row = np.array([])
    col = np.array([])
    data = np.array([])

    for (x, y, v) in list_rep:
        row = np.append(row, x)
        col = np.append(col, y)
        data = np.append(data, v)
    return sp.csc_matrix((data, (row, col)), shape=(m, n))

def to_list(a):
    return [a]
	
def append(a, b):
    a.append(b)
    return a
	
def extend(a, b):
    a.extend(b)
    return a
	
def mapper(aij, aik, cj_norm, ck_norm, gamma=1.0):
    if random.randint(0, 1) >= min(1.0, gamma / (ck_norm * cj_norm)):
        return aij * aik
	
def dimsum_algorithm(mat,number_of_simulations:int=10,gamma:float=0.5) -> Counter:                     
    NUMBER_OF_SIMULATIONS=number_of_simulations
    GAMMA=gamma
    norm_cols = (
        mat.transpose()
        .entries.map(lambda e: (e.i, (e.j, e.value)))
        .combineByKey(to_list, append, extend)
        .map(lambda e: (e[0], LA.norm(list(map(lambda x: x[1], e[1])))))
    )

    left = (
        mat.transpose()
        .entries.map(lambda e: (e.i, (e.j, e.value)))
        .leftOuterJoin(norm_cols)
        .map(lambda e: (e[1][0][0], (e[0], e[1][0][1], e[1][1])))
    )

    right = (
        mat.entries.map(lambda e: (e.j, (e.i, e.j, e.value)))
        .leftOuterJoin(norm_cols)
        .map(lambda e: (e[1][0][0], (e[1][0][1], e[1][0][2], e[1][1])))
    )

    productEntriesMap = (
        left.join(right)
        .map(
            lambda e: (
                (e[1][0][0], e[1][1][0]),
                mapper(e[1][0][1], e[1][1][1], e[1][0][2], e[1][1][2], gamma=GAMMA),
            )
        )
        .filter(lambda e: e[1] is not None)
    )
    list_multiple_results_dimsum=[]
    for _ in range(NUMBER_OF_SIMULATIONS):
        final_product = (
            productEntriesMap.map(lambda e: (e[0][0], (e[0][1], e[1])))
            .leftOuterJoin(norm_cols)
            .map(lambda e: (e[1][0][0], ((e[0], e[1][0][0]), e[1][0][1], e[1][1])))
            .leftOuterJoin(norm_cols)
            .map(lambda e: ((*e[1][0][0], e[1][0][2] * e[1][1]), e[1][0][1]))
            .reduceByKey(lambda x, y: x + y)
            .map(lambda e: ((e[0][0], e[0][1]), e[1] / min(GAMMA, e[0][2])))
        )
        list_dimsum=sorted(final_product.collect())
        list_multiple_results_dimsum.append(list_dimsum)

    counter=Counter()
    for result in list_multiple_results_dimsum:
        counter+=Counter({k:v for (k,v) in result})
    for k in counter.keys():
        counter[k] /= NUMBER_OF_SIMULATIONS
    return counter