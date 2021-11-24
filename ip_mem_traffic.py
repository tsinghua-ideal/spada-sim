import os
import pickle
import numpy as np
from scipy import io as spio
from scipy.sparse import csr_matrix, coo_matrix
from scipy import sparse

ss_filepath="../adaptive_sparse_accelerator/workload/suitesparse_matrices/all"
mat_name = "EternityII_Etilde"
mat_path = os.path.join(ss_filepath, mat_name + '.mtx')
with open(mat_path, 'r') as f:
    a = spio.mmread(f).tocsr()
    if a.shape[0] == a.shape[1]:
        b = a
    else:
        b = a.transpose()
        b = b.tocsr()

    a = sparse.csr_matrix(a)
    b = sparse.csc_matrix(b)
    print(a.shape, b.shape)

    a_traffic = a.getnnz() * 2
    b_traffic = 0
    c_traffic = (a @ b).getnnz() * 2
    b_cache = []
    b_cache_occp = 0
    b_cache_size = 196608

    for r_idx in range(a.get_shape()[0]):
        row = a.getrow(r_idx)
        for b_col_idx in row.indices:
            if b_col_idx >= b.shape[1]:
                continue
            b_col_size = b.getcol(b_col_idx).indices.size * 2
            if b_col_idx not in b_cache:
                while b_cache_occp + b_col_size > b_cache_size:
                    rm_b_col_idx = b_cache[0]
                    b_cache.pop(0)
                    b_cache_occp -= b.getcol(rm_b_col_idx).indices.size * 2
                b_cache_occp += b_col_size
                b_cache.append(b_col_idx)
                b_traffic += b_col_size * 2
        if r_idx % 1000 == 0:
            print("arow:", r_idx)
            print("a trrafic", a_traffic)
            print("b traffic:", b_traffic)
            print("c traffic:", c_traffic)

    print("a trrafic", a_traffic)
    print("b traffic:", b_traffic)
    print("c traffic:", c_traffic)