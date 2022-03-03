import os
import pickle
import numpy as np
from scipy import io as spio
from scipy.sparse import csr_matrix, coo_matrix
from scipy import sparse
import matplotlib.pyplot as plt

ss_filepath="../adaptive_sparse_accelerator/workload/suitesparse_matrices/all"
mat_name = "2cubes_sphere"
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

    prev_indices = set()
    midx_lens = []
    row_lens = []
    rows = list(range(a.get_shape()[0]))

    for r_idx in range(a.get_shape()[0]):
        row = a.getrow(r_idx)
        cur_indices = set(row.indices)
        midx_len = len(prev_indices & cur_indices)
        row_len = len(row.indices)
        midx_lens.append(midx_len)
        row_lens.append(row_len)
        prev_indices = cur_indices
        if r_idx % 10000 == 0:
            print(r_idx)

    fig = plt.figure()
    plt.plot(rows, midx_lens)
    plt.plot(rows, row_lens)
    plt.savefig('/data/lizhiyao/2cubes_sphere_idx_rlen_dist.pdf', dpi=600, format='pdf', transparent=True)
