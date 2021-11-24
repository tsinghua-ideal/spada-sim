from scipy.sparse.sputils import matrix
import os

matrix_mult = []
def retrieve_mm_mat(dir_fp, mat_name):
    print('---- Python Interface ----')
    import os
    import pickle
    import numpy as np
    from scipy import io as spio
    from scipy.sparse import csr_matrix, coo_matrix
    import scipy.sparse
    print(f'% Load {mat_name} from {dir_fp}')
    mat_path = os.path.join(dir_fp, mat_name + '.mtx')
    with open(mat_path, 'r') as f:
        A = spio.mmread(f).tocsr()
        if A.shape[0] == A.shape[1]:
            B = A
        else:
            B = A.transpose()
            B = B.tocsr()

        total_sum = 0
        for indptr in A.indices:
            total_sum += B.indptr[indptr+1] - B.indptr[indptr]
        matrix_mult.append(total_sum)
        print(total_sum)

def retrieve_pickled_csr(pickle_gemm_fp, pickle_gemm_name):
    print('---- Python Interface ----')
    import pickle
    import numpy as np
    from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
    print(f'% Load {pickle_gemm_name} from', pickle_gemm_fp)
    with open(pickle_gemm_fp, 'rb') as f:
        gemms = pickle.load(f)

        A, B = gemms[pickle_gemm_name]
        if isinstance(A, (csc_matrix, coo_matrix)):
            A = A.tocsr()
        elif isinstance(A, (np.ndarray)):
            A = csr_matrix(A)
        elif not isinstance(A, csr_matrix):
            raise TypeError('Unsupported matrix type: {}'.format(type(A)))

        if isinstance(B, (csc_matrix, coo_matrix)):
            B = B.tocsr()
        elif isinstance(B, (np.ndarray)):
            B = csr_matrix(B)
        elif not isinstance(B, csr_matrix):
            raise TypeError('Unsupported matrix type: {}'.format(type(B)))

        total_sum = 0
        for indptr in A.indices:
            total_sum += B.indptr[indptr+1] - B.indptr[indptr]
        matrix_mult.append(total_sum)
        print(total_sum)


ss_pref_g=('poisson3Da', 'filter3D', 'wiki-Vote', 'email-Enron', 'ca-CondMat', 'gupta2', 'Ge87H76', 'raefsky3', 'x104', 'm_t1', 'ship_001', 'msc10848', 'EternityII_Etilde', 'opt1', 'ramage02', 'nemsemm1')
ss_pref_s=('bas1lp', 'bibd_16_8', 'bundle1', 'c-64', 'c8_mat11', 'cari', 'dbir2', 'exdata_1', 'fem_filter', 'Ga10As10H30', 'heart1', 'HFE18_96_in', 'jendrec1', 'lp_fit2d', 'nd3k', 'nsct', 'orani678', 'psmigr_2', 'Si34H36', 'SiO', 'std1_Jac3', 'Trec13', 'TSOPF_FS_b162_c1', 'Zd_Jac3')
nn=('alexnetconv0', 'alexnetconv1', 'alexnetconv2', 'alexnetconv3', 'alexnetconv4', 'alexnetfc0', 'alexnetfc1', 'alexnetfc2', 'resnet50conv0', 'resnet50layer1_conv1', 'resnet50layer1_conv2', 'resnet50layer1_conv3', 'resnet50layer2_conv1', 'resnet50layer2_conv2', 'resnet50layer2_conv3', 'resnet50layer3_conv1', 'resnet50layer3_conv2', 'resnet50layer3_conv3', 'resnet50layer4_conv1', 'resnet50layer4_conv2', 'resnet50layer4_conv3', 'resnet50fc')

ss_filepath="../adaptive_sparse_accelerator/workload/suitesparse_matrices/all"
nn_filepath="../adaptive_sparse_accelerator/workload/pickled_gemms/NN_gemms_02_01_21_31.pkl"

for w in ss_pref_g:
    retrieve_mm_mat(ss_filepath, w)

for w in ss_pref_s:
    retrieve_mm_mat(ss_filepath, w)

for w in nn:
    retrieve_pickled_csr(nn_filepath, w)

retrieve_mm_mat(ss_filepath, 'lpi_forest6')

print("-----")
print(matrix_mult)