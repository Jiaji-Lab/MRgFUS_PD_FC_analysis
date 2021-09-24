#%%
from scipy.io import loadmat
import numpy as np
import os

def readin_NBS_subnetwork(mat_path, index=0):
    mat = loadmat(mat_path)
    subnetwork = mat['nbs']['NBS'][0][0]['con_mat'][0][0][0][index]
    return subnetwork

def readin_NBS_test_stats(mat_path, index=0):
    mat = loadmat(mat_path)
    test_stats = mat['nbs']['NBS'][0][0]['test_stat'][0][0]
    return test_stats

def extract_subnetwork_nodes(subnetwork):
    row_nodes, col_nodes = subnetwork.nonzero()
    all_nodes = np.concatenate((row_nodes, col_nodes), axis=0)
    all_nodes_unique = np.unique(all_nodes)
    return all_nodes_unique, row_nodes, col_nodes

def extract_subnetwork_edges(subnetwork):
    row_nodes, col_nodes = subnetwork.nonzero()
    return zip(row_nodes, col_nodes)

def extract_submatrix(matrix, nodes):
    return matrix[np.ix_(nodes, nodes)]

#%%
"""
out_path = r'G:\006pd_DTI\03_NBS_data\01_base_vs_360d\left+right\forbrantnetviewer'
subnetwork = readin_NBS_subnetwork(r'G:\006pd_DTI\03_NBS_data\01_base_vs_360d\left+right\forbrantnetviewer\c-1_t2.mat')
all_nodes_unique, row_nodes, col_nodes = extract_subnetwork_nodes(subnetwork)

matrix246 = np.zeros(shape=(246, 246))
for row_node, col_node in zip(row_nodes, col_nodes):
    matrix246[row_node][col_node] = 1
    matrix246[col_node][row_node] = 1

submatrix = extract_submatrix(matrix246, all_nodes_unique)

np.savetxt(os.path.join(out_path, '246matrix.txt'), matrix246, fmt='%1d')
np.savetxt(os.path.join(out_path, 'submatrix.txt'), submatrix, fmt='%1d')
"""
#%%
"""
import seaborn as sns
subjects = load_subjects(r'G:\006pd_DTI\subject_info.csv')
max_num = 0
for subject in subjects:
    for observation in subject.get_all_observation():
        dti = observation.dti
        network = dti.get_network_fa()
        flatten_matrix = network.flatten()
        flatten_matrix[flatten_matrix<=500] = 0
        flatten_matrix[flatten_matrix>500] = 1
        print(np.sum(flatten_matrix))
        m = np.max(network)
        if m > max_num:
            max_num = m
        print('{}:{} num:{}'.format(subject.name,
                                    observation.name,
                                    m))
max_num
#%%
#全脑的转化
n = 500
out_path = r'G:\006pd_DTI\04_structural_network_analysis\02_num030d_subnetwork'
subnetwork = readin_NBS_subnetwork(r'G:\006pd_DTI\04_structural_network_analysis\02_num030d_subnetwork\basevs04_t2_68nodes_c1.mat')
all_nodes_unique, *_ = extract_subnetwork_nodes(subnetwork)

for subject in subjects:
    for observation in subject.get_all_observation():
        dti = observation.dti
        network = dti.get_network_num()
        network = network / n
        np.fill_diagonal(network, 0)
        to_path = os.path.join(out_path, '{}_{}_num.txt'.format(subject.name, observation.name))
        np.savetxt(to_path, network, fmt='%10f')
# %%
#num子网络的转化
n = 500
out_path = r'G:\006pd_DTI\04_structural_network_analysis\04_fa360d_subnetwork\3.0'
subnetwork = readin_NBS_subnetwork(r'G:\006pd_DTI\04_structural_network_analysis\04_fa360d_subnetwork\3.0\base_360_-1_t3.0.mat')
all_nodes_unique, *_ = extract_subnetwork_nodes(subnetwork)

for subject in subjects:
    for observation in subject.get_all_observation():
        dti = observation.dti
        network = dti.get_network_num()
        submatrix = extract_submatrix(network, all_nodes_unique)
        submatrix = submatrix / n
        np.fill_diagonal(submatrix, 0)
        to_path = os.path.join(out_path, '{}_{}_num.txt'.format(subject.name, observation.name))
        np.savetxt(to_path, submatrix, fmt='%10f')
# %%
#fa子网络的转化
n = 500
out_path = r'G:\006pd_DTI\04_structural_network_analysis\04_fa360d_subnetwork\3.0'
subnetwork = readin_NBS_subnetwork(r'G:\006pd_DTI\04_structural_network_analysis\04_fa360d_subnetwork\3.0\base_360_-1_t3.0.mat')
all_nodes_unique, *_ = extract_subnetwork_nodes(subnetwork)

for subject in subjects:
    for observation in subject.get_all_observation():
        dti = observation.dti
        network = dti.get_network_num()#fa子网络的转化
        submatrix = extract_submatrix(network, all_nodes_unique)
        submatrix = submatrix / n
        np.fill_diagonal(submatrix, 0)
        to_path = os.path.join(out_path, '{}_{}_num.txt'.format(subject.name, observation.name))#fa子网络的转化
        np.savetxt(to_path, submatrix, fmt='%10f')
"""