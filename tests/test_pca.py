import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


def standardize(x: np.ndarray):
    n, _ = x.shape
    mean_vector = np.sum(x, axis=0) / n
    centered_matrix = x - mean_vector  # numpy broadcasting
    covmat = np.matmul(centered_matrix.transpose(), centered_matrix) / (n-1)
    standard_errors = np.sqrt(covmat.diagonal())
    return centered_matrix / standard_errors


navy_df = pd.read_csv(
    'navy.dat',
    header=None,
    index_col=False,
    delim_whitespace=True)
navy_df.columns = [
    'ID',
    'ADO',  # avg daily occupancy
    'MAC',  # avg number of check-ins
    'WHR',  # weekly hrs of service desk operation
    'CUA',  # sq ft of common use area
    'WINGS',  # number of building wings
    'OBC',  # operational berthing capacity
    'RMS',  # number of rooms
    'MMH'  # monthly man-hours required to operate
]
navy_np = np.array(navy_df.iloc[:, 1:])  # exclude ID column
# navy_np = standardize(navy_np)
pca = PCA()
pca_fit = pca.fit(navy_np)

eigval = pd.DataFrame()
eigval["EVal"] = pca.explained_variance_
eigval["prop"] = np.round(pca.explained_variance_ratio_, 4)
eigval["cumm"] = np.cumsum(eigval.prop)
print(eigval)

pcs = pd.DataFrame(pca.transform(navy_np)).iloc[:,0]
pcs.columns = ['pc1']
print(pcs)