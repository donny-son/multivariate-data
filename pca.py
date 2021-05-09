import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
from numpy.core.fromnumeric import diagonal
from pandas import DataFrame


def get_centered_matrix(x: np.ndarray):
    n, _ = x.shape
    mean_vector = np.sum(x, axis=0) / n
    result = x - mean_vector  # numpy broadcasting
    return result


def get_covariance_matrix(x: np.ndarray):
    n, _ = x.shape
    centered_matrix = get_centered_matrix(x)
    return np.matmul(centered_matrix.transpose(), centered_matrix) / (n-1)


def get_correlation_matrix(x: np.ndarray):
    covmat = get_covariance_matrix(x)
    standard_errors = np.sqrt(covmat.diagonal())
    return covmat / np.outer(standard_errors, standard_errors)


def get_proportion_of_variance(x: np.ndarray):
    result = {}
    mat = get_correlation_matrix(x)
    trace = np.trace(mat)
    eigenval, eigenvec = np.linalg.eig(mat)
    proportion_of_varaince = (eigenval / trace).tolist()
    cum_proportion = [reduce(lambda x, y: x+y, proportion_of_varaince[:i+1], 0)
                      for i, _ in enumerate(proportion_of_varaince)]
    for i, v in enumerate(cum_proportion):
        result[i] = {
            "cummulated-proportion": v,
            "eigenvalue": eigenval[i],
            "eigenvector": eigenvec[i],
        }
    return result


def get_principle_component_scores(x: np.ndarray, eigenvalueCut=1):
    result = []
    proportion_dict = get_proportion_of_variance(x)
    centered_x = get_centered_matrix(x)
    for i, v in proportion_dict.items():
        if v["eigenvalue"] > eigenvalueCut:
            score = np.matmul(centered_x, v["eigenvector"])
            result.append(score)
    return np.array(result).transpose()


def compare_correlations_principle_component_scores_and_variables(df: pd.DataFrame, eigenvalueCut=1):
    ary = np.array(df)
    _, num_vars = ary.shape
    pcs = get_principle_component_scores(ary, eigenvalueCut)
    pcs_df = pd.DataFrame(pcs)
    _, num_pc = pcs_df.shape
    pcs_colnames = ['pc'+f'{i}' for i in range(num_pc)]
    pcs_df.columns = pcs_colnames
    corr_mat = pd.concat([df, pcs_df], axis=1).corr()
    return corr_mat.iloc[:num_vars, num_vars:]


def draw_scree_plot(x: np.ndarray):
    mat = get_correlation_matrix(x)
    eigenval, _ = np.linalg.eig(mat)
    import matplotlib.pyplot as plt
    plt.title("SCREE PLOT OF EIGENVALUES")
    plt.xlabel("index")
    plt.plot(eigenval, 'o-')
    plt.show()


def get_zero_containing_variables(df: pd.DataFrame):
    zero_truth_series = (df == 0).sum() > 0
    zero_containing_colnames = zero_truth_series.index[zero_truth_series == True].to_list(
    )
    return zero_containing_colnames


def add_one_to_zero_containing_variables(df: pd.DataFrame):
    cols = get_zero_containing_variables(df)
    df[cols] += 1


def log_transformation(df: pd.DataFrame):
    add_one_to_zero_containing_variables(df)
    new_colnames = ['log'+n for n in df.columns.to_list()]
    transformed_df = np.log(df)
    transformed_df.columns = new_colnames
    return transformed_df


if __name__ == "__main__":
    import pandas as pd
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
    navy_df = navy_df.iloc[:, 1:]
    navy_df.head(1)  # 25 obs, 8 features(p)
    logNavy = log_transformation(navy_df)
    logNavy_array = np.array(logNavy)
    from pprint import pprint as prnt
    prnt(get_proportion_of_variance(logNavy_array))
    # selects only one principle component
    pcs = get_principle_component_scores(logNavy_array, eigenvalueCut=1)
    print(compare_correlations_principle_component_scores_and_variables(
        logNavy, eigenvalueCut=1)
    )
    import scipy.stats as stats
    pcs_flattened = pcs.flatten()
    from scipy.stats import kstest, shapiro, anderson, cramervonmises
    x = pcs_flattened
    m = x.mean()
    s = x.std()
    print("Shapiro-Wilk: ", shapiro(x))
    print("KS test: ", kstest(x, 'norm', args=(m, s)))
    print("Anderson: ", anderson(x, 'norm'))
    print("Cramer Von Mises: ", cramervonmises(x, "norm", args=(m, s)))
    (stats.probplot(pcs_flattened, plot=plt))
