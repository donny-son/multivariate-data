#  _____                                _       _____
# |  __ \                              | |     / ____|
# | |  | | ___  _ __   __ _  ___   ___ | | __ | (___   ___  _ __
# | |  | |/ _ \| '_ \ / _` |/ _ \ / _ \| |/ /  \___ \ / _ \| '_ \
# | |__| | (_) | | | | (_| | (_) | (_) |   <   ____) | (_) | | | |
# |_____/ \___/|_| |_|\__, |\___/ \___/|_|\_\ |_____/ \___/|_| |_|
#                      __/ |
#                     |___/
import numpy as np
from scipy.stats import chi2, f, t
from Multivar import MultivariateData


def two_population_mean_comparison(multivardata1: MultivariateData, multivardata2: MultivariateData, test_only=False, alpha=0.05):
    """
    Compare means between Multivariate Data from two populations

    params:
        multivardata1: MultivariateData from first population
        multivardata2: MultivariateData from second population
        alpha: 1-significant level
        return_constant: return const if true
            bool
    result:
        float: f-statistic value
        tuple: (int, int)
        returns: miscalaeneous parameters
    """
    assert isinstance(multivardata1, MultivariateData)
    assert isinstance(multivardata2, MultivariateData)
    assert multivardata1.p == multivardata2.p, f"Dimension Error: {multivardata1.p} != {multivardata2.p}"

    results = {}
    significance = 1-alpha
    n1 = multivardata1.n
    results['n1'] = n1
    n2 = multivardata2.n
    results['n2'] = n2
    p = multivardata1.p
    results['p'] = p
    mean1 = multivardata1.mean_vector
    results['mean1'] = mean1
    mean2 = multivardata2.mean_vector
    results['mean2'] = mean2
    mean_diff = mean1 - mean2
    cov1 = multivardata1.covariance_matrix
    results['cov1'] = cov1
    cov2 = multivardata2.covariance_matrix
    results['cov2'] = cov2
    s_p = ((n1-1)*cov1 + (n2-1)*cov2) / (n1 + n2 - 2)
    results['s_p'] = s_p
    t_sqrd = ((n1 * n2) / (n1 + n2)) * \
        np.matmul(np.matmul(mean_diff, np.linalg.inv(s_p)), mean_diff)
    const = (n1 + n2 - p - 1) / (p*(n1 + n2 - 2))
    f_statistic = const * t_sqrd
    results['f-statistic'] = f_statistic
    deg_free = (p, n1 + n2 - p - 1)
    results['df'] = deg_free
    c_sqrd = 1/const * f.ppf(significance, deg_free[0], deg_free[1])
    results['c_sqrd'] = c_sqrd
    if test_only:
        print(f"---------------------HOTELLING'S T^2 TEST----------------------")
        print(
            f"Null Hypothesis:\n  Mean vector {mean1}\n  is equal to {mean2}")
        print(f"Distribution: F{deg_free}")
        print(f"F statistic: {f_statistic}")
        p_value = 1 - f.cdf(f_statistic, deg_free[0], deg_free[1])
        print(f"Significance: {significance*100}%")
        print(f"P-value: {p_value}")
        if p_value < alpha:
            print(f"Conclusion: REJECT the null hypothesis")
        else:
            print(f"Conclusion: DO NOT reject the null hypothesis")
        print(f"---------------------------------------------------------------")
        return
    return results


def ellipsoid_info(m1: MultivariateData, m2: MultivariateData, alpha=0.05):
    """
    returns ellipsoid information for two multivariate data samples from separate populations

    params:
        m1: MultivariateData from first population
        m2: MultivariateData from second population
        alpha: 1-significance level

    return:
        dict: contains axis and length information. Degree of freedom is derived from mean_comparison
        Example {
            "axis": (floats...),
            "length": (floats...)
        }
    """
    params = two_population_mean_comparison(
        m1, m2, test_only=False, alpha=alpha)
    n1 = params['n1']
    n2 = params['n2']
    s_p = params['s_p']
    c_sqrd = params['c_sqrd']
    result = {}
    significance = 1-alpha
    eigenvalues, eigenvectors = np.linalg.eig(s_p)
    for i, lmbda in enumerate(eigenvalues):
        conf_half_len = np.sqrt(lmbda) * np.sqrt((1/n1 + 1/n2) * c_sqrd)
        conf_axe_abs = conf_half_len * eigenvectors[i]
        result[i] = {
            "axis": conf_axe_abs,
            "length": conf_half_len * 2
        }
    return result


def component_means_confidence_interval(m1: MultivariateData, m2: MultivariateData, is_bonferroni=False, alpha=0.05):
    """
    returns lower and upperbounds of component means

    params:
        m1: MultivariateData from first population
        m2: MultivariateData from second population
        is_bonferroni: use bonferroni method if true, standard method using sqrt of c_sqrt if false.
        alpha: 1-significance level

    return:
        dict: lower and upperbounds of features
        Example {
            "feature1": {
                "ub": float,
                "lb": float,
            },
            "feature2": {...},
            ...
        }
    """
    result = {}
    params = two_population_mean_comparison(
        m1, m2, test_only=False, alpha=alpha)
    c = np.sqrt(params['c_sqrd'])
    p = params['p']
    n1 = params['n1']
    n2 = params['n2']
    s_p = params['s_p']
    mean1 = params['mean1']
    mean2 = params['mean2']
    mean_diff = mean1 - mean2
    if not is_bonferroni:
        for i in range(p):
            ci = {
                'ub': mean_diff[i] + c * np.sqrt((1/n1 + 1/n2) * s_p[i, i]),
                'lb': mean_diff[i] - c * np.sqrt((1/n1 + 1/n2) * s_p[i, i])
            }
            result[f"feature{i+1}"] = ci
    else:
        for i in range(p):
            ci = {
                'ub': mean_diff[i] + t.ppf(1 - alpha/(2*p), n1+n2-2) * np.sqrt((1/n1 + 1/n2) * s_p[i, i]),
                'lb': mean_diff[i] - t.ppf(1 - alpha/(2*p), n1+n2-2) * np.sqrt((1/n1 + 1/n2) * s_p[i, i])
            }
            result[f"feature{i+1}"] = ci
    return result


def two_population_profile_analysis(m1: MultivariateData, m2: MultivariateData, method="parallel", alpha=0.05):
    """
    conduct profile analysis between two multivariate data derived from two populations

    params:
        m1: multivariate data from population1
        m2: multivariate data from population2
        method: "parallel" or "coincident"(also means flat)
            str
        alpha: 1 - significance
    """
    def __stats_calc(c_mat, mean_difference, n1, n2, s_p):
        c_matXmean_diff = np.matmul(c_mat, mean_difference)

        if c_mat.shape != (p,):
            middle_term = np.linalg.inv(
                (1/n1 + 1/n2)*np.matmul(np.matmul(c_mat, s_p), np.transpose(c_mat)))
            t_statistic = np.matmul(np.matmul(np.transpose(
                c_matXmean_diff), middle_term), c_matXmean_diff)
            df = (p-1, n1+n2-p)
            d_sqrd = ((n1 + n2 - 2) * (p-1) / (n1 + n2 - p)) * \
                f.ppf(significance, df[0], df[1])
        else:  # when middle term is constant
            middle_term = 1 / \
                ((1/n1 + 1/n2)*np.matmul(np.matmul(c_mat, s_p), np.transpose(c_mat)))
            t_statistic = c_matXmean_diff * middle_term * c_matXmean_diff
            df = (1, n1 + n2 - 2)
            d_sqrd = f.ppf(significance, df[0], df[1])
        return t_statistic, df, d_sqrd

    def __get_parallel_c_matrix(p):
        minus_one_matrix = np.delete(
            np.hstack((np.zeros((p, 1)), -np.identity(p))), -1, 1)
        identity_matrix = np.identity(p)
        return np.delete(identity_matrix + minus_one_matrix, -1, 0)

    params = two_population_mean_comparison(
        m1, m2, test_only=False, alpha=alpha)
    p = params['p']
    n1 = params['n1']
    n2 = params['n2']
    s_p = params['s_p']
    mean1 = params['mean1']
    mean2 = params['mean2']
    mean_diff = mean1 - mean2
    significance = 1-alpha

    if method == "parallel":
        c_matrix = __get_parallel_c_matrix(p)
        t_statistic, df, d_sqrd = __stats_calc(
            c_matrix, mean_diff, n1, n2, s_p)

    elif method == "coincident":
        c_matrix = np.transpose(np.ones(p))
        t_statistic, df, d_sqrd = __stats_calc(
            c_matrix, mean_diff, n1, n2, s_p)

    print(f"------------------------PROFILE ANALYSIS-------------------------")
    print(f"C-matrix: \n{c_matrix}")
    print(f"Mean vector | pop1: {mean1}")
    print(f"Mean vector | pop2: {mean2}")
    print(
        f"Null Hypothesis:\n {np.matmul(c_matrix, mean1)} is equal to\n {np.matmul(c_matrix, mean2)}")
    print(f"Distribution: F{df}")
    print(f"T^2 Statistic: {t_statistic}")
    print(f"d^2: {d_sqrd}")
    print(f"Significance: {significance*100}%")
    if t_statistic > d_sqrd:
        print(f"Conclusion: REJECT the null hypothesis")
    else:
        print(f"Conclusion: DO NOT reject the null hypothesis")
    print(f"-----------------------------------------------------------------")
    return


if __name__ == "__main__":
    import pprint
    import pandas as pd
    turtle_df = pd.read_csv(
        'turtle.dat',
        header=None,
        index_col=False,
        delim_whitespace=True)
    turtle_df.columns = ['x1', 'x2', 'x3', 'gender']
    fem = MultivariateData(
        turtle_df[turtle_df['gender'] == 'female'].iloc[:, 0:3])
    mal = MultivariateData(
        turtle_df[turtle_df['gender'] == 'male'].iloc[:, 0:3])
    two_population_profile_analysis(fem, mal, method="coincident")
