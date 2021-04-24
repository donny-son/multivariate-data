from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import chi2, f, t
import matplotlib.pyplot as plt
import pprint
try:
    import termplotlib as tpl
except Exception as e:
    print(f"termploblib is not installed.\nUsing matplotlib as default.")


class MultivariateData:
    """
        Object for computing multivariate data

        Attributes:
            data (np.array): input data
            n, (int):
            p (int):
            mean_vector (np.array):
            covariance_matrix (np.array):

        Args:
            inputdata (np.array, list, tuple, ...): any iterable object that numpy supports
    """

    def __init__(self, inputdata) -> None:
        self.data = np.array(inputdata)
        self.n, self.p = self.data.shape
        self.mean_vector = np.mean(self.data, axis=0)
        self.covariance_matrix = np.cov(self.data.transpose())

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            return MultivariateData(self.data - other)
        elif isinstance(other, MultivariateData):
            return MultivariateData(self.data - other.data)
        else:
            raise ValueError("Object must be np.array or MultivariateData")

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return MultivariateData(self.data + other)
        elif isinstance(other, MultivariateData):
            return MultivariateData(self.data + other.data)
        else:
            raise ValueError("Object must be np.array or MultivariateData")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return MultivariateData(self.data * other)
        elif isinstance(other, MultivariateData):
            if self.p == other.n:
                return MultivariateData(np.matmul(self.data, other.data))
            else:
                raise ValueError("Dimension does not match.")
        elif isinstance(other, np.ndarray):
            return MultivariateData(np.matmul(self.data, other))
        else:
            raise TypeError("Unsupported operation between types")

    def __repr__(self) -> str:
        return f"MultivariateData(SampleSize:{self.n}, Features:{self.p})"

    def append(self, other, orientation: str = 'h'):
        """Appends MultivariateData in given orientation

        Args:
            other (MultivariateData): Other multivariate object
            orientation (str): 'h' for horizontal, 'v' for vertical
        """
        assert isinstance(other, MultivariateData)
        axis = 1 if orientation == 'v' else 0
        return MultivariateData(np.concatenate((self.data, other.data), axis=axis))

    def generalized_squared_distance(self) -> list:
        result = []
        inv_cov = np.linalg.inv(self.covariance_matrix)
        for row in self.data:
            diff = row - self.mean_vector
            # numpy broadcasting
            result.append(np.matmul(np.matmul(diff, inv_cov), diff))
        assert len(result) == self.n
        return result

    def __get_qq_tuples(self) -> list:
        result = []
        sorted_general_distance = sorted(self.generalized_squared_distance())
        for i, x in enumerate(sorted_general_distance):
            x_probability_value = (i+1 - 0.5) / self.n
            q_value = chi2.ppf(x_probability_value, self.p)
            result.append(
                (q_value, x)
            )
        return result

    def qqplot(self, terminal=False):
        """Draws qqplot for Multivariate Data

        Args:
            terminal (bool, optional): [Option for drawing the qqplot in terminal].
            If False -> draws via matplotlib
        """
        qq_tuples = self.__get_qq_tuples()
        x = [x for x, _ in qq_tuples]
        y = [y for _, y in qq_tuples]
        if terminal:
            fig = tpl.figure()
            fig.plot(x, y, width=60, height=20)
            fig.show()
        else:
            plt.scatter(x, y)
            plt.show()

    def hotellings_t_test(self, mu_vector_null, alpha=0.05, method="p"):
        """Performs Hotellings test for mean comparison, via adjusted F distribution

        Args:
            mu_vector_null ([int, float]): vector of mean under the null hypothesis
            alpha (float, optional): 1-alpha = Significance level. Defaults to 0.05.
            method (str, optional): Method of testing. Either 'p' or 'critical'. Defaults to "p".
        """
        significance = 1-alpha
        assert (isinstance(mu_vector_null, list)
                or isinstance(mu_vector_null, np.ndarray))
        diff = self.mean_vector - mu_vector_null
        if self.p > 1:
            inv_cov = np.linalg.inv(self.covariance_matrix)
            t_2_statistic = self.n * np.matmul(np.matmul(diff, inv_cov), diff)
            critical_value = ((self.n - 1) * self.p)/(self.n-self.p) * \
                f.ppf(significance, self.p, self.n - self.p)
            f_statistic = ((self.n - self.p) * t_2_statistic) / \
                ((self.n-1) * self.p)
            p_value = 1 - f.cdf(f_statistic, self.p, self.n - self.p)
            print(f"---------------------HOTELLING'S T^2 TEST----------------------")
            print(
                f"Null Hypothesis:\n  Mean vector {self.mean_vector}\n  is equal to {np.array(mu_vector_null)}")
            print(f"Distribution: F{(self.p, self.n-self.p)}")
            print(f"F statistic: {f_statistic}")
            print(f"t^2 statistic: {t_2_statistic}")
        else:
            print(f"---------------------      F TEST        ----------------------")
            cov = self.covariance_matrix.max()
            x_bar = diff.max()
            mu = mu_vector_null[0]
            n = self.n
            print(
                f"Null Hypothesis:\n  Mean {x_bar}\n  is equal to {mu}")
            t_statistic = (x_bar - mu) / (np.sqrt(cov / n))
            f_statistic = t_statistic ** 2
            p_value = 1 - f.cdf(f_statistic, 1, self.n - 1)
            print(f"Distribution: F({(1, self.n - 1)})")
            print(f"F statistic: {f_statistic}")
        print(f"Significance: {significance*100}%")
        if method == 'p':
            print(f"P-value: {p_value}")
        elif method == 'critical':
            print(
                f"Critical Value: {critical_value}")

        if p_value < alpha:
            print(f"Conclusion: REJECT the null hypothesis")
        else:
            print(f"Conclusion: DO NOT reject the null hypothesis")
        print(f"---------------------------------------------------------------")

    def confidence_ellipsoid_info(self, alpha=0.05) -> dict:
        """Calculates the axis and the length of the ellipsoide of the multivariate data.

        Args:
            significance (float, optional): [Level of significance]. Defaults to 0.05.

        Returns:
            dict: integer keys will be the axes in the descending order. Each key has two keys("axis", "length")
                  axis denotes the direction of the ellipsoide
                  length denotes the length of the axis.
        """
        result = {}
        significance = 1-alpha
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix)
        for i, v in enumerate(eigenvalues):
            conf_half_len = np.sqrt(v) * np.sqrt((self.n - 1) * self.p * f.ppf(
                significance, self.p, self.n - self.p) / (self.n * (self.n - self.p)))
            conf_axe_abs = conf_half_len * eigenvectors[i]
            result[i] = {
                "axis": (conf_axe_abs, -conf_axe_abs),
                "length": conf_half_len * 2
            }
        return result

    def simultaneous_confidence_interval(self, vector, alpha=0.05, large_sample=False) -> tuple:
        """Calculates the simultaneous confidence interval given a transformation vector and a significance level.
             The default method would be not assuming the data as a large sample.

        Args:
           vector (list or ndarray): [The transformation vector].
           significance (float, optional): [Level of significance]. Defaults to 0.05.
           large_sample (bool, optional): [Use large sample assumptions]. Defaults to False.

        Returns:
           tuple: (lowerbound: float, upperbound: float)
        """
        significance = 1-alpha
        assert len(vector) == self.p
        if not isinstance(vector, np.ndarray):
            vec = np.array(vector)
        else:
            vec = vector
        if not large_sample:
            conf_width = np.sqrt(
                self.p * (self.n - 1) * f.ppf(significance, self.p, self.n - self.p) * vec.dot(self.covariance_matrix).dot(vec) / (self.n * (self.n - self.p)))
            t_mean = vec.dot(self.mean_vector)
            return (t_mean - conf_width, t_mean + conf_width)
        else:
            conf_width = np.sqrt(chi2.ppf(significance, self.p) *
                                 vec.dot(self.covariance_matrix).dot(vec)/self.n)
            t_mean = vec.dot(self.mean_vector)
            return (t_mean - conf_width, t_mean + conf_width)

    def profile_analysis(self, flat=True, c_matrix=None, alpha=0.05, method="p"):
        if flat:
            c_matrix = self.__flat_c_matrix()
            transformed_data = MultivariateData(
                np.matmul(c_matrix, self.data.T).T)
            transformed_data.hotellings_t_test(
                np.zeros(transformed_data.p), alpha, method)
        else:
            assert c_matrix is not None, "If not flat, c_matrix is required."
            c_mat = np.array(c_matrix)
            try:
                _, c_mat_n_col = c_mat.shape
            except Exception as e:
                if isinstance(e, ValueError) & (e.args[0] == 'not enough values to unpack (expected 2, got 1)'):
                    _, c_mat_n_col = (len(c_mat), 1)
            transformed_array = np.matmul(c_mat, self.data.T)
            # transformed_data = np.reshape(transformed_array , (len(transformed_array),c_mat_n_col))
            transformed_data = transformed_array.T
            transformed_multivar_data = MultivariateData(transformed_data)
            transformed_multivar_data.hotellings_t_test(
                [0]*len(c_mat), alpha, method)
        return

    def __flat_c_matrix(self):
        minus_identity_matrix = -np.identity(self.p)
        col_ones = np.ones((self.p, 1))
        return np.hstack((col_ones, minus_identity_matrix))[:self.p-1, :self.p]

def mean_comparison(multivardata1: MultivariateData, multivardata2: MultivariateData, no_test=False,alpha=0.05):
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
    n1 = multivardata1.n; results['n1']=n1
    n2 = multivardata2.n; results['n2']=n2
    p = multivardata1.p; results['p']=p
    mean1 = multivardata1.mean_vector
    mean2 = multivardata2.mean_vector
    mean_diff = mean1 - mean2
    cov1 = multivardata1.covariance_matrix
    cov2 = multivardata2.covariance_matrix
    s_p = ((n1-1)*cov1 + (n2-1)*cov2) / (n1 + n2 -2); results['s_p']=s_p
    t_sqrd = ((n1 * n2) / (n1 + n2)) *  np.matmul(np.matmul(mean_diff, np.linalg.inv(s_p)), mean_diff)
    const = (n1 + n2 - p - 1) / (p*(n1 + n2 -2))
    f_statistic = const * t_sqrd
    deg_free = (p, n1 + n2 -p - 1)
    c_sqrd = 1/const * f.cdf(significance, deg_free[0], deg_free[1]); results['c_sqrd']=c_sqrd
    if not no_test:
        print(f"---------------------HOTELLING'S T^2 TEST----------------------")
        print(f"Null Hypothesis:\n  Mean vector {mean1}\n  is equal to {mean2}")
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
    return f_statistic, deg_free, results


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
        _,_, params = mean_comparison(m1, m2, no_test=True, alpha=alpha)
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
