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

    def __repr__(self) -> str:
        return f"MultivariateData(SampleSize:{self.n}, Features:{self.p})"

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

    def hotellings_t_test(self, mu_vector_null, significance=0.05, method="p"):
        """Performs Hotellings test for mean comparison, via adjusted F distribution

        Args:
            mu_vector_null ([int, float]): vector of mean under the null hypothesis
            significance (float, optional): Significance level. Defaults to 0.05.
            method (str, optional): Method of testing. Either 'p' or 'critical'. Defaults to "p".
        """
        assert (isinstance(mu_vector_null, list)
                or isinstance(mu_vector_null, np.ndarray))
        assert (0 < significance < 1)
        inv_cov = np.linalg.inv(self.covariance_matrix)
        diff = self.mean_vector - mu_vector_null
        t_2_statistic = self.n * np.matmul(np.matmul(diff, inv_cov), diff)
        critical_value = ((self.n - 1) * self.p)/(self.n-self.p) * \
            f.ppf(significance, self.p, self.n - self.p)
        f_statistic = ((self.n - self.p) * t_2_statistic) / \
            ((self.n-1) * self.p)
        p_value = 1-f.cdf(f_statistic, self.p, self.n - self.p)
        print(f"---------------------HOTELLING'S T^2 TEST----------------------")
        print(
            f"Null Hypothesis:\n  Mean vector {self.mean_vector}\n  is equal to {np.array(mu_vector_null)}")
        print(f"T^2 statistic: {t_2_statistic}")
        print(f"F statistic: {f_statistic}")

        print(f"Significance(alpha): {significance}")

        if method == 'p':
            print(f"P-value: {p_value}")
        elif method == 'critical':
            print(
                f"Critical Value: {critical_value}")

        if p_value < significance:
            print(f"Conclusion: REJECT the null hypothesis")
        else:
            print(f"Conclusion: DO NOT reject the null hypothesis")
        print(f"---------------------------------------------------------------")

    def confidence_ellipsoid_info(self, significance=0.05) -> dict:
        """Calculates the axis and the length of the ellipsoide of the multivariate data.

        Args:
            significance (float, optional): [Level of significance]. Defaults to 0.05.

        Returns:
            dict: integer keys will be the axes in the descending order. Each key has two keys("axis", "length")
                  axis denotes the direction of the ellipsoide
                  length denotes the length of the axis.
        """
        result = {}
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

    def simultaneous_confidence_interval(self, vector, significance=0.05, large_sample=False) -> tuple:
        """Calculates the simultaneous confidence interval given a transformation vector and a significance level.
             The default method would be not assuming the data as a large sample.

        Args:
           vector (list or ndarray): [The transformation vector].
           significance (float, optional): [Level of significance]. Defaults to 0.05.
           large_sample (bool, optional): [Use large sample assumptions]. Defaults to False.

        Returns:
           tuple: (lowerbound: float, upperbound: float)
        """
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


if __name__ == "__main__":
    # import data
    college_dat = pd.read_csv(
        "college.DAT", delim_whitespace=True, header=None)
    college_dat.columns = ["ssh", "vrbl", "sci"]

    # initialize class
    cd = MultivariateData(college_dat)

    # perform test
    cd.hotellings_t_test([500, 50, 30], method='p')
    pprint.pprint(cd.confidence_ellipsoid_info())
    print(cd.simultaneous_confidence_interval([1, -2, 1]))

    # import data
    stiff = pd.read_csv('stiff.DAT',
                        header=None, delim_whitespace=True)
    stiff.columns = ['x1', 'x2', 'x3', 'x4', 'x5']

    stf = MultivariateData(stiff)
    stf.qqplot(True)
    print(stf.simultaneous_confidence_interval([1, 2, -1, -2, 0]))
    print(stf.simultaneous_confidence_interval(
        [1, 2, -1, -2, 0], large_sample=True))
