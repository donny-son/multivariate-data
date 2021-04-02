import numpy as np
import pandas as pd
from scipy.stats import chi2, f
import matplotlib.pyplot as plt
try:
    import termplotlib as tpl
except Exception as e:
    print(f"termploblib is not installed.\nUsing matplotlib as default.")


class MultivariateData:
    """Object for processing multivariate data. 

    params
    ------
    src: input matrix of shape (n:int, p:int) where n is the sample size and p is the number of dependent variables
        np.ndarray

    methods
    -------
    _get_mean_vector: gets the mean vector along the features
        np.array

    _get_cov_mat: gets the covariance matrix in (p by p).
        np.array

    _generalized_squared_distance: gets list of squared distance by dimension n.
        list

    _get_qq_tuples: gets the list of tuples of the qq pair for the chisquare distribution with df=p
        list

    draw_qqplot: draws the plot by using matplotlib
    """

    def __init__(self, src) -> None:
        self.src = self._numpy_coersion(src)
        self.n, self.p = self.src.shape
        self.mean_vector = self._get_mean_vector()
        self.cov_matrix = self._get_cov_mat()

    @staticmethod
    def _numpy_coersion(data) -> np.array:
        # coerce pandas or other iterative data to numpy array.
        if not isinstance(data, np.ndarray):
            try:
                result = np.array(data)
            except Exception as e:
                print(
                    f"{data.__class__} cannot be coerced to Numpy array!\nERROR:{e}")
            return result
        else:
            return data

    def _get_mean_vector(self) -> np.array:
        return (np.mean(self.src, axis=0))

    def _get_cov_mat(self) -> np.array:
        result = np.cov(self.src.transpose())
        assert result.shape == (self.p, self.p)
        return result

    def _generalized_squared_distance(self) -> list:
        result = []
        inv_cov = np.linalg.inv(self.cov_matrix)
        for row in self.src:
            diff = row - self.mean_vector
            # numpy broadcasting
            result.append(np.matmul(np.matmul(diff, inv_cov), diff))
        assert len(result) == self.n
        return result

    def _get_qq_tuples(self) -> list:
        result = []
        sorted_general_distance = sorted(self._generalized_squared_distance())
        for i, x in enumerate(sorted_general_distance):
            x_probability_value = (i+1 - 0.5) / self.n
            q_value = chi2.ppf(x_probability_value, self.p)
            result.append(
                (q_value, x)
            )
        return result

    def draw_qqplot(self, terminal=False):
        qq_tuples = self._get_qq_tuples()
        x = [x for x, _ in qq_tuples]
        y = [y for _, y in qq_tuples]
        if terminal:
            fig = tpl.figure()
            fig.plot(x, y, width=60, height=20)
            fig.show()
        else:
            plt.scatter(x, y)
            plt.show()

    def hotellings_t_test(self, mu_vector_null, significance=0.05):
        assert (isinstance(mu_vector_null, list)
                or isinstance(mu_vector_null, np.ndarray))
        assert (significance < 1) & (significance > 0)
        inv_cov = np.linalg.inv(self.cov_matrix)
        diff = self.mean_vector - mu_vector_null
        statistic = self.n * np.matmul(np.matmul(diff, inv_cov), diff)
        critical_value = ((self.n - 1) * self.p)/(self.n-self.p) * \
            f.ppf(significance, self.p, self.n - self.p)
        print(f"---------------------HOTELLING'S T^2 TEST----------------------")
        print(
            f"Null Hypothesis:\n  Mean vector {self.mean_vector}\n  is equal to {mu_vector_null}")
        print(f"Test statistic: {statistic}")
        print(
            f"Critical Value with {(1-significance)*100}% significance: {critical_value}")
        if statistic > critical_value:
            print(f"Conclusion: REJECT the null hypothesis")
        else:
            print(f"Conclusion: DO NOT reject the null hypothesis")
        print(f"---------------------------------------------------------------")
        return statistic


if __name__ == "__main__":
    college_dat = pd.read_csv(
        "college.DAT", delim_whitespace=True, header=None)
    college_dat.columns = ["ssh", "vrbl", "sci"]
    cd = MultivariateData(college_dat)
    cd.hotellings_t_test([500, 50, 30])
