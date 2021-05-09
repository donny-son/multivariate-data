import numpy as np
from functools import reduce

from numpy.core.fromnumeric import diagonal
from Multivar import MultivariateData
from scipy.stats import f, chi2
from pandas import DataFrame


class MultipleRegression:
    """
    Object for conducting Multiple Regression

    params
    ------
    y : response matrix
        MultivariateData or np.array
    X : predictor matrix
        MultivariateData or np.array
    """

    def __init__(self, y, X):
        if isinstance(y, MultivariateData):
            self.y = y.data
            self.n = y.n
            self.m = y.p
        if isinstance(X, MultivariateData):
            col_ones = np.ones((self.n, 1))
            self.Z = np.hstack((col_ones, X.data))
            self.r = self.Z.shape[1] - 1
        assert isinstance(self.y, np.ndarray)
        assert isinstance(self.Z, np.ndarray)
        self.Zt = np.transpose(self.Z)
        self.invZtZ = np.linalg.inv(np.matmul(self.Zt, self.Z))
        self.hat_matrix = reduce(np.matmul, (self.Z, self.invZtZ, self.Zt))
        self.beta_hat = reduce(np.matmul, (self.invZtZ, self.Zt, self.y))
        self.y_hat = np.apply_along_axis(
            lambda x: np.matmul(self.hat_matrix, x), 0, self.y)
        self.residuals = np.subtract(self.y, self.y_hat)
        self.cov_matrix = np.matmul(
            np.transpose(self.residuals), self.residuals)
        self.sigma_hat = self.cov_matrix / self.n

    def get_mean_confidence_interval(self, sample_data, alpha=0.05):
        """Constructs 1-alpha level c.i. given sample data

        Args:
            sample_data (np.array): fixed sample data
            alpha (float, optional): alpha confidence level. 
                Defaults to 0.05.

        Return:
            pandas.DataFrame where rows are corresponding response variables 
            and two columns are upperbound and lowerbound of confidence interval
        """
        result = {}
        sample_data = np.hstack((1, sample_data))
        y_mean = np.matmul(np.transpose(sample_data), self.beta_hat)
        _const = (self.m * (self.n - self.r - 1)) / (self.n - self.r - self.m)
        _f_val = f.ppf(alpha, self.m, self.n - self.r - self.m)
        _crit_part = np.sqrt(_const * _f_val)
        _se_left = reduce(np.matmul, (np.transpose(
            sample_data), self.invZtZ, sample_data))
        _se_right = np.array([self.n / (self.n - self.r - 1) *
                              se for se in np.diagonal(self.cov_matrix)])
        _se_part = np.sqrt(_se_left * _se_right)
        result['upper'] = y_mean + _crit_part * _se_part
        result['lower'] = y_mean - _crit_part * _se_part
        return DataFrame(result)

    def get_predictor_confidence_interval(self, sample_data, alpha=0.05):
        """Constructs alpha-level c.i. given sample data

        Args:
            sample_data (np.array): fixed sample data
            alpha (float, optional): alpha confidence level. 
                Defaults to 0.05.
        """
        result = {}
        sample_data = np.hstack((1, sample_data))
        y_mean = np.matmul(np.transpose(sample_data), self.beta_hat)
        _const = (self.m * (self.n - self.r - 1)) / (self.n - self.r - self.m)
        _f_val = f.ppf(alpha, self.m, self.n - self.r - self.m)
        _crit_part = np.sqrt(_const * _f_val)
        _se_left = 1 + reduce(np.matmul, (np.transpose(
            sample_data), self.invZtZ, sample_data))
        _se_right = np.array([self.n / (self.n - self.r - 1) *
                              se for se in np.diagonal(self.cov_matrix)])
        _se_part = np.sqrt(_se_left * _se_right)
        result['upper'] = y_mean + _crit_part * _se_part
        result['lower'] = y_mean - _crit_part * _se_part
        return DataFrame(result)

    def test(self, zero_beta_indices: list, alpha=0.05):
        beta_hat_reduced = np.copy(self.beta_hat)
        q = len(zero_beta_indices) - 1
        for i in zero_beta_indices:
            beta_hat_reduced[i] = 0
        y_hat_reduced = np.matmul(self.Z, beta_hat_reduced)
        residuals_reduced = np.subtract(self.y, y_hat_reduced)
        cov_matrix_reduced = np.matmul(
            np.transpose(residuals_reduced), residuals_reduced)
        cov_diff = np.subtract(cov_matrix_reduced, self.cov_matrix)
        inv_cov = np.linalg.inv(self.cov_matrix)
        eigenvals, _ = np.linalg.eig(np.matmul(cov_diff, inv_cov))
        wilks_lambda = reduce(np.multiply, map(lambda x: 1/(1+x), eigenvals))
        stats = -(self.n - self.r - 1 - 0.5 *
                  (self.m - self.r + q + 1)) * np.log(wilks_lambda)
        df = self.m * (self.r - q)
        pval = 1 - chi2.cdf(stats, df)

        print(f"----------------------𝑾𝒊𝒍𝒌𝒔 𝑳𝒂𝒎𝒃𝒅𝒂 𝒕𝒆𝒔𝒕-------------------------")
        print(
            f"Null Hypothesis:\n {zero_beta_indices}th Betas = 0\n where 0th Beta:Intercept")
        print(f"Wilks Lambda : {wilks_lambda}")
        print(f"Chi2 Statistic : {stats}")
        print(f"Distribution : Chi2 with df={df}")
        print(f"P values : {pval}")
        print(f"Significance: {(1-alpha)*100}%")
        if pval < alpha:
            print(f"Conclusion: REJECT the null hypothesis")
        else:
            print(f"Conclusion: DO NOT reject the null hypothesis")
        print(f"----------------------------------------------------------------")
        return


if __name__ == "__main__":
    import pandas as pd
    amit_df = pd.read_csv(
        'amit.DAT',
        header=None,
        index_col=False,
        delim_whitespace=True)
    amit_df.columns = ['y1', 'y2', 'z1', 'z2', 'z3', 'z4', 'z5']
    amit_y = MultivariateData(amit_df.iloc[:, 0:2])
    amit_x = MultivariateData(amit_df.iloc[:, 2:])
    mReg = MultipleRegression(amit_y, amit_x)
    # MultivariateData(mReg.residuals).qqplot(terminal=True)
    sample_z = np.array([1, 1200, 140, 70, 85])

    e_y1 = mReg.get_mean_confidence_interval(sample_z).iloc[0, :]
    e_y1

    y2_pred_interval = mReg.get_predictor_confidence_interval(
        sample_z).iloc[1, :]
    y2_pred_interval

    mReg.sigma_hat

    zero_beta_ids = [3, 4, 5]
    mReg.test(zero_beta_ids)
    pass
