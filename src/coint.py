import numpy as np
from statsmodels.tsa.tsatools import lagmat

import models.critical_values as cv
class Johansen(object):
    def __init__(self, x, model=1, k=1, trace=False,  significance_level=1):
        self.x = x
        self.k = k
        #self.idim=[0,1,2,3]
        self.trace = trace
        self.model = model
        self.significance_level = significance_level
        if trace:
            key = "TRACE_{}".format(model)
        else:
            key = "MAX_EVAL_{}".format(model)

        critical_values_str = cv.mapping[key]

        select_critical_values = np.array(
            critical_values_str.split(),
            float).reshape(-1, 3)
        #print(select_critical_values)
        self.critical_values = select_critical_values[:, significance_level]
    def mle(self):
        nx=self.x.cpu().numpy()
        #print(type(self.x))
        #print(np.diff(nx, axis=0))
        x_diff = np.diff(nx, axis=0)
        x_diff_lags = lagmat(x_diff, self.k, trim='both')
        x_lag = lagmat(nx, 1, trim='both')
        
        x_diff = x_diff[self.k:]

        x_lag = x_lag[self.k:]
         # Include intercept in the regressions if self.model != 0.
        if self.model != 0:
            ones = np.ones((x_diff_lags.shape[0], 1))
            x_diff_lags = np.append(x_diff_lags, ones, axis=1)
            #print('hhhh')
        # Include time trend in the regression if self.model = 3 or 4.
        if self.model in (3, 4):
            times = np.asarray(range(x_diff_lags.shape[0])).reshape((-1, 1))
            x_diff_lags = np.append(x_diff_lags, times, axis=1)

        # Residuals of the regressions of x_diff and x_lag on x_diff_lags.
     #   print('hhhh')
        try:
            inverse = np.linalg.pinv(x_diff_lags)
        except:
         #   print("Unable to take inverse of x_diff_lags.")
            return None

        u = x_diff - np.dot(x_diff_lags, np.dot(inverse, x_diff))
        v = x_lag - np.dot(x_diff_lags, np.dot(inverse, x_lag))
        # Covariance matrices of the residuals.
        t = x_diff_lags.shape[0]
        Svv = np.dot(v.T, v) / t
        Suu = np.dot(u.T, u) / t
        Suv = np.dot(u.T, v) / t
        Svu = Suv.T

        try:
            Svv_inv = np.linalg.inv(Svv)
        except:
          #  print("Unable to take inverse of Svv.")
            return None
        try:
            Suu_inv = np.linalg.inv(Suu)
        except:
           # print("Unable to take inverse of Suu.")
            return None

        # Eigenvalues and eigenvectors of the product of covariances.
        cov_prod = np.dot(Svv_inv, np.dot(Svu, np.dot(Suu_inv, Suv)))
        eigenvalues, eigenvectors = np.linalg.eig(cov_prod)

        # Normalize the eigenvectors using Cholesky decomposition.
        evec_Svv_evec = np.dot(eigenvectors.T, np.dot(Svv, eigenvectors))
        cholesky_factor = np.linalg.cholesky(evec_Svv_evec)
        try:
            eigenvectors = np.dot(eigenvectors,
                                  np.linalg.inv(cholesky_factor.T))
        except:
          #  print("Unable to take the inverse of the Cholesky factor.")
            return None

        # Ordering the eigenvalues and eigenvectors from largest to smallest.
        indices_ordered = np.argsort(eigenvalues)
       # print(indices_ordered)
        indices_ordered = np.flipud(indices_ordered)
        eigenvalues = eigenvalues[indices_ordered]
        eigenvectors = eigenvectors[:, indices_ordered]

        return eigenvectors, eigenvalues,x_diff
    def h_test(self, eigenvalues, r):
        
        nobs, m = self.x.shape
        #print('nob:',nobs,'m:',m)
        t = nobs - self.k - 1

        if self.trace:
            m = len(eigenvalues)
            statistic = -t * np.sum(np.log(np.ones(m) - eigenvalues)[r:])
        else:
           # print('test')
            statistic = -t * np.sum(np.log(1 - eigenvalues[r]))
           # print(statistic)

        critical_value = self.critical_values[m - r - 1]

        if statistic > critical_value:
     #       print(True)
            return True

        else:
            return False
    def johansen(self):
        """Obtain the possible cointegrating relations and numbers of them.

        See the documentation for methods mle and h_test.

        :return: The possible cointegrating relations, i.e. the eigenvectors
        obtained from maximum likelihood estimation, and the numbers of
        cointegrating relations for which the null hypothesis is rejected.

        """

        nobs, m = self.x.shape

        try:
            eigenvectors, eigenvalues,x_diff = self.mle()
        #print(eigenvectors,eigenvalues)
        except:
      #      print("Unable to obtain possible cointegrating relations.")
            return None

        rejected_r_values = []
        for r in range(m):
            if self.h_test(eigenvalues, r):
                rejected_r_values.append(r)
        
        return True

