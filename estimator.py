import numpy as np
import pandas as pd


class covEstimator:
    def __init__(self):
        self.estimatorType = 'Base Class Estimator'


    @staticmethod
    def computeSampleCovMatrix(self, X):
        """
        sample covariance estimator
        """
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2, "input X must be 2D of shape (N, T)"
        N, T = X.shape

        X_centered = X - X.mean(axis=1, keepdims=True)
        sampleCovMatrix = (X_centered @ X_centered.T) / (T - 1)

        return sampleCovMatrix
    
    def computeFrobDis(self, covMatrix, trueCovMatrix):
        """
        This function calculate the distance between the true covariance matrix and the estimated one using Forbenius norm
        Note: In the paper, in order to compare the estimation accuracy of matrices of different shape, a normalizing constant c = 1/N is multiplied to the original Frobenius Norm
        """

        N, _ = covMatrix.shape
        frobNorm = np.linalg.norm(covMatrix - trueCovMatrix, ord='fro')
        frobDistance = frobNorm / N

        return frobDistance
    
    def computeEigenStats(self, covMatrix):
        """
        This function calculates the statistics related to the eigenvalues of the estimated covariance matrix using the given estimator, statistics include maximum eigenvalue, minimum eigenvalue, variance of the eigenvalues etc.
        """

        eigenVals = np.linalg.eigvalsh(covMatrix)
        eigenMax, eigenMin, eigenStd = np.max(eigenVals), np.min(eigenVals), np.std(eigenVals)

        return {"max": eigenMax, "min": eigenMin, "std": eigenStd}




class sampleCovEstimator(covEstimator):
    def __init__(self):
        super().__init__()
        self.estimatorType = 'Sample Covariance Estimator'

    @staticmethod
    def computeCovMatrix(X):
        """
        estimate using sample covariance
        """
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2, "input X must be 2D of shape (N, T)"
        N, T = X.shape
        X_centered = X - X.mean(axis=1, keepdims=True)
        sampleCovMatrix = (X_centered @ X_centered.T) / (T - 1)
        return sampleCovMatrix



class shrinkageCovEstimator(covEstimator):
    def __init__(self):
        super().__init__()
        self.estimatorType = 'Shrinkage Covariance Estimator'

    @staticmethod
    def computeCovMatrix(X):
        """
        linear shrinkage method for calculating the covariance matrix
        """
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2, "input X must be 2D of shape (N, T)"
        N, T = X.shape

        X_centered = X - X.mean(axis=1, keepdims=True)
        sampleCovMatrix = (X_centered @ X_centered.T) / (T - 1)

        mHat = np.trace(sampleCovMatrix) / N
        I = np.eye(N)

        dHatSquare = np.linalg.norm(sampleCovMatrix - mHat * I, ord='fro') ** 2

        if dHatSquare <= 1e-16:
            return sampleCovMatrix

        r2HatSquare = 0.0
        for t in range(T):
            xt = X_centered[:, t:t+1]
            outerProd = xt @ xt.T
            matDiff = outerProd - sampleCovMatrix
            r2HatSquare += np.linalg.norm(matDiff, ord='fro') ** 2
        r2HatSquare /= (T ** 2)

        r1HatSquare = dHatSquare - r2HatSquare

        weightSampleCov = r1HatSquare / dHatSquare
        weightIdentityMat = r2HatSquare / dHatSquare

        covMatrix = weightSampleCov * sampleCovMatrix + weightIdentityMat * (mHat * I)
        return covMatrix