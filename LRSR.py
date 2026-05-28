# coding=utf-8

import numpy as np
from scipy import linalg


def LRSR(DictLRR, DictSRC, data, beta, lmda):
    """
    A9: Augmented Lagrangian Method (ALM) / ADMM — optimizes low-rank Z, sparse S, and noise E.
    Low-Rank and Sparse Representation (LRSR) optimization module.
    
    Methodology References:
    - Section 3.1: Core decomposition model X = D_bg Z + D_an S + E (Equation 1)
    - Section 3.7: Low-Rank and Sparse Decomposition optimization problem
    - Section 3.8: Optimization Using Augmented Lagrangian Method (ALM)
    
    :param DictLRR: the background dictionary D_bg ∈ R^(B×K_bg)
    :param DictSRC: the anomaly dictionary D_an ∈ R^(B×K_an)
    :param data: the normalized data X ∈ R^(B×N)
    :param beta: sparsity regularization parameter
    :param lmda: noise regularization parameter
    :return: Z: the low rank coefficients (background)
             S: the sparse coefficients (anomaly)
             E: the noise or modeling error
    """

    # Section 3.1: Initialize variables for decomposition model
    # X = D_bg Z + D_an S + E (Equation 1)
    dataRows, dataCols = data.shape
    DLRows, DLCols = DictLRR.shape
    DSRows, DSCols = DictSRC.shape
    ILRR = np.eye(DLCols)
    ISRC = np.eye(DSCols)
    Z = np.zeros((DLCols, dataCols))  # Low-rank background coefficients
    J = np.zeros((DLCols, dataCols))  # Auxiliary variable for low-rank
    E = np.zeros((dataRows, dataCols))  # Noise/error term
    S = np.zeros((DSCols, dataCols))  # Sparse anomaly coefficients
    L = np.zeros((DSCols, dataCols))  # Auxiliary variable for sparse
    # A9: Augmented Lagrangian multipliers Y1, Y2, Y3 and penalty parameter mu
    # Section 3.8: Augmented Lagrangian multipliers
    Y1 = np.zeros((dataRows, dataCols))  # Multiplier for constraint
    Y2 = np.zeros((DLCols, dataCols))  # Multiplier for Z = J
    Y3 = np.zeros((DSCols, dataCols))  # Multiplier for S = L
    mu = 0.0001  # Augmented Lagrangian parameter
    mu_max = 10 ** 10
    p = 1.1  # Parameter update factor
    err = 0.000001  # Convergence threshold (Equation 17)
    err_absolute = 1e-10  # Absolute convergence threshold (effectively zero)
    err_relative = 1e-6   # Relative improvement threshold
    itera = 1
    prev_rlerr = np.inf
    inv_Z = np.linalg.inv(np.dot(DictLRR.transpose(), DictLRR) + ILRR)
    inv_S = np.linalg.inv(np.dot(DictSRC.transpose(), DictSRC) + ISRC)

    # Section 3.7: Optimization problem (Equation 11-12)
    # min_(Z,S,E) ||Z||_* + β||S||_1 + λ||E||_2,1
    # subject to: X = D_bg Z + D_an S + E
    # A9: ALM iteration loop — alternate Z, S, E updates until convergence (Equation 17)
    while itera < 500:
        print("iteration:{0}".format(itera))
        # Section 3.8, Step 1: Low-rank update using Singular Value Thresholding
        # Equation (13): Z^(k+1) = SVT_(1/μ)(Z^k + Y_1/μ)
        operator1 = 1 / mu
        tmpJ = Z + Y2 / mu
        Ju, Jsigma, Jvt = linalg.svd(tmpJ, full_matrices=False)
        # threshold1 =1/mu
        evp = Jsigma[Jsigma > operator1].shape[0]
        if evp >= 1:
            Jsigma[0:evp] -= operator1
            JsigmaM = np.diag(Jsigma[0:evp])
            print ("current evp is: {0}".format(evp))
        else:
            evp = 1
            JsigmaM = 0

        J = np.dot(np.dot(Ju[:, 0:evp], JsigmaM), Jvt[0:evp, :])
        # Section 3.8, Step 3: Noise update using soft thresholding
        # Equation (15): E^(k+1) = T_(λ/μ)(E^k + Y_3/μ)
        # ||E||_2,1 mixed norm modeling spectral noise
        operator3 = lmda / mu
        tmpE = data - np.dot(DictLRR, Z) - np.dot(DictSRC, S) + Y1 / mu
        terows, tecols = tmpE.shape
        for i in range(tecols):
            tmpValue1 = linalg.norm(tmpE[:, i])
            if tmpValue1 > operator3:
                E[:, i] = ((tmpValue1 - operator3) / tmpValue1) * tmpE[:, i]
            else:
                E[:, i] = 0
        # Section 3.8, Step 2: Sparse anomaly update using soft thresholding
        # Equation (14): S^(k+1) = S_(β/μ)(S^k + Y_2/μ)
        # ||S||_1 sparsity promoting anomaly isolation
        tmpL = S + Y3 / mu
        operator2 = beta / mu
        tmpL[tmpL > operator2] -= operator2
        tmpL[tmpL < -operator2] += operator2
        tmpL[(tmpL >= -operator2) & (tmpL <= operator2)] = 0
        L = tmpL.copy()
        # Update Z using ADMM
        tmpZ = np.dot(DictLRR.transpose(), data - np.dot(DictSRC, S) - E) + J + \
            (np.dot(DictLRR.transpose(), Y1) - Y2) / mu
        Z = np.dot(inv_Z, tmpZ)
        # Update S using ADMM
        tmpS = np.dot(DictSRC.transpose(), data - np.dot(DictLRR, Z) - E) + L + \
            (np.dot(DictSRC.transpose(), Y1) - Y3) / mu
        S = np.dot(inv_S, tmpS)
        # Section 3.8, Step 4: Multiplier updates
        # Equation (16): Y^(k+1) = Y^k + μ(X - D_bg Z^(k+1) - D_an S^(k+1) - E^(k+1))
        T1 = data - np.dot(DictLRR, Z) - E - np.dot(DictSRC, S)
        T2 = Z - J
        T3 = S - L
        Y1 += mu * T1
        Y2 += mu * T2
        Y3 += mu * T3
        # Update penalty parameter and check convergence
        # Equation (17): ||X - D_bg Z - D_an S - E||_∞ < ε
        err1 = linalg.norm(T1, np.inf)
        err2 = linalg.norm(T2, np.inf)
        err3 = linalg.norm(T3, np.inf)
        rlerr = max(err1, err2, err3)
        mu = min(p * mu, mu_max)

        itera += 1
        print("max err is:{0}".format(rlerr))
        print("current mu is:{0}".format(mu))
        
        # Section 3.8: Convergence check
        # Equation (17): Convergence determined by ||X - D_bg Z - D_an S - E||_∞ < ε
        # Enhanced convergence checks
        # 1. Original convergence check (relative error < threshold)
        if rlerr < err:
            print("Converged: max error < {0}".format(err))
            break
        
        # 2. Absolute convergence (error effectively zero)
        if rlerr < err_absolute:
            print("Converged: max error effectively zero (< {0})".format(err_absolute))
            break
        
        # 3. Relative improvement check (error not improving significantly)
        if itera > 10:  # Wait a few iterations before checking relative improvement
            relative_improvement = abs(prev_rlerr - rlerr) / (prev_rlerr + 1e-12)
            if relative_improvement < err_relative and rlerr < err * 10:
                print("Converged: relative improvement < {0} and error < {1}".format(err_relative, err * 10))
                break
        
        prev_rlerr = rlerr
    return Z, E, S
