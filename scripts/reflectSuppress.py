"""
Single image reflection suppression via gradient thresholding and solving
PDE using discrete cosine transform(DCT)

Input:
Im      - the input image
h       - the gradient thresholding parameter
epsilon - the epsilon in Eq.(3) in the paper

Output:
T - the dereflected image
"""

import numpy as np
from numpy import sin, cos
from scipy.fftpack import dct, idct
import pywt
import cv2


__author__ = "Thiago Lagges"
__version__ = "1.0.0"


def reflectSuppress(Im, h, epsilon):     # move epsilon out of inputs

    Y = cv2.normalize(Im, None, alpha=0, beta=1,
                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    m, n, r = np.shape(Y)
    T = np.zeros((m, n, r))
    Y_Laplacian_2 = np.zeros((m, n, r))

    for dim in range(r):

        GRAD = grad(Y[:, :, dim])
        GRAD_x = GRAD[:, :, 0]
        GRAD_y = GRAD[:, :, 1]

        GRAD_norm = np.sqrt(np.square(GRAD_x) +
                            np.square(GRAD_y))  # element wise
        GRAD_norm_thresh = pywt.threshold(
            GRAD_norm, h, mode='hard', substitute=0)  # gradient thresholding

        ind = (GRAD_norm_thresh == 0)

        GRAD_x[ind] = 0
        GRAD_y[ind] = 0

        GRAD_thresh = np.empty((np.shape(GRAD_x)[0], np.shape(GRAD_x)[1], 2))
        GRAD_thresh[:, :, 0] = GRAD_x
        GRAD_thresh[:, :, 1] = GRAD_y
        # compute L(div(delta_h(Y)))
        Y_Laplacian_2[:, :, dim] = div(grad(div(GRAD_thresh)))

    rhs = Y_Laplacian_2 + epsilon * Y

    for dim in range(r):
        T[:, :, dim] = PoissonDCT_variant(
            rhs[:, :, dim], 1, 0, epsilon)      # solve the PDE using DCT

    return T


def PoissonDCT_variant(rhs, mu, lambda_, epsilon):
    """
    Solve the equation  (mu*L^2 - lambda_*L + epsilon)*u = rhs via DCT
    where L means Laplacian operator
    """
    M, N = rhs.shape

    k = np.arange(1, M+1)
    l = np.arange(1, N+1)
    k = k.T

    eN = np.ones((1, N))
    eM = np.ones((M, 1))

    k = cos(np.pi/M*(k-1))
    l = cos(np.pi/N*(l-1))

    k = np.kron(k, eN)
    k = k.reshape(M, N)

    l = np.kron(eM, l)
    l = l.reshape(M, N)

    kappa = 2*(k+l-2)

    const = mu * np.square(kappa) - lambda_ * kappa + epsilon

    u = dct2(rhs)
    u = u/const
    u = idct2(u)         # refer to Theorem 1 in the paper

    return u


def grad(A):
    """
    Compute the gradient of a 2D image array
    Input A - 2D image array
    """
    m, n = np.shape(A)
    B = np.zeros((m, n, 2))

    Ar = np.zeros((m, n))
    Ar[:, 0:n-1] = A[:, 1:n]
    Ar[:, n-1] = A[:, n-1]

    Au = np.zeros((m, n))
    Au[0:m-1, :] = A[1:m, :]
    Au[m-1, :] = A[m-1, :]

    B[:, :, 0] = Ar - A
    B[:, :, 1] = Au - A

    return B


def div(A):
    """
    Compute the divergence of gradient
    Input A - matrix of size m*n*2
    A[:,:,1] is the derivative along the x direction
    A[:,:,2] is the derivative along the y direction
    """
    m, n, _ = np.shape(A)
    B = np.zeros((m, n))

    T = A[:, :, 0]
    T1 = np.zeros((m, n))
    T1[:, 1:n] = T[:, 0:n-1]

    B = B + T - T1

    T = A[:, :, 1]
    T1 = np.zeros((m, n))
    T1[1:m, :] = T[0:m-1, :]

    B = B + T - T1

    return B


def dct2(X):
    """
    Discrete cosine transform
    """
    return dct(dct(X.T, norm='ortho').T, norm='ortho')


def idct2(X):
    """
    Inverse discrete cosine transform
    """
    return idct(idct(X.T, norm='ortho').T, norm='ortho')
