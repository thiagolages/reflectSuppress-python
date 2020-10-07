# single image reflection suppression via gradient thresholding and solving
# PDE using discrete cosine transform(DCT)

# Input:
# Im      - the input image
# h       - the gradient thresholding parameter
# epsilon - the epsilon in Eq.(3) in the paper

# Output:
# T - the dereflected image


# Sample run:
# python py_test.py 

import numpy as np
from numpy import sin, cos
from scipy.fftpack import dct, idct # discrete cosine transform and inverse transform
import pywt                         # Python Wavelets package
import cv2

debug = False

def reflectSuppress(Im, h, epsilon):     # move epsilon out of inputs

    Y = cv2.normalize(Im, None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    m, n, r = np.shape(Y)
    T = np.zeros((m,n,r))
    Y_Laplacian_2 = np.zeros((m,n,r))
    
    for dim in range(r):
        GRAD   = grad(Y[:,:,dim])
        GRAD_x = GRAD[:,:,0]
        
        if debug:
            print(GRAD_x[0:5, 0:5])
            print(GRAD_x.shape)
            input("GRAD_x Press Enter to continue...")
        
        GRAD_y = GRAD[:,:,1]
        if debug:
            print(GRAD_y[0:5, 0:5])
            print(GRAD_y.shape)
            input("GRAD_y Press Enter to continue...")

        GRAD_norm = np.sqrt(np.square(GRAD_x) + np.square(GRAD_y)) # element wise 
        if debug:
            print(GRAD_norm[0:5, 0:5])
            print(GRAD_norm.shape)
            input("GRAD_norm Press Enter to continue...")

        GRAD_norm_thresh = pywt.threshold(GRAD_norm, h, mode='hard', substitute=0)# gradient thresholding
        if debug:
            print(GRAD_norm_thresh[0:5, 0:5])
            print(GRAD_norm_thresh.shape)
            input("GRAD_norm_thresh Press Enter to continue...")

        ind = (GRAD_norm_thresh == 0)
        if debug:
            print(ind[0:5, 0:5])
            input("ind Press Enter to continue...")

        GRAD_x[ind] = 0
        if debug:
            print(GRAD_x[0:5, 0:5])
            print(GRAD_x.shape)
            input("GRAD_x Press Enter to continue...")
        GRAD_y[ind] = 0
        
        if debug:
            print(GRAD_y[0:5, 0:5])
            print(GRAD_y.shape)
            input("GRAD_y Press Enter to continue...")
        
        GRAD_thresh = np.empty((np.shape(GRAD_x)[0], np.shape(GRAD_x)[1], 2))
        GRAD_thresh[:,:,0] = GRAD_x
        GRAD_thresh[:,:,1] = GRAD_y                                       
        Y_Laplacian_2[:,:,dim] = div(grad(div( GRAD_thresh )))             # compute L(div(delta_h(Y)))

        # print(Y_Laplacian_2[0:5, 0:5, 0])
        # print(Y_Laplacian_2.shape)
        # input("Y_Laplacian_2[0:5, 0:5, 0] Press Enter to continue...")
        
    rhs = Y_Laplacian_2 + epsilon * Y     
        
    for dim in range(r):
        T[:,:,dim] = PoissonDCT_variant(rhs[:,:,dim], 1, 0, epsilon)      # solve the PDE using DCT 
    
    # print(T[0:5, 0:5, 0])
    # print(T.shape)
    # input("T[0:5, 0:5, 0] Press Enter to continue...")

    return T




# solve the equation  (mu*L^2 - lambda_*L + epsilon)*u = rhs via DCT
# where L means Laplacian operator 
def PoissonDCT_variant(rhs, mu, lambda_, epsilon):

    # print(rhs[0:5, 0:5])
    # print(rhs.shape)
    # input("rhs[0:5, 0:5] Press Enter to continue...")

    M,N = rhs.shape
    k = np.arange(1, M+1)
    l = np.arange(1, N+1)
    k = k.T
    eN = np.ones((1,N))
    eM = np.ones((M,1))
    k = cos(np.pi/M*(k-1))
    # print(k[0:5])
    # print(k.shape)
    # input("k[0:5] Press Enter to continue...")
    
    l = cos(np.pi/N*(l-1))
    # print(l[0:5])
    # print(l.shape)
    # input("l[0:5] Press Enter to continue...")

    k = np.kron(k,eN)
    k = k.reshape(M,N)
    l = np.kron(eM,l)
    l = l.reshape(M,N)

    # print(k[0:5,0:5])
    # print(k.shape)
    # input("k-kron[0:5,0:5] Press Enter to continue...")

    # print(l[0:5,0:5])
    # print(l.shape)
    # input("l-kron[0:5,0:5] Press Enter to continue...")

    kappa = 2*(k+l-2)

    # print(kappa[0:5,0:5])
    # print(kappa.shape)
    # input("kappa[0:5,0:5] Press Enter to continue...")

    const = mu * np.square(kappa) - lambda_ * kappa + epsilon
    # print(const[0:5,0:5])
    # print(const.shape)
    # input("const[0:5,0:5] Press Enter to continue...")
    
    # print(rhs[0:5,0:5])
    # print(rhs.shape)
    # input("rhs[0:5,0:5] Press Enter to continue...")
    
    u = dct2(rhs)
    # print(u[0:5,0:5])
    # print(u.shape)
    # input("u[0:5,0:5] Press Enter to continue...")
    # np.save('u.npy', u)
    # np.save('rhs.npy', rhs)
    # a = np.sum(u, axis=0)
    # print(a[0:10])
    # print(np.sum(u, axis=0).shape)
    # input("sum(u,0)[0:10] Press Enter to continue...")
    
    u = u/const
    # print(u[0:5,0:5])
    # print(u.shape)
    # input("u[0:5,0:5] Press Enter to continue...")
    u = idct2(u)                       # refer to Theorem 1 in the paper
    # print(u[0:5,0:5])
    # print(u.shape)
    # input("u[0:5,0:5] Press Enter to continue...")

    return u



# compute the gradient of a 2D image array

def grad(A):
    global debug

    m, n = np.shape(A)
    B = np.zeros((m,n,2))
    if debug:
        print(A[0:7,0:6,0])
        print(A.shape)
        input("A Press Enter to continue...")

    Ar = np.zeros((m,n))
    Ar[:,0:n-1] = A[:,1:n]
    Ar[:,n-1] = A[:,n-1]
    if debug:
        print(Ar[0:7,0:6])
        print(Ar.shape)
        input("Ar Press Enter to continue...")

    Au = np.zeros((m,n))
    Au[0:m-1,:] = A[1:m,:]
    Au[m-1,:] = A[m-1,:]
    if debug:
        print(Au[0:7,0:6])
        print(Au.shape)
        input("Au Press Enter to continue...")
    
    B[:,:,0] = Ar - A     
    B[:,:,1] = Au - A     
    if debug:
        print(B[0:7,0:6,0])
        print(B.shape)
        input("B Press Enter to continue...")

    return B


# compute the divergence of gradient
# Input A is a matrix of size m*n*2
# A[:,:,1] is the derivative along the x direction
# A[:,:,2] is the derivative along the y direction

def div(A):
    global debug

    m, n, _ = np.shape(A)
    B = np.zeros((m,n))
    if debug:
        print(A[0:5, 0:5])
        print(A.shape)
        input("A (div) Press Enter to continue...")

    T = A[:,:,0]
    T1 = np.zeros((m,n))
    T1[:,1:n] = T[:,0:n-1]
    if debug:
        print(T1[0:5, 0:5])
        print(T1.shape)
        input("T1 (div) Press Enter to continue...")

    B = B + T - T1

    T = A[:,:,1]
    T1 = np.zeros((m,n))
    T1[1:m,:] = T[0:m-1,:]

    B = B + T - T1
    if debug:
        print(B[0:5, 0:5])
        print(B.shape)
        input("B (div) Press Enter to continue...")

    return B


# def dct2(X):
#     return dct(dct(X.T, norm='ortho').T, norm='ortho')
def dct2(y): #2D DCT bulid from numpy and using prvious DCT function
    
    # print(y[0:5, 0:5])
    # print(y.shape)
    # input("rhs (div) Press Enter to continue...")
    M = y.shape[0]
    N = y.shape[1]
    a = np.empty([M,N],np.float64)
    b = np.empty([M,N],np.float64)
    for i in range(M):
        a[i,:] = dct(y[i,:], norm='ortho')
    for j in range(N):
        b[:,j] = dct(a[:,j], norm='ortho')
    return b

def idct2(X):
    return idct(idct(X.T, norm='ortho').T, norm='ortho')