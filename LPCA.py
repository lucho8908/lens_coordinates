from __future__ import division

import time
import numpy as np
import scipy as sp
import math
import cmath
import itertools
import sys

from ripser import ripser
from persim import plot_diagrams

np.random.seed( 10 )

# Its actually maxmin subsampling. l_next = argmax_X(min_L(d(x,l)))
def minmax_subsample_distance_matrix(X, num_landmarks, seed=[]):
    '''
    This function computes minmax subsampling using a square distance matrix.

    :type X: numpy array
    :param X: Square distance matrix

    :type num_landmarks: int
    :param num_landmarks: Number of landmarks

    :type seed: list
    :param list: Default []. List of indices to seed the sampling algorith.
    '''
    num_points = len(X)

    if not(seed):
        ind_L = [np.random.randint(0,num_points)] 
    else:
        ind_L = seed
        num_landmarks += 1

    distance_to_L = np.min(X[ind_L, :], axis=0)

    for i in range(num_landmarks-1):
        ind_max = np.argmax(distance_to_L)
        ind_L.append(ind_max)

        dist_temp = X[ind_max, :]

        distance_to_L = np.minimum(distance_to_L, dist_temp)
            
    return {'indices':ind_L, 'distance_to_L':distance_to_L}

def minmax_subsample_point_cloud(X, num_landmarks, distance):
    '''
    This function computes minmax subsampling using point cloud and a distance function.

    :type X: numpy array
    :param X: Point cloud. If X is a nxm matrix, then we are working with a pointcloud with n points and m variables.

    :type num_landmarks: int
    :param num_landmarks: Number of landmarks

    :type distance: function
    :param  distance: Distance function. Must be able to compute distance between 2 point cloud with same dimension and different number of points in each point cloud.
    '''
    num_points = len(X)
    ind_L = [np.random.randint(0,num_points)]  

    distance_to_L = distance(X[ind_L,:], X)

    for i in range(num_landmarks-1):
        ind_max = np.argmax(distance_to_L)
        ind_L.append(ind_max)
        
        dist_temp = distance(X[[ind_max],:], X)

        distance_to_L = np.minimum(distance_to_L, dist_temp)

    return {'indices':ind_L, 'distance_to_L':distance_to_L}

def variance_circle(theta):
    '''
    This function computes the variance of a data set contained in S^1 using the geodesic distance.

    :param theta: 1D-array representing the point cloud in S^1.
    :type theta: array
    
    :return: numpy 2D array -- A Nx2 numpy array with the points drawn as the rows.

    '''
    n = len(theta)
    
    theta = np.array(theta)
    
    m = np.mean(theta)
    
    c = np.array([np.exp(1j*m)*np.exp(2*np.pi*1j*k/n) for k in range(n)])
    
    c = c.reshape((n, -1))
    
    data = np.exp(1j*theta).reshape(-1,n)
    
    mat = np.matmul(c, np.conj(data))
    
    mat_ang = np.sum(np.power(np.angle(mat), 2), axis=0)
    
    ind_min = np.argmin(mat_ang)

    return {'geodmean':np.angle(c[ind_min]), 'geodvar':mat_ang[ind_min]/(n)}

# def partition_unity(dist_matrix, epsilon):
    
#     max_matrix = np.maximum(epsilon-dist_matrix, 0)

#     total = np.sum(max_matrix, axis=0)
    
#     return np.divide(max_matrix, total)

def partition_unity(dist_matrix, weights_vector, epsilon, function_type='cone', truncate_height='None'):
    '''
    Given a data set X and a landmark set L (subset of X).

    This function computes a partition of unity subordinated to metric balls, defined by the formula:

    .. math::
        \phi_{l}(x) = \frac{ \omega_{l} f( \epsilon - d(x,l) ) }{ \sum\limits_{l' \in L} \omega_{l'} f( \epsilon - d(x,l') ) } 

    where f is a non-decreasing, non-negative real-valued function.

    :param dist_matrix: |L| x |X| distance matrix where L is the landmarks set and |X| is the data set. |L| is the size of L and |X| is the size of X.
    :type N: numpy.array

    :param weights_vector: Array of size |L| that define the weight of each landmark point.
    :type weights_vector: numpy.array

    :param epsilon: Radius of the metric balls.
    :type epsilon: float

    :param function_type: 'cone' or 'truncated_cone'. This defines the function f used in the definition of the partition of unity. We provide to options
                          in this implementation a cone centered at the ladmark or a truncated cone centered a the landmark.
    :type function_type: str

    :param truncate_height: Heigh to truncate the cone that shapes tha partition of unity. This is used only if function_type = 'truncated_cone'.
    :type truncate_height: float

    :return: |L| x |X| array. The (i,j) entry in the returned matrix corresponds to \phi_{i}(x_j) for i \in L and x_{j} \in X.

    '''
    
    if function_type == 'cone':
        max_matrix = np.maximum(epsilon-dist_matrix, 0)

    if function_type == 'truncated_cone':
        max_matrix = np.maximum(epsilon-dist_matrix, 0)
        
        th = epsilon/2 if truncate_height == 'None' else truncate_height

        max_matrix = np.minimum(max_matrix, th)
        
    
    max_matrix = weights_vector.reshape(len(weights_vector),-1)*max_matrix
    
    total = np.sum(max_matrix, axis=0)
    
    excluded_points = total == 0

    included_points = np.logical_not(excluded_points)

    result = np.zeros(max_matrix.shape)

    result[:, included_points] = np.divide(max_matrix[:, included_points], total[included_points])
    
    return result

def uneven_partition_unity(dist_matrix, ind_landmarks, weights_vector, epsilon, function_type='cone', truncate_height='None'):
    '''
    Given a data set X and a landmark set L (subset of X).

    This function computes a partition of unity subordinated to metric balls with diferent raious, defined by the formula:

    .. math::
        \phi_{l}(x) = \frac{ \omega_{l} f( \epsilon - d(x,l) ) }{ \sum\limits_{l' \in L} \omega_{l'} f( \epsilon - d(x,l') ) } 

    where f is a non-decreasing, non-negative real-valued function.

    :param dist_matrix: |X| x |X| distance matrix, where X is the data set. 
    :type N: numpy.array

    :param ind_landmarks: Array that contains the indices for the landmarks in L.
    :type N: numpy.array

    :param weights_vector: Array of size |L| that define the weight of each landmark point.
    :type weights_vector: numpy.array

    :param epsilon: Radius of the metric balls.
    :type epsilon: float

    :param function_type: 'cone' or 'truncated_cone'. This defines the function f used in the definition of the partition of unity. We provide to options
                          in this implementation a cone centered at the ladmark or a truncated cone centered a the landmark.
    :type function_type: str

    :param truncate_height: Heigh to truncate the cone that shapes tha partition of unity. This is used only if function_type = 'truncated_cone'.
    :type truncate_height: float

    :return: |L| x |X| array. The (i,j) entry in the returned matrix corresponds to \phi_{i}(x_j) for i \in L and x_{j} \in X.

    '''

    distance_matrix_L = dist_matrix[:,ind_landmarks]

    radii = np.sort(distance_matrix_L, axis=0)[1,:]
    
    if function_type == 'cone':
        max_matrix = np.maximum(radii[:,np.newaxis]-dist_matrix, 0)

    if function_type == 'truncated_cone':
        max_matrix = np.maximum(radii[:,np.newaxis]-dist_matrix, 0)

        th = epsilon/2 if truncate_height == 'None' else truncate_height

        max_matrix = np.minimum(max_matrix, th)
        
    
    max_matrix = weights_vector.reshape(len(weights_vector),-1)*max_matrix
    
    total = np.sum(max_matrix, axis=0)
    
    excluded_points = total == 0

    included_points = np.logical_not(excluded_points)

    result = np.zeros(max_matrix.shape)

    result[:, included_points] = np.divide(max_matrix[:, included_points], total[included_points])
    
    return result

def rotM(a):
    '''
    This function computes the rotation matrix (orientation preserving) in R^3 perpendicular to the vector a.

    :param a: Vector in R^3.
    :type a: numpy.array

    :return: 3 x 3 rotation matrix.
    '''
    a = np.reshape(a, (-1,1)) 
    n = len(a)
    a = a / np.sqrt(np.real(np.vdot(a,a)))

    b = np.zeros(n)
    b[-1] = 1
    b = np.reshape(b, (-1,1))

    c = a - (np.transpose(np.conj(b))@a)*b

    if np.sqrt(np.vdot(c,c)) < 1e-15:
        rot = np.conj(np.transpose(np.conj(b))@a)*np.ones((n,n))
    else:
        c = c / np.sqrt(np.real(np.vdot(c,c)))
        l = np.transpose(np.conj(b))@a
        beta = np.sqrt(1 - np.vdot(l,l))
        rot = np.identity(n) - (1-l)*(c@np.transpose(np.conj(c))) - (1-np.conj(l))*(b@np.transpose(np.conj(b))) + beta*(b@np.transpose(np.conj(c)) - c@np.transpose(np.conj(b)))

    return rot

def lens_coordinates(eta, distance_matrix, epsilon, z_q, weights_vector=None, multiplicity_list=None):
    '''
    This function compute the lens coordinates as presented in `Coordinatizing Data With Lens Spaces and Persistent Cohomology by L. Polanco and J.A. Perea <https://arxiv.org/abs/1905.00350>`_

    :param eta:  m x 3 matrix, representing the cochain of dimension 1 whe will use to compute lencs coordinates. The first 2 coumns represent the 1-simplices inthe cochain and the last column contains the coefficient in the cochain for the corresponding 1-simplex (see staendard output of ripser).
    :type eta: numpy.array

    :param distance_matrix: |L| x |X| distance matrix where L is the landmarks set and |X| is the data set. |L| is the size of L and |X| is the size of X.
    :type distance_matrix: numpy.array

    :param epsilon: Radius of the metric balls used to define the parttion on unity used to compute the lens coordinates.
    :type epsilon: float

    :param z_q: Prime number used to the define the field Z/qZ.
    :type z_q: int

    :param weights_vector: Array of size |L| that define the weight of each landmark point.
    :type weights_vector: numpy.array

    :param multiplicity_list: List containing the indices for each landmark.
    :type multiplicity_list: list or numpy.array

    :return: |X| x |L| complex-valued matrix. Each row contains the lens coordinates for the correponding point in X.
    '''
    
    if weights_vector is None:
        weights_vector = np.ones(distance_matrix.shape[0])

    weights_vector = np.array(weights_vector, dtype=float)

    if multiplicity_list is None:
        multiplicity_list = np.ones(distance_matrix.shape[0])

    multiplicity_list = np.array(multiplicity_list, dtype=int)

    if not(int(sum(multiplicity_list)) == distance_matrix.shape[0]):
        print('ERROR: The matrix "distance_matrix" must have as many rows as the total of points in the multiset defined by "multiplicity_list".')
        return

    if weights_vector.shape[0] != distance_matrix.shape[0]:
        print('ERROR: The vector of weights "weights_vector" must have the same number of columns/rows as "distance_matrix".')
        return
    
    num_landmarks = len(multiplicity_list)
    num_points = distance_matrix.shape[1]
    
    ETA = np.ones((num_landmarks,num_landmarks), dtype=complex)

    for j in range(num_landmarks):
        for k in range(j+1, num_landmarks):
            simplex = np.array([k,j])
            
            ind = np.where((eta[:,[0,1]] == (simplex[0], simplex[1])).all(axis=1))[0]
            
            if ind.size > 0:

                ETA[j,k] = np.power(z_q, eta[ind,2][0])
                ETA[k,j] = np.power(ETA[j,k], -1)

    ETA = ETA[np.repeat(np.arange(num_landmarks), multiplicity_list),:][:,np.repeat(np.arange(num_landmarks), multiplicity_list)]

    sqrt_part_unity = np.sqrt(np.transpose(partition_unity(distance_matrix, weights_vector, epsilon)))

    K = np.argmin(distance_matrix, axis=0)

    ETA_tilda = ETA[K,:]

    lens = np.multiply(sqrt_part_unity, ETA_tilda)

    lens = np.array(lens, dtype='complex')

    return lens.reshape( (num_points, int(sum(multiplicity_list))) )


def lens_coordinates_pullback(eta, distance_matrix, weights_vector, function, epsilon, z_q):
    
    num_landmarks = max(function)+1
    num_points = distance_matrix.shape[1]
    
    ETA = np.ones((num_landmarks,num_landmarks), dtype=complex)

    for j in range(num_landmarks):
        for k in range(j+1, num_landmarks):
            simplex = np.array([k,j])
            
            ind = np.where((eta[:,[0,1]] == (simplex[0], simplex[1])).all(axis=1))[0]
            
            if ind.size > 0:

                ETA[j,k] = np.power(z_q, eta[ind,2][0])
                ETA[k,j] = np.power(ETA[j,k], -1)

    ETA = ETA[function,:][:,function]
    # print(distance_matrix.shape)
    # print(weights_vector)
    sqrt_part_unity = np.sqrt(np.transpose(partition_unity(distance_matrix, weights_vector, epsilon)))
    
    # print(sqrt_part_unity)

    # non_zero_ind = np.sum(sqrt_part_unity**2, axis=1) != 0 

    # print(sqrt_part_unity[:,non_zero_ind])
    
    K = np.argmin(distance_matrix, axis=0)

    ETA_tilda = ETA[K, :]

    lens = np.multiply(sqrt_part_unity, ETA_tilda)

    lens = np.array(lens, dtype='complex')

    return lens.reshape( (-1, len(function)) )

# def lens_coordinates_with_mltiplicity(eta, distance_matrix, multiplicity_list,epsilon, z_q):

#     num_landmarks = distance_matrix.shape[0]
#     num_points = distance_matrix.shape[1]

#     ETA = np.ones((num_landmarks,num_landmarks), dtype=complex)

#     ind_L_multipliciy = np.repeat(np.arange(len(multiplicity_list)), multiplicity_list)

#     for j in ind_L_multipliciy:
#         for k in ind_L_multipliciy[1:]:
#             simplex = np.array([k,j])
            
#             ind = np.where((eta[:,[0,1]] == (simplex[0], simplex[1])).all(axis=1))[0]
            
#             if ind.size > 0:

#                 ETA[j,k] = np.power(z_q, eta[ind,2][0])
#                 ETA[k,j] = np.power(ETA[j,k], -1)

#     sqrt_part_unity = np.sqrt(np.transpose(partition_unity(distance_matrix, epsilon)))

#     K = np.argmin(distance_matrix, axis=0)

#     ETA_tilda = ETA[K,:]

#     lens = np.multiply(sqrt_part_unity, ETA_tilda)

#     lens = np.array(lens, dtype='complex')
#     return lens.reshape( (num_points, num_landmarks) )

# 1/N (Σ d(x_j, P_U^c(x_j))^2 )
# Computes the mean square distance from X to its projection on the orhtogonal complement of the columns of U
def sqr_ditance_orthogonal_projection(U, X):
    norm_colums = np.sqrt(1 - np.linalg.norm(np.transpose(np.conj(U))@X, axis=0)**2)
    return np.mean(np.power(np.arccos( norm_colums ), 2))

# 1/N (Σ d(x_j, P_U(x_j))^2 )
# Computes the mean square distance from X to its projection onto the columns of U
def sqr_ditance_projection(U, X):
    norm_colums = np.maximum(np.minimum(np.linalg.norm(np.transpose(np.conj(U))@X, axis=0), 1), -1)
    return np.mean(np.power(np.arccos( norm_colums ), 2))

def distance_covariance_squared(A,B):
    '''
    This function computes the Sample Distance Covariance squared given 2 distances matrices, it is computed by the following formula

    .. math::
        dCov^2(A,B) = \frac{1}{n_^2} \sum\limits_{j,k=1}^{n} A_{jk}B{jk}

    :type A: numpy array
    :param A: Distance matrix

    :type B: numpy array
    :param B: Distance matrix

    :return: float - Sample distance covariance squared
    '''
    if A.shape[0] == B.shape[0]:
        return np.sum(np.multiply(A,B))/(A.shape[0]**2)
    else:
        print('Distance matrices must be square and have the same dimension')
        return

def distance_correlation_squared(A,B):
    '''
    This function computes the Sample Distance Covariance squared given 2 distances matrices, it is computed by the following formula

    .. math::
        dCor^2(A,B) = \frac{dCov^2(X,Y)}{dCov^2(X,X) dCov^2(Y,Y)}

    :type A: numpy array
    :param A: Distance matrix

    :type B: numpy array
    :param B: Distance matrix

    :return: float - Distance correlation squared
    '''
    if A.shape[0] == B.shape[0]:
        dcov_AB = distance_covariance_squared(A,B)
        dcov_AA = distance_covariance_squared(A,A)
        dcov_BB = distance_covariance_squared(B,B)

        if dcov_AA*dcov_BB != 0:
            return dcov_AB/(dcov_AA*dcov_BB)
        else:
            return 0
    else:
        print('Distance matrices must be square and have the same dimension')
        return