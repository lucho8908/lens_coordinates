from __future__ import division

import time
import timeit
import numpy as np
import scipy as sp
import sklearn as sk
import math
import cmath
import itertools
import sys
import matplotlib
matplotlib.use('agg')

from ripser import ripser, plot_dgms

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import LPCA as lpca

np.set_printoptions(precision=2)

np.random.seed( 42 )

def euclidean_distance(X,Y):
    if X.ndim == 1:
        X = X.reshape((1,len(X)))
    if Y.ndim == 1:
        Y = Y.reshape((1,len(X)))

    return sp.spatial.distance.cdist(X,Y, metric='euclidean')

def lens_distance(X,Y):
    n_X = X.shape[0]
    n_Y = Y.shape[0]

    dimension = X.shape[1]

    X = np.array(X)
    Y = np.array(Y)

    q = 3

    m = X@np.transpose(Y)
    m = np.maximum(np.minimum(m, 1), -1)
    dist_X_Y = np.arccos(m)

    zq = np.array([[np.cos(2*np.pi/q), -np.sin(2*np.pi/q)], [np.sin(2*np.pi/q), np.cos(2*np.pi/q)]])
    ZQ = np.kron(np.eye( int(dimension/2) ), zq)

    ZQ_k = ZQ

    for k in range(1,q):
        arg = np.minimum( np.maximum( X@(ZQ_k@np.transpose(Y)), -1 ), 1 )
        dist_X_Y = np.minimum(dist_X_Y,  np.arccos( arg ))

        ZQ_k = ZQ_k@ZQ
    
    return dist_X_Y

# This function geenrates points in D^2 with a slight bias towards the frontier. 
def gen_S3(N):
    
    X = np.random.uniform(-1,1,(N,4))
    
    norms = np.sqrt(np.diag(X@np.transpose(X)))
    
    norms = np.tile(np.array([norms]).transpose(), (1, 4))
    
    X = X / norms
    
    return X

def gen_s1_d2(N, bias=False):
    '''
    This function generates points in a solid torus: S^1 x D^2

    :type N: int
    :param N: Number of points to generate.

    :type bias: bool
    :param bias: By default = False. If True the it generates a sample bias towards S^1 x {0}.
    '''
    theta = np.random.uniform(0, 2*np.pi, N)

    phi = np.random.uniform(0, 2*np.pi, N)

    if bias:
        r1 = np.random.uniform(0, 1, int((90*N)/100))
        r2 = np.zeros( int((10*N)/100) )
        r = np.concatenate((r1,r2))
    else: 
        r = np.random.uniform(0, 1, N)

    z = np.exp(theta*1j)
    w = r*np.exp(phi*1j)

    return np.array([z, w]).transpose()

def gen_s1_d2_seed(N):
    #theta = np.random.uniform(0, 2*np.pi, N)
    theta = np.linspace(0, 2*np.pi, N)

    phi = np.random.uniform(0, 2*np.pi, N)
    r = np.zeros( N )
 
    z = np.exp(theta*1j)
    w = r*np.exp(phi*1j)

    return np.array([z, w]).transpose()


q = 3

t0 = time.time()
#X = gen_S3(5000) # Data set X

num_p = 10000

X_solid_torus = gen_s1_d2(num_p)

num_seed = 10 # Number of poits to seed in S^1 x {0} \subset S^1 x D^2 \subset C^2

seed_solid_torus = gen_s1_d2_seed(num_seed) # Generates points in S^1 x {0} \subset C^2

X_solid_torus = np.concatenate((X_solid_torus, seed_solid_torus), axis=0)

X = np.array([np.sqrt(1 - np.abs(X_solid_torus[:,1])**2)*X_solid_torus[:,0], X_solid_torus[:,1]]).transpose()

X = np.array([np.real(X[:,0]), np.imag(X[:,0]), np.real(X[:,1]), np.imag(X[:,1])]).transpose()


seed_indices = list(np.arange(num_p, num_p + num_seed))

t1 = time.time()
print('Generate X = {} s'.format(t1-t0))

t0 = time.time()
dm_X = lens_distance(X,X)
t1 = time.time()
print('Distance matrix = {} s'.format(t1-t0))

cohomology_test = lpca.minmax_subsample_distance_matrix(dm_X, 1000)['indices']

X = X[cohomology_test, :]

dm_X = dm_X[cohomology_test,:][:,cohomology_test]

def my_knn(distance_matrix, k):
    
    ind = np.argsort(distance_matrix, axis=1)
    not_knn = ind[:,k:]
    
    res = np.copy(distance_matrix)
    for i in range(len(distance_matrix)):
        
        res[i, not_knn[i,:]] = 0
    
    return res
        
t0 = time.time()
knn_dm = my_knn(dm_X, 10)

knn_dm = np.maximum(knn_dm, knn_dm.T)
t1 = time.time()
print('KNN = {} s'.format(t1-t0))

from sklearn.utils import graph_shortest_path

t0 = time.time()
D = graph_shortest_path.graph_shortest_path(knn_dm)
t1 = time.time()
print('Shortest path = {} s'.format(t1-t0))


from sklearn.manifold import MDS

t0 = time.time()
mds = MDS(n_components=4, dissimilarity="precomputed", n_jobs=8)

iso = mds.fit_transform(D)
t1 = time.time()
print('MDS = {} s'.format(t1-t0))

print(iso.shape)

import os

os.system('mkdir lens_iso')
os.system('rm -r lens_iso/*')

diagrams = ripser(iso, coeff=2, maxdim=1)['dgms']

H_1_diagram = diagrams[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)
a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

larg_per = b-a
second_larg_per = H_1_diagram[H_1_persistence_sort_ind[-2], 1] - H_1_diagram[H_1_persistence_sort_ind[-2], 0]

os.system('echo lens, q=2, {}, {} >> lens_iso/largest_persistence.txt'.format(larg_per, second_larg_per))

plot_dgms(diagrams, xy_range=[-0.05,0.8,-0.05,0.8])

plt.title(r'$PH^1(Iso(X)\subset \mathbb{R}^{ 4 }, \mathbb{Z}_2)$')
plt.savefig('./lens_iso/lens_iso_q_2.png')
plt.title('Persistent homolgy Isomap transformation')
plt.close()


diagrams = ripser(iso, coeff=3, maxdim=1)['dgms']

H_1_diagram = diagrams[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)
a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

larg_per = b-a
second_larg_per = H_1_diagram[H_1_persistence_sort_ind[-2], 1] - H_1_diagram[H_1_persistence_sort_ind[-2], 0]

os.system('echo lens, q=3, {}, {} >> lens_iso/largest_persistence.txt'.format(larg_per, second_larg_per))

plot_dgms(diagrams, xy_range=[-0.05,0.8,-0.05,0.8])

plt.title(r'$PH^1(Iso(X)\subset \mathbb{R}^{ 4 }, \mathbb{Z}_2)$')
plt.savefig('./lens_iso/lens_iso_q_3.png')
plt.title('Persistent homolgy Isomap transformation')
plt.close()