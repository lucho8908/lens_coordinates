from __future__ import division

import os
import time
import numpy as np
import scipy as sp
import math
import cmath
import itertools
import sys
import matplotlib
matplotlib.use('agg')

from ripser import ripser
from persim import plot_diagrams

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import LPCA as lpca

np.set_printoptions(precision=2)

np.random.seed( 42 )

# This function geenrates points in D^2 with a slight bias towards the frontier. 
def gen_D2(N):
    
    theta = np.random.uniform(0, 2*math.pi, N)
    r0 = np.random.uniform(0, 1, int(90*N/100))
    r1 = np.random.uniform(0.8, 1, int(10*N/100))

    r = np.concatenate((r0,r1))

    x = np.multiply( r, np.cos(theta))
    y = np.multiply( r, np.sin(theta))

    x = x.flatten()
    y = y.flatten()

    return np.column_stack((x,y))

def gen_S1(N):
    theta = np.random.uniform(0, 2*math.pi, N)

    x = np.cos(theta)
    y = np.sin(theta)

    x = x.flatten()
    y = y.flatten()

    return np.column_stack((x,y))


# Distance function in X = (D^2 ∐ S^1)/f, with f:S^1 ---> S^1 defined by f(x) = x^3.
# Given 0<r<1 we define:
#   phi = max{1/(1-r)(|x|+|y|)/2 - r/(1-r), 0}
#   d(x,y) = (1-phi)|x-y| + (phi)|x^3-y^3|
#
#   |
#   1                  /
#   |                 /
#   |                /
#   |               /          --> phi
#   |              /
#   |             /
#   |            /
#   +-----------r------1-
def quotient_distance(X,Y):
    if X.ndim == 1:
        X = X.reshape((1,len(X)))
    if Y.ndim == 1:
        Y = Y.reshape((1,len(Y)))

    n_X = X.shape[0]
    n_Y = Y.shape[0]

    dimension = X.shape[1]
    
    norm_X = np.linalg.norm(X, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)

    X_in_S = np.isclose(norm_X, 1, rtol=1e-05, atol=1e-08, equal_nan=False)
    X_in_D = np.logical_not(X_in_S)
    
    DX = X[X_in_D,:]
    SX = X[X_in_S,:]

    Y_in_S = np.isclose(norm_Y, 1, rtol=1e-05, atol=1e-08, equal_nan=False)
    Y_in_D = np.logical_not(Y_in_S)
    
    DY = Y[Y_in_D,:]
    SY = Y[Y_in_S,:]

    dist_DX_DY = sp.spatial.distance.cdist(DX,DY, metric='euclidean')

    dist_DX_SY = sp.spatial.distance.cdist(DX,SY, metric='euclidean')

    dist_SX_DY = sp.spatial.distance.cdist(SX,DY, metric='euclidean')    

    dist_SX_SY = sp.spatial.distance.cdist(SX,SY, metric='euclidean')

    q = 3

    Z_q = np.array([[np.cos(2*np.pi/q), -np.sin(2*np.pi/q)], [np.sin(2*np.pi/q), np.cos(2*np.pi/q)]])
    ZQ = np.kron(np.eye( int(dimension/2) ), Z_q)

    ZQ_k = ZQ

    for k in range(1,q):
        dist_DX_SY = np.minimum(dist_DX_SY, sp.spatial.distance.cdist(DX, SY@np.transpose(ZQ_k), metric='euclidean'))
        dist_SX_DY = np.minimum(dist_SX_DY, sp.spatial.distance.cdist(SX, DY@np.transpose(ZQ_k), metric='euclidean'))
        dist_SX_SY = np.minimum(dist_SX_SY, sp.spatial.distance.cdist(SX, SY@np.transpose(ZQ_k), metric='euclidean'))
        ZQ_k = ZQ_k@ZQ

    dist_M = np.concatenate((np.concatenate((dist_DX_DY, dist_DX_SY), axis=1), np.concatenate((dist_SX_DY, dist_SX_SY), axis=1)), axis=0)
    
    dist_M = dist_M.reshape((n_X, n_Y))

    return dist_M

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

q = 3

t0 = time.time()

D = gen_D2(13000)
S = gen_S1(2000)

X = np.concatenate((D,S), axis=0)

t1 = time.time()
print('Generate X = {} s'.format(t1-t0))

cohomology_test = np.copy(X)

os.system('mkdir quotient_homolgy_lpca')
os.system('rm -r quotient_homolgy_lpca/*')

test_dm = quotient_distance(cohomology_test, cohomology_test)

sub_ind = lpca.minmax_subsample_distance_matrix(test_dm, 1000)['indices']


plot_diagrams(ripser( test_dm[sub_ind,:][:,sub_ind] , coeff=3, maxdim=1, distance_matrix=True)['dgms'], xy_range=[-0.05,0.8,-0.05,0.8])
plt.title(r'$PH_i(X, \mathbb{Z}_3)$')
plt.savefig('./quotient_homolgy_lpca/homology_original_p_3.png', format='png')
plt.close()

plot_diagrams(ripser( test_dm[sub_ind,:][:,sub_ind] , coeff=2, maxdim=1, distance_matrix=True)['dgms'], xy_range=[-0.05,0.8,-0.05,0.8])
plt.title(r'$PH_i(X, \mathbb{Z}_2)$')
plt.savefig('./quotient_homolgy_lpca/homology_original_p_2.png', format='png')
plt.close()

#------------------------------------------------------------------------------
# Plot data set X
#------------------------------------------------------------------------------
fig, ax = plt.subplots()

# c = np.array([np.absolute(X[i,0] + 1j*X[i,1]) for i in range(len(X))])
#c = np.array([ X[i,0] for i in range(len(X))])
c = np.array([np.mod( np.angle(X[i,0] + 1j*X[i,1]), 2*np.pi/q ) for i in range(len(X))])

cmap = matplotlib.cm.get_cmap('hsv')
normalize = matplotlib.colors.Normalize(vmin=min(c), vmax=max(c))
colors = np.array([cmap(normalize(value)) for value in c])

fig = plt.scatter(X[:,0], X[:,1], color=colors)
#------------------------------------------------------------------------------

t0 = time.time()

dm_D = quotient_distance(D,D)
dm_S = quotient_distance(S,S)

t1 = time.time()
print('Distance matrix = {} s'.format(t1-t0))

#------------------------------------------------------------------------------
# Lets create abd artificial landmark set

# q = 3

# alpha = 2*np.pi/q

# #my_L = np.array([np.exp(0*1j) , np.exp((alpha/4)*1j), np.exp((2*alpha/4)*1j), np.exp((3*alpha/4)*1j), (1-alpha/4)*np.exp(0*1j), (1-alpha/4)*np.exp(alpha*1j), (1-alpha/4)*np.exp(2*alpha*1j)])

# # my_L = np.array([np.exp(0*1j) , np.exp((alpha/7)*1j), np.exp((2*alpha/7)*1j), np.exp((3*alpha/7)*1j), np.exp((4*alpha/7)*1j), np.exp((5*alpha/7)*1j), np.exp((6*alpha/7)*1j), 
# #     (0.4)*np.exp(0*1j), (0.4)*np.exp(alpha*1j), (0.4)*np.exp(2*alpha*1j)])

# ir = 0.37
# my_L = np.array([np.exp(0*1j) , np.exp((alpha/7)*1j), np.exp((2*alpha/7)*1j), np.exp((3*alpha/7)*1j), np.exp((4*alpha/7)*1j), np.exp((5*alpha/7)*1j), np.exp((6*alpha/7)*1j),
#     (ir)*np.exp(0*1j), (ir)*np.exp((2*np.pi/6)*1j), (ir)*np.exp((2*2*np.pi/6)*1j), (ir)*np.exp((3*2*np.pi/6)*1j),(ir)*np.exp((4*2*np.pi/6)*1j),
#     (ir)*np.exp((5*2*np.pi/6)*1j)])

# my_L = np.transpose(np.array((np.real(my_L), np.imag(my_L))))

# dm_my_L = quotient_distance(my_L, my_L)

# result_my_L = ripser(dm_my_L, coeff=3, do_cocycles=True, maxdim=1, distance_matrix=True)

# #my_eta = result_my_L['cocycles'][1][0]

# dm_my_X = quotient_distance(my_L, X)

#------------------------------------------------------------------------------

num_points_sample = 70 # Number of pointsin the min-max subsample

t0 = time.time()

subsam_D = lpca.minmax_subsample_distance_matrix(dm_D, num_points_sample - 10) # Subsample L\subset X

subsam_S = lpca.minmax_subsample_point_cloud(S, 10, sp.spatial.distance.cdist)

t1 = time.time()
print('Maxmin subsampl = {} s'.format(t1-t0))

ind_L_D = subsam_D['indices'] 
ind_L_S = subsam_S['indices']

L = np.concatenate((D[ind_L_D, :], S[ind_L_S, :]))

#------------------------------------------------------------------------------
plt.scatter(D[ind_L_D,0], D[ind_L_D,1], color='black')
plt.scatter(S[ind_L_S,0], S[ind_L_S,1], color='black')

# plt.scatter(my_L[:,0], my_L[:,1], color='red')

plt.savefig('quotient.png')
plt.close()
#------------------------------------------------------------------------------

dm_X = quotient_distance(L, X)

dm_L = quotient_distance(L, L) # Distace matrix for L


t0 = time.time()
result = ripser(dm_L, coeff=q, do_cocycles=True, maxdim=1, distance_matrix=True) # Persistent cohomology
t1 = time.time()
print('Persistent homology = {} s'.format(t1-t0))

diagrams = result['dgms'] # Persisten diagrams
cocycles = result['cocycles'] 
# D = result['dm'] # Distance matrix used by Ripser to compute persistence

H_1 = cocycles[1] # PH^1(R(L); Z_q)
H_1_diagram = diagrams[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)

#------------------------------------------------------------------------------
# Plot persitent homology of X: H_*(L; Z_q)
#------------------------------------------------------------------------------
plot_diagrams(result['dgms'], xy_range=[-0.05,0.8,-0.05,0.8], size=50)
plt.title(r'$PH^1(L, \mathbb{Z}_3)$' )
plt.savefig('quotient_homoloy.png')
plt.close()

plot_diagrams( ripser(dm_L, coeff=2, maxdim=1, distance_matrix=True)['dgms'], xy_range=[-0.05,0.8,-0.05,0.8], size=50)
plt.title(r'$PH^1(L, \mathbb{Z}_2)$' )
plt.savefig('quotient_homoloy_2.png')
plt.close()
#------------------------------------------------------------------------------

a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

my_eta = H_1[H_1_persistence_sort_ind[-1]] # Cochain representtive of the largest class in PH^1(R(L); Z_q)

# epsilon = (a + (b - a)*0.99)/2 # Epsilon is the radius e usefor the balls with centers in the landmarks.
print(a, b)
epsilon = b*(1/2)

# dm_L = quotient_distance(my_L, my_L)

# np.fill_diagonal(dm_L, 10)

# r_j = np.min(dm_L, axis=1)

# epsilon = np.reshape(1.5*r_j, (-1,1))

print(epsilon)
# We nee to verify PH^1(R(L); Z_q) has a class with perssitence long enough
# if not(a<epsilon and 2*epsilon<b):
#     print('{}WARNING: The largest class (a,b) in PH^1(R(L); Z_q) is not long enough: 2a is NOT smaller than b.{}'.format('\033[33m', '\033[0m'))
dist_to_L = np.min(dm_X, axis=0)
cover_r = max(dist_to_L)

# if cover_r > epsilon:
#     print('{}WARNING: Covering radius is larger than epsilon. Some points in X will be ignored.{}'.format('\033[33m', '\033[0m'))

#     points_covered = dist_to_L < epsilon
#     X = X[points_covered, :]
#     dm_X = dm_X[:, points_covered]
#     print('{}New data array shape = {}{}'.format('\033[33m', dm_X.shape ,'\033[0m'))
#     colors = colors[points_covered]

z_q = np.exp((2*np.pi*1j)/q) # z_q is a root of unity != 1

#points_covered = dist_to_L < epsilon

is_in_landmark_ball = np.maximum( np.subtract(epsilon, dm_X),0 )
is_in_any_ball = np.logical_not(np.isclose(np.sum(is_in_landmark_ball, axis=0), 0))
points_covered = is_in_any_ball
X = X[points_covered, :]
dm_X = dm_X[:, points_covered]
print('{}New data array shape = {}{}'.format('\033[33m', dm_X.shape ,'\033[0m'))

colors = colors[points_covered]

t0 = time.time()
lens = lpca.lens_coordinates(my_eta, dm_X, weights_vector=np.ones(dm_X.shape[0]), epsilon = epsilon , z_q=z_q)
t1 = time.time()
print('Lens coordinates = {} s'.format(t1-t0))

#------------------------------------------------------------------------------
cohomology_test = []
for row in np.transpose(lens):
    cohomology_test.append(np.array([np.real(row), np.imag(row)]))
cohomology_test = np.array(cohomology_test).reshape((-1,dm_X.shape[1]))
cohomology_test = np.transpose(cohomology_test)

lens_dm = lens_distance(cohomology_test, cohomology_test)

sub_ind = lpca.minmax_subsample_distance_matrix(lens_dm, 1000)['indices']

lens_dm = lens_distance(cohomology_test, cohomology_test)
plot_diagrams(ripser( lens_dm[sub_ind, :][:, sub_ind] , coeff=3, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH^1(f(X), \mathbb{Z}_3)$' )
plt.savefig('./quotient_homolgy_lpca/homology_lpca_iteration_{}_p_3.png'.format(0), format='png')
plt.close()

plot_diagrams(ripser( lens_dm[sub_ind, :][:, sub_ind] , coeff=2, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH^1(f(X), \mathbb{Z}_2)$' )
plt.savefig('./quotient_homolgy_lpca/homology_lpca_iteration_{}_p_2.png'.format(0), format='png')
plt.close()

#------------------------------------------------------------------------------

XX = np.transpose(lens) # XX matrix in C^(landmark x num_points)

XX = XX / (np.ones((len(XX), 1))*np.sqrt(np.real(np.diag(np.transpose(np.conj(XX))@XX)))) # we normalize XX 

variance = []
run_times = []

tolerance = 0.02 # User parameter used to set up the first projection

t0 = time.time()

svd_t0 = time.time()
U, s, V = np.linalg.svd(XX, full_matrices=True) # compute the full SVD of XX, XX = USV*
svd_t1 = time.time()
run_times.append(svd_t1 - svd_t0)

v_0 = lpca.sqr_ditance_projection(U[:, 0:1], XX)
v_1 = 0

k_break = len(U)

for i in range(2,len(U)+1):
    v_1 = lpca.sqr_ditance_projection(U[:, 0:i], XX)

    difference_v = abs(v_0 - v_1)

    if difference_v < tolerance:
        k_break = i
        break
    
    v_0 = v_1

U_tilde = U[:, 0:k_break]

variance.append( v_0 ) # lost variance inthe projection

XX = np.transpose(np.conj(U_tilde))@XX # project XX into the direction given by U_tilde

XX = XX / (np.ones((len(XX), 1))*np.sqrt(np.real(np.diag(np.transpose(np.conj(XX))@XX)))) # normalize the result

#------------------------------------------------------------------------------
cohomology_test = []
for row in XX:
    cohomology_test.append(np.array([np.real(row), np.imag(row)]))
cohomology_test = np.array(cohomology_test).reshape((-1,dm_X.shape[1]))
cohomology_test = np.transpose(cohomology_test)

sub_ind = lpca.minmax_subsample_distance_matrix(lens_dm, 1000)['indices']

lens_dm = lens_distance(cohomology_test, cohomology_test)

plot_diagrams(ripser( lens_dm[sub_ind, :][:, sub_ind] , coeff=3, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH^1(f(X)\subset \mathbb{C}^{ %d }, \mathbb{Z}_3)$' % XX.shape[0] )
plt.savefig('./quotient_homolgy_lpca/homology_lpca_iteration_{}_p_3.png'.format(1), format='png')
plt.close()

plot_diagrams(ripser( lens_dm[sub_ind, :][:, sub_ind] , coeff=2, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH^1(f(X)\subset \mathbb{C}^{ %d }, \mathbb{Z}_2)$' % XX.shape[0] )
plt.savefig('./quotient_homolgy_lpca/homology_lpca_iteration_{}_p_2.png'.format(1), format='png')
plt.close()
#------------------------------------------------------------------------------

i = 2
while XX.shape[0] > 2:
    print(XX.shape[0])
    svd_t0 = time.time()
    val_smallest, vec_smallest = sp.sparse.linalg.eigs(XX@np.transpose(np.conj(XX)), k=1, which='LM', sigma=0)

    svd_t1 = time.time()
    run_times.append(svd_t1 - svd_t0)

    rotation_matrix = lpca.rotM(vec_smallest)

    Y = rotation_matrix@XX

    Y = np.delete(Y, (-1), axis=0)

    variance.append( lpca.sqr_ditance_orthogonal_projection(vec_smallest, XX) )

    XX = Y / (np.ones((len(Y), 1))*np.sqrt(np.real(np.diag(np.transpose(np.conj(Y))@Y))))

    if XX.shape[0] == 2:
        #------------------------------------------------------------------------------
        cohomology_test = []
        for row in XX:
            cohomology_test.append(np.array([np.real(row), np.imag(row)]))
        cohomology_test = np.array(cohomology_test).reshape((-1,dm_X.shape[1]))
        cohomology_test = np.transpose(cohomology_test)
        
        sub_ind = lpca.minmax_subsample_distance_matrix(lens_dm, 2000)['indices']

        lens_dm = quotient_distance(cohomology_test, cohomology_test)

        diagrams = ripser(lens_dm[sub_ind, :][:, sub_ind], coeff=3, maxdim=1, distance_matrix=True)['dgms']

        H_1_diagram = diagrams[1] # dgm(PH^1(R(L); Z_q))
        H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
        H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)
        a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
        b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

        larg_per = b-a
        second_larg_per = H_1_diagram[H_1_persistence_sort_ind[-2], 1] - H_1_diagram[H_1_persistence_sort_ind[-2], 0]

        os.system('echo LPCA >> quotient_iso/largest_persistence.txt')

        os.system('echo S^1, q=3, {}, {} >> quotient_iso/largest_persistence.txt'.format(larg_per, second_larg_per))

        plot_diagrams(diagrams, xy_range=[-0.05,0.8,-0.05,0.8])
        plt.title(r'$PH^1(f(X)\subset \mathbb{C}^{ %d }, \mathbb{Z}_3)$' % XX.shape[0] )
        plt.savefig('./quotient_homolgy_lpca/homology_lpca_iteration_{}_p_3.png'.format(i), format='png')
        plt.close()

        diagrams = ripser(lens_dm[sub_ind, :][:, sub_ind], coeff=2, maxdim=1, distance_matrix=True)['dgms']

        H_1_diagram = diagrams[1] # dgm(PH^1(R(L); Z_q))
        H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
        H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)
        a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
        b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

        larg_per = b-a
        second_larg_per = H_1_diagram[H_1_persistence_sort_ind[-2], 1] - H_1_diagram[H_1_persistence_sort_ind[-2], 0]

        os.system('echo S^1, q=2, {}, {} >> quotient_iso/largest_persistence.txt'.format(larg_per, second_larg_per))

        plot_diagrams(diagrams, xy_range=[-0.05,0.8,-0.05,0.8])
        plt.title(r'$PH^1(f(X)\subset \mathbb{C}^{ %d }, \mathbb{Z}_2)$' % XX.shape[0] )
        plt.savefig('./quotient_homolgy_lpca/homology_lpca_iteration_{}_p_2.png'.format(i), format='png')
        plt.close()
    i += 1
    #------------------------------------------------------------------------------

ZZ = np.copy(XX)

# We are centering the data in C^2xN for visualization purposes
#m = np.mean(ZZ, axis=1)
#ZZ = ZZ - m[:, np.newaxis]

vals, vecs = np.linalg.eig( ZZ@np.transpose(np.conj(ZZ)) )

ind_min = np.argmin(np.abs(vals))
val_smallest = vals[ind_min]
vec_smallest = vecs[:,ind_min]

rotation_matrix = lpca.rotM(vec_smallest)

embedding_S3 = rotation_matrix@ZZ

embedding_S3 = embedding_S3 / (np.ones((len(embedding_S3), 1))*np.sqrt(np.real(np.diag(np.transpose(np.conj(embedding_S3))@embedding_S3))))

embedding_S3 = np.transpose(embedding_S3)

# Here we contonue the dimensionality reduction with uncentered data.
svd_t0 = time.time()
vals, vecs = np.linalg.eig( XX@np.transpose(np.conj(XX)) )
svd_t1 = time.time()
run_times.append(svd_t1 - svd_t0)

ind_min = np.argmin(np.abs(vals))
val_smallest = vals[ind_min]
vec_smallest = vecs[:,ind_min]

rotation_matrix = lpca.rotM(vec_smallest)

variance.append( lpca.sqr_ditance_orthogonal_projection(vec_smallest.reshape(2, 1), XX) )

Y = rotation_matrix@XX

Y = np.delete(Y, (-1), axis=0)

XX = Y / (np.ones((len(Y), 1))*np.sqrt(np.real(np.diag(np.transpose(np.conj(Y))@Y))))
t1 = time.time()
print('LPCA = {} s'.format(t1-t0))


theta = np.angle(XX)

plt.scatter(np.cos(theta), np.sin(theta), color=colors)
plt.savefig('quotient_embedding_in_s1.png')
plt.close()

theta = 3*theta
plt.scatter(np.cos(theta), np.sin(theta), color=colors)
plt.savefig('quotient_embedding_in_s1_2.png')
plt.close()

variance_s1 = lpca.variance_circle(theta[0])['geodvar']

variance = [variance_s1] + variance[::-1]
#print(variance)


variance = np.cumsum(variance)

variance = variance/variance[-1]
print(variance)
x_axis = list(range(1,k_break+1)) + [len(U)]
#print(x_axis)
fig, ax = plt.subplots()

fig = plt.plot(x_axis, variance, 'o-')

ax.axes.axhline(y=0.7, linestyle='--', color='rebeccapurple')

plt.xticks(np.array(x_axis)[[0,3,5,8,12,15,18,len(x_axis)-1]])

plt.xlabel('Lens embedding dimension')

plt.ylabel('Percentage of cumulative variance')

#ax.set_xticklabels([r'$S^{%s}$' % i for i in range(1,2*num_points_sample-1 +1,2)])

plt.ylim(0,1.01)

#plt.yticks(variance)

plt.savefig('quotient_explained_variance.png')

plt.close()

#------------------------------------------------------------------------------
def gen_wire_positive(N, q):
    #theta, phi = np.linspace(0, 2*np.pi/q, N), np.linspace(0, 2*np.pi/q, N)
    theta, phi = np.linspace(0, 2*np.pi, N), np.linspace(0, 2*np.pi, N)
    THETA, PHI = np.meshgrid(theta, phi)
    
    x = np.cos(THETA) * np.cos(PHI)
    y = np.cos(THETA) * np.sin(PHI)
    z = (np.pi/q)*np.sin(THETA)

    return [x, y, z]

wire_torus = gen_wire_positive(60, q)

print('mean_embeding', np.mean(np.abs(embedding_S3), axis=0))
print('var embeding', np.var(np.abs(embedding_S3), axis=0))

cloud_R3 = []
for p in embedding_S3:
    p1 = p[1]
    p2 = p[0]

    arg_z = np.mod(np.angle(p1), 2*np.pi)

    theta = np.mod(arg_z, 2*np.pi/q)

    k = np.floor((arg_z - theta) / (2*np.pi/q))    

    phi = np.mod(np.angle(p2), 2*np.pi)
    r = np.abs(p2)

    phi = phi - k*(2*np.pi/q)

    # x = (1 + r*np.cos(phi))*np.cos(theta)
    # y = (1 + r*np.cos(phi))*np.sin(theta)
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    #z = r*np.sin(phi)

    #z = (q/np.pi)*(theta - np.pi/q)*np.sqrt(max(1-r**2, 0))
    z = (theta - np.pi/q)*np.sqrt(max(1-r**2, 0))

    cloud_R3.append([x,y,z])

cloud_R3 = np.array(cloud_R3)

w, h = plt.figaspect(0.5)
fig = plt.figure(figsize=(w,h))
#------------------------------------------------------------------------------
ax = fig.add_subplot(2, 2, 1, projection='3d')

ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors)


# Dark wires
theta = np.linspace(0, np.pi/2, 100)
    
phi = np.linspace(0, 4*np.pi/3, 3) 

for i in range(len(phi)):
    x = np.cos(theta) * np.sin(phi[i])
    y = np.cos(theta) * np.cos(phi[i])
    z = np.sin(theta)

    ax.plot(x,y,z, color=(0, 0, 0, 0.9))

# Light horizontal wires
theta = np.linspace(0, 2*np.pi, 100)
    
phi = np.linspace(0, 2*np.pi, 10) 

for i in range(len(phi)):
    x = np.cos(theta) * np.cos(phi[i])
    y = np.cos(theta) * np.sin(phi[i])
    z = np.sin(theta)

    ax.plot(x,y,z, color=(0, 0, 0, 0.1))
    
# Light vertical wires
theta = np.linspace(0, 2*np.pi, 10)
    
phi = np.linspace(0, 2*np.pi, 100) 

for i in range(len(theta)):
    x = np.cos(theta[i]) * np.cos(phi)
    y = np.cos(theta[i]) * np.sin(phi)
    z = np.sin(theta[i])

    ax.plot(x,y,z, color=(0, 0, 0, 0.1))


ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')

ax.view_init(30, 30)
#------------------------------------------------------------------------------
ax = fig.add_subplot(2, 2, 2, projection='3d')

ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors)


# Dark wires
theta = np.linspace(0, np.pi/2, 100)
    
phi = np.linspace(0, 4*np.pi/3, 3) 

for i in range(len(phi)):
    x = np.cos(theta) * np.sin(phi[i])
    y = np.cos(theta) * np.cos(phi[i])
    z = np.sin(theta)

    ax.plot(x,y,z, color=(0, 0, 0, 0.9))

# Light horizontal wires
theta = np.linspace(0, 2*np.pi, 100)
    
phi = np.linspace(0, 2*np.pi, 10) 

for i in range(len(phi)):
    x = np.cos(theta) * np.cos(phi[i])
    y = np.cos(theta) * np.sin(phi[i])
    z = np.sin(theta)

    ax.plot(x,y,z, color=(0, 0, 0, 0.1))
    
# Light vertical wires
theta = np.linspace(0, 2*np.pi, 10)
    
phi = np.linspace(0, 2*np.pi, 100) 

for i in range(len(theta)):
    x = np.cos(theta[i]) * np.cos(phi)
    y = np.cos(theta[i]) * np.sin(phi)
    z = np.sin(theta[i])

    ax.plot(x,y,z, color=(0, 0, 0, 0.1))


ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlabel('')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(0, 0)
#------------------------------------------------------------------------------
ax = fig.add_subplot(2, 2, 3, projection='3d')

ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors)


# Dark wires
theta = np.linspace(0, np.pi/2, 100)
    
phi = np.linspace(0, 4*np.pi/3, 3) 

for i in range(len(phi)):
    x = np.cos(theta) * np.sin(phi[i])
    y = np.cos(theta) * np.cos(phi[i])
    z = np.sin(theta)

    ax.plot(x,y,z, color=(0, 0, 0, 0.9))

# Light horizontal wires
theta = np.linspace(0, 2*np.pi, 100)
    
phi = np.linspace(0, 2*np.pi, 10) 

for i in range(len(phi)):
    x = np.cos(theta) * np.cos(phi[i])
    y = np.cos(theta) * np.sin(phi[i])
    z = np.sin(theta)

    ax.plot(x,y,z, color=(0, 0, 0, 0.1))
    
# Light vertical wires
theta = np.linspace(0, 2*np.pi, 10)
    
phi = np.linspace(0, 2*np.pi, 100) 

for i in range(len(theta)):
    x = np.cos(theta[i]) * np.cos(phi)
    y = np.cos(theta[i]) * np.sin(phi)
    z = np.sin(theta[i])

    ax.plot(x,y,z, color=(0, 0, 0, 0.1))


ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlabel('x')
ax.set_ylabel('')
ax.set_zlabel('z')

ax.view_init(0, 90)
#------------------------------------------------------------------------------
ax = fig.add_subplot(2, 2, 4, projection='3d')

ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors)


# Dark wires
theta = np.linspace(0, np.pi/2, 100)
    
phi = np.linspace(0, 4*np.pi/3, 3) 

for i in range(len(phi)):
    x = np.cos(theta) * np.sin(phi[i])
    y = np.cos(theta) * np.cos(phi[i])
    z = np.sin(theta)

    ax.plot(x,y,z, color=(0, 0, 0, 0.9))

# Light horizontal wires
theta = np.linspace(0, 2*np.pi, 100)
    
phi = np.linspace(0, 2*np.pi, 10) 

for i in range(len(phi)):
    x = np.cos(theta) * np.cos(phi[i])
    y = np.cos(theta) * np.sin(phi[i])
    z = np.sin(theta)

    ax.plot(x,y,z, color=(0, 0, 0, 0.1))
    
# Light vertical wires
theta = np.linspace(0, 2*np.pi, 10)
    
phi = np.linspace(0, 2*np.pi, 100) 

for i in range(len(theta)):
    x = np.cos(theta[i]) * np.cos(phi)
    y = np.cos(theta[i]) * np.sin(phi)
    z = np.sin(theta[i])

    ax.plot(x,y,z, color=(0, 0, 0, 0.1))


ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('')

ax.view_init(90, 0)
#------------------------------------------------------------------------------
#plt.savefig('s1_embedding.eps', format='eps')

plt.savefig('quotient_embedding.png', format='png', bbox_inches='tight')

plt.close()

# Movie
sys.exit()

os.system('mkdir movie')
os.system('rm -r movie/*')

for i in range(61):
    w, h = plt.figaspect(1)
    fig = plt.figure(figsize=(w,h))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors)

    ax.plot_wireframe(wire_torus[0], wire_torus[1], wire_torus[2], rstride=10, cstride=10, color=(0, 0, 0, 0.5))

    ax.view_init(30, i*6)

    plt.savefig('movie/{}.png'.format(i), format='png', bbox_inches='tight')
    plt.close()


import os
os.system('mkdir movie')
os.system("ffmpeg -f image2 -r 6 -i ./movie/%01d.png -vcodec mpeg4 -y ./movie/quotient.mp4")

os.system("ffmpeg -f image2 -r 6 -i ./movie/%01d.png -vcodec gif -y ./movie/quotient.gif")