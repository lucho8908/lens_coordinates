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

#------------------------------------------------------------------------------
# Plot data set X
#------------------------------------------------------------------------------

fig = plt.figure()

arg_z = np.angle(X_solid_torus[:,0])
arg_w = np.angle(X_solid_torus[:,1])
norm_w = np.abs(X_solid_torus[:,1])

arg_z = np.mod(arg_z, 2*np.pi)
arg_z_q = np.mod(arg_z, 2*np.pi/q)
k = np.floor((arg_z - arg_z_q) / (2*np.pi/q))

arg_w = arg_w - 2*k*np.pi/q

x_coor = norm_w*np.cos(arg_w)
y_coor = norm_w*np.sin(arg_w)
z_coor = (q/np.pi)*(arg_z_q - np.pi/q)*np.sqrt( np.maximum(1-norm_w**2, 0) )

#c = np.sqrt(x_coor**2 + y_coor**2 + z_coor**2)
c = x_coor

cmap = matplotlib.cm.get_cmap('hsv')
normalize = matplotlib.colors.Normalize(vmin=min(c), vmax=max(c))
colors = np.array([cmap(normalize(value)) for value in c])

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(x_coor, y_coor, z_coor, color=colors, s=5)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.savefig('Lq.png')
plt.close()
#------------------------------------------------------------------------------

t0 = time.time()
dm_X = lens_distance(X,X)
t1 = time.time()
print('Distance matrix = {} s'.format(t1-t0))

#------------------------------------------------------------------------------
# Homology verification
#------------------------------------------------------------------------------
cohomology_test = lpca.minmax_subsample_distance_matrix(dm_X, 1000)['indices']

os.system('mkdir lens_homolgy_lpca')
os.system('rm -r lens_homolgy_lpca/*')

test_dm = lens_distance(X[cohomology_test, :], X[cohomology_test, :])

plot_dgms(ripser( test_dm , coeff=3, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH_i(X, \mathbb{Z}_3)$')
plt.savefig('./lens_homolgy_lpca/lens_homology_original_p_3.png', format='png')
plt.close()

plot_dgms(ripser( test_dm , coeff=2, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH_i(X, \mathbb{Z}_2)$')
plt.savefig('./lens_homolgy_lpca/lens_homology_original_p_2.png', format='png')
plt.close()

#------------------------------------------------------------------------------

num_points_sample = 190 # Number of pointsin the min-max subsample

# solid_torus = gen_s1_d2(num_points_sample, bias=True)

# L = np.array([np.sqrt(1 - np.abs(solid_torus[:,1])**2)*solid_torus[:,0], solid_torus[:,1]]).transpose()

# L = np.array([np.real(L[:,0]), np.imag(L[:,0]), np.real(L[:,1]), np.imag(L[:,1])]).transpose()

t0 = time.time()
subsam = lpca.minmax_subsample_distance_matrix(dm_X, num_points_sample, seed=seed_indices) # Subsample L\subset X
t1 = time.time()
print('Maxmin subsampl = {} s'.format(t1-t0))

L = X[subsam['indices'], :]

# ind_L = subsam['indices']
# dist_to_L = subsam['distance_to_L']

# cover_r = max(dist_to_L)

#------------------------------------------------------------------------------

p0 = L[:,0] + L[:,1]*1j
p1 = L[:,2] + L[:,3]*1j

arg_z = np.angle(p0)
arg_w = np.angle(p1)
norm_w = np.abs(p1)

arg_z = np.mod(arg_z, 2*np.pi)
arg_z_q = np.mod(arg_z, 2*np.pi/q)
k = np.floor((arg_z - arg_z_q) / (2*np.pi/q))

arg_w = arg_w - 2*k*np.pi/q

x_coor_L = norm_w*np.cos(arg_w)
y_coor_L = norm_w*np.sin(arg_w)
z_coor_L = (q/np.pi)*(arg_z_q - np.pi/q)*np.sqrt( np.maximum(1-norm_w**2, 0) )


fig, ax = plt.subplots()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(x_coor_L, y_coor_L, z_coor_L, color='red')
plt.savefig('Lq_landmarks.png')
plt.close()
#------------------------------------------------------------------------------

dm_X =lens_distance(L, X)

dm_L =lens_distance(L, L) # Distace matrix for L

t0 = time.time()
result = ripser(dm_L, coeff=q, do_cocycles=True, maxdim=1, distance_matrix=True) # Persistent cohomology
t1 = time.time()
print('Persistent homology = {} s'.format(t1-t0))

diagrams = result['dgms'] # Persisten diagrams
cocycles = result['cocycles'] 
D = result['dm'] # Distance matrix used by Ripser to compute persistence

H_1 = cocycles[1] # PH^1(R(L); Z_q)
H_1_diagram = diagrams[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)

#------------------------------------------------------------------------------
# Plot persitent homology of X: H_*(L; Z_q)
#------------------------------------------------------------------------------
plot_dgms(result['dgms'], xy_range=[-0.05,0.8,-0.05,0.8])
plt.title(r'$PH^1(L, \mathbb{Z}_3)$' )
plt.savefig('Lq_homoloy.png')
plt.close()

plot_dgms( ripser(dm_L, coeff=2, maxdim=1, distance_matrix=True)['dgms'], xy_range=[-0.05,0.8,-0.05,0.8])
plt.title(r'$PH^1(L, \mathbb{Z}_2)$' )
plt.savefig('Lq_homoloy_2.png')
plt.close()
#------------------------------------------------------------------------------

a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

eta = H_1[H_1_persistence_sort_ind[-1]] # Cochain representtive of the largest class in PH^1(R(L); Z_q)

#epsilon = (a + (b - a)*0.9)/2.25 # Epsilon is the radius e usefor the balls with centers in the landmarks.
print(a, b)
epsilon = b*(1/2)
print(epsilon)
# We nee to verify PH^1(R(L); Z_q) has a class with perssitence long enough
if not(a< epsilon):
    print('{}WARNING: The largest class (a,b) in PH^1(R(L); Z_q) is not long enough: 2a is NOT smaller than b.{}'.format('\033[33m', '\033[0m'))

dist_to_L = np.min(dm_X, axis=0)
cover_r = max(dist_to_L)

# if cover_r > epsilon:
#     print('{}WARNING: Covering radius is larger than epsilon. Some points in X will be ignored.{}'.format('\033[33m', '\033[0m'))
#     points_covered = dist_to_L < epsilon
#     X = X[points_covered, :]
#     dm_X = dm_X[:, points_covered]
#     print('{}New data array shape = {}{}'.format('\033[33m', dm_X.shape ,'\033[0m'))
#     colors = colors[points_covered]

is_in_landmark_ball = np.maximum( np.subtract(epsilon, dm_X),0 )
is_in_any_ball = np.logical_not(np.isclose(np.sum(is_in_landmark_ball, axis=0), 0))
points_covered = is_in_any_ball
X = X[points_covered, :]
x_coor = x_coor[points_covered]
y_coor = y_coor[points_covered]
z_coor = z_coor[points_covered]
dm_X = dm_X[:, points_covered]
print('{}New data array shape = {}{}'.format('\033[33m', dm_X.shape ,'\033[0m'))

colors = colors[points_covered]

z_q = np.exp((2*math.pi*1j)/q) # z_q is a root of unity != 1

t0 = time.time()
lens = lpca.lens_coordinates(eta, dm_X, epsilon, z_q)
t1 = time.time()
print('Lens coordinates = {} s'.format(t1-t0))

#------------------------------------------------------------------------------
# Homology verification
#------------------------------------------------------------------------------
cohomology_test = []
for row in np.transpose(lens):
    cohomology_test.append(np.array([np.real(row), np.imag(row)]))
cohomology_test = np.array(cohomology_test).reshape((-1,dm_X.shape[1]))
cohomology_test = np.transpose(cohomology_test)

lens_dm = lens_distance(cohomology_test, cohomology_test)

sub_ind = lpca.minmax_subsample_distance_matrix(lens_dm, 1000)['indices']

plot_dgms(ripser( lens_dm[sub_ind, :][:, sub_ind] , coeff=3, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH^1(f(X), \mathbb{Z}_3)$' )
plt.savefig('./lens_homolgy_lpca/homology_lpca_iteration_{}_p_3.png'.format(0), format='png')
plt.close()

plot_dgms(ripser( lens_dm[sub_ind, :][:, sub_ind] , coeff=2, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH^1(f(X), \mathbb{Z}_2)$' )
plt.savefig('./lens_homolgy_lpca/homology_lpca_iteration_{}_p_2.png'.format(0), format='png')
plt.close()
#------------------------------------------------------------------------------

XX = np.transpose(lens) # XX matrix in C^(landmark x num_points)

XX = XX / (np.ones((len(XX), 1))*np.sqrt(np.real(np.diag(np.transpose(np.conj(XX))@XX)))) # we normalize XX 

variance = []
run_times = []

tolerance = 0.03 # User parameter used to set up the first projection

#tolerance = 1e-7 # User parameter used to set up the first projection

t0 = time.time()

svd_t0 = time.time()
U, s, V = np.linalg.svd(XX, full_matrices=True) # compute the full SVD of XX, XX = USV*
svd_t1 = time.time()
run_times.append(svd_t1 - svd_t0)

v_0 = lpca.sqr_ditance_projection(U[:, 0:1], XX)
v_1 = 0
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
# Homology verification
#------------------------------------------------------------------------------
cohomology_test = []
for row in XX:
    cohomology_test.append(np.array([np.real(row), np.imag(row)]))
cohomology_test = np.array(cohomology_test).reshape((-1,dm_X.shape[1]))
cohomology_test = np.transpose(cohomology_test)

lens_dm = lens_distance(cohomology_test, cohomology_test)

sub_ind = lpca.minmax_subsample_distance_matrix(lens_dm, 1000)['indices']

plot_dgms(ripser( lens_dm[sub_ind, :][:, sub_ind] , coeff=3, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH^1(f(X)\subset \mathbb{C}^{ %d }, \mathbb{Z}_3)$' % XX.shape[0] )
plt.savefig('./lens_homolgy_lpca/homology_lpca_iteration_{}_p_3.png'.format(1), format='png')
plt.close()

plot_dgms(ripser( lens_dm[sub_ind, :][:, sub_ind] , coeff=2, maxdim=1, distance_matrix=True)['dgms'])
plt.title(r'$PH^1(f(X)\subset \mathbb{C}^{ %d }, \mathbb{Z}_2)$' % XX.shape[0] )
plt.savefig('./lens_homolgy_lpca/homology_lpca_iteration_{}_p_2.png'.format(1), format='png')
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

        lens_dm = lens_distance(cohomology_test, cohomology_test)

        diagrams = ripser(lens_dm[sub_ind, :][:, sub_ind], coeff=3, maxdim=1, distance_matrix=True)['dgms']

        H_1_diagram = diagrams[1] # dgm(PH^1(R(L); Z_q))
        H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
        H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)
        a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
        b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

        larg_per = b-a
        second_larg_per = H_1_diagram[H_1_persistence_sort_ind[-2], 1] - H_1_diagram[H_1_persistence_sort_ind[-2], 0]

        os.system('echo LPCA >> lens_iso/largest_persistence.txt')

        os.system('echo S^1, q=3, {}, {} >> lens_iso/largest_persistence.txt'.format(larg_per, second_larg_per))

        plot_dgms(diagrams, xy_range=[-0.05,0.8,-0.05,0.8])
        plt.title(r'$PH^1(f(X)\subset \mathbb{C}^{ %d }, \mathbb{Z}_3)$' % XX.shape[0] )
        plt.savefig('./lens_homolgy_lpca/lens_homology_lpca_iteration_{}_p_3.png'.format(i), format='png')
        plt.close()

        diagrams = ripser(lens_dm[sub_ind, :][:, sub_ind], coeff=2, maxdim=1, distance_matrix=True)['dgms']

        H_1_diagram = diagrams[1] # dgm(PH^1(R(L); Z_q))
        H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
        H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)
        a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
        b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

        larg_per = b-a
        second_larg_per = H_1_diagram[H_1_persistence_sort_ind[-2], 1] - H_1_diagram[H_1_persistence_sort_ind[-2], 0]

        os.system('echo S^1, q=2, {}, {} >> lens_iso/largest_persistence.txt'.format(larg_per, second_larg_per))

        plot_dgms(diagrams, xy_range=[-0.05,0.8,-0.05,0.8])
        plt.title(r'$PH^1(f(X)\subset \mathbb{C}^{ %d }, \mathbb{Z}_2)$' % XX.shape[0] )
        plt.savefig('./lens_homolgy_lpca/lens_homology_lpca_iteration_{}_p_2.png'.format(i), format='png')
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

val_smallest = vals[1]
vec_smallest = vecs[:,1]

rotation_matrix = lpca.rotM(vec_smallest)

variance.append( lpca.sqr_ditance_orthogonal_projection(vec_smallest.reshape(2, 1), XX) )

Y = rotation_matrix@XX

Y = np.delete(Y, (-1), axis=0)

XX = Y / (np.ones((len(Y), 1))*np.sqrt(np.real(np.diag(np.transpose(np.conj(Y))@Y))))
t1 = time.time()
print('LPCA = {} s'.format(t1-t0))



theta = np.angle(XX)

plt.scatter(np.cos(theta), np.sin(theta), color=colors)
plt.savefig('Lq_in_s1.png')
plt.close()

theta = 3*theta
plt.scatter(np.cos(theta), np.sin(theta), color=colors)
plt.savefig('Lq_in_s1_2.png')
plt.close()

variance_s1 = lpca.variance_circle(theta[0])['geodvar']

variance = [variance_s1] + variance[::-1]
print(variance)


variance = np.cumsum(variance)

variance = variance/variance[-1]
print(variance)
x_axis = list(range(1,k_break+1)) + [len(U)]
print(x_axis)
fig, ax = plt.subplots()

fig = plt.plot(x_axis, variance, 'o-')

ax.axes.axhline(y=0.7, linestyle='--', color='rebeccapurple')

plt.xticks(np.array(x_axis)[[0,3,5,8,12,15,18,len(x_axis)-1]])

plt.xlabel('Lens embedding dimension')

plt.ylabel('Percentage of cumulative variance')

#ax.set_xticklabels([r'$S^{%s}$' % i for i in range(1,2*num_points_sample-1 +1,2)])

plt.ylim(0,1.01)

#plt.yticks(variance)

plt.savefig('Lq_explained_variance.png')

plt.close()

#------------------------------------------------------------------------------
def gen_wire_positive(N, q):
    #theta, phi = np.linspace(0, 2*np.pi/q, N), np.linspace(0, 2*np.pi/q, N)
    theta, phi = np.linspace(0, 2*np.pi, N), np.linspace(0, 2*np.pi, N)
    THETA, PHI = np.meshgrid(theta, phi)
    
    x = np.cos(THETA) * np.cos(PHI)
    y = np.cos(THETA) * np.sin(PHI)
    z = np.sin(THETA)

    return [x, y, z]

wire_torus = gen_wire_positive(60, q)

print(np.mean(np.abs(embedding_S3), axis=0))
print(np.var(np.abs(embedding_S3), axis=0))

cloud_R3 = []
for p in embedding_S3:
    #print(np.vdot(p,p))
    p1 = p[0]
    p2 = p[1]
    #print(np.vdot(p1,p1), np.vdot(p2,p2))
    arg_z = np.mod(np.angle(p1), 2*np.pi)

    theta = np.mod(arg_z, 2*np.pi/q)

    k = np.floor((arg_z - theta) / (2*np.pi/q))  

    phi = np.mod(np.angle(p2), 2*np.pi)
    r = np.abs(p2)

    phi = phi - 2*k*np.pi/q

    # x = (1 + r*np.cos(phi))*np.cos(theta)
    # y = (1 + r*np.cos(phi))*np.sin(theta)
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    #z = r*np.sin(phi)
    z = (q/np.pi)*(theta - np.pi/q)*np.sqrt(1-r**2)

    cloud_R3.append([x,y,z])

cloud_R3 = np.array(cloud_R3)

w, h = plt.figaspect(0.5)
fig = plt.figure(figsize=(w,h))
#------------------------------------------------------------------------------
ax = fig.add_subplot(2, 2, 1, projection='3d')

ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors, s=5)


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

ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors, s=5)


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

ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors, s=5)


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

ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors, s=5)


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

plt.savefig('Lq_embedding.png', format='png', bbox_inches='tight')

plt.close()

#------------------------------------------------------------------------------
# Matrix plot
#------------------------------------------------------------------------------
MP_original = [x_coor, y_coor, z_coor]
MP_lens = [cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2]]

fig = plt.figure()

k = 1
for i in range(3):
    for j in range(3):
        ax = fig.add_subplot(3, 3, k)
        ax.scatter(MP_original[j], MP_lens[i], color=colors)
        k += 1
plt.savefig('Lq_matrix_plot.png', format='png')

plt.close()

sys.exit()

# Movie
os.system('mkdir lens_movie')

for i in range(61):
    w, h = plt.figaspect(1)
    fig = plt.figure(figsize=(w,h))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(cloud_R3[:,0], cloud_R3[:,1], cloud_R3[:,2], color=colors, s=5)

    ax.plot_wireframe(wire_torus[0], wire_torus[1], wire_torus[2], rstride=10, cstride=10, color=(0, 0, 0, 0.5))

    ax.view_init(30, i*6)

    plt.savefig('lens_movie/{}.png'.format(i), format='png')
    plt.close()

os.system("ffmpeg -f image2 -r 6 -i ./lens_movie/%01d.png -vcodec mpeg4 -y ./lens_movie/lens.mp4")

os.system("ffmpeg -f image2 -r 6 -i ./lens_movie/%01d.png -vcodec gif -y ./lens_movie/lens.gif")
