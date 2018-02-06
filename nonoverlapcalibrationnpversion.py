""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np
import TrackSimulate as ts
from pyquaternion import Quaternion

from joblib import Parallel, delayed


numFrames = 128


def Tql(q):
    w, x, y, z = q
    Tq = np.zeros((4,4))
    Tq [0,0] = w
    Tq [0,1] = -x
    Tq [0,2] = -y
    Tq [0,3] = -z
    Tq [1,0] = x
    Tq [1,1] = w
    Tq [1,2] = -z
    Tq [1,3] = y
    Tq [2,0] = y
    Tq [2,1] = z
    Tq [2,2] = w
    Tq [2,3] = -x
    Tq [3,0] = z
    Tq [3,1] = -y
    Tq [3,2] = x
    Tq [3,3] = w
    return Tq

def Tqr(q):
    w, x, y, z = q
    Tq = np.zeros((4,4))
    Tq [0,0] = w
    Tq [0,1] = -x
    Tq [0,2] = -y
    Tq [0,3] = -z
    Tq [1,0] = x
    Tq [1,1] = w
    Tq [1,2] = z
    Tq [1,3] = -y
    Tq [2,0] = y
    Tq [2,1] = -z
    Tq [2,2] = w
    Tq [2,3] = x
    Tq [3,0] = z
    Tq [3,1] = y
    Tq [3,2] = -x
    Tq [3,3] = w
    return Tq

# nonoverlapping method "Calibration of a Multi-Camera Rig From Non-Overlapping Views"
# init variables
Trajectory_size = 1
scalegt = 1
#scalegt = np.random.rand(1)[0]
Xinput, Yinput = ts.gen_inputdata_v4(Trajectory_size, numFrames, scalegt)
Xinput = np.float32(Xinput)
Yinput = np.float32(Yinput)



# estimate rig rotation
# T_m for master camera, T_s for slave camera
# T_i for i-th frame, therefore T_mi for master camera's i-th frame
# T_delta for transformation between master and slave camera

# so for rotation we have
# T^si_m0 = T^mi_m0 T^si_mi = T^s0_m0 T^si_s0
# where T_delta = T^si_mi = T^s0_m0

from numpy import linalg as LA

qdelta = np.zeros(4)
qqd = Quaternion(Yinput[0, 0:4])
#for i in range(0, Xinput[0].shape[0] - 8, 8):
singleframesize = (4+3)*2
A = [None]*numFrames
cnt = 0
for i in range(0, singleframesize*numFrames , singleframesize):

    qm = Xinput[0, i:i+4]
    qs = Xinput[0, i+7:i+11]

#    qqm = Quaternion(qm)
#    qqs = Quaternion(qs)
#    rm = qqm.rotation_matrix
#    rs = qqs.rotation_matrix
#    rd = qqd.rotation_matrix
    
    #euqation 11 in the paper
    A[cnt] = (Tql(qm)-Tqr(qs))    
    cnt += 1

    
#shape change of A matrix
# (A - lambda*I)*x =0
newA = np.array(A).reshape(-1, 4)
w, v = LA.eig( np.dot( newA.transpose(), newA) )
v = v.transpose()

index_min = np.argmin(abs(w))


print( qqd, " Ground turth " )
print(  Quaternion(v[index_min]), " Result by eigen vector ")


U, s, V = np.linalg.svd(newA, full_matrices=True)
index_min = np.argmin(abs(s))
deltaR = V[index_min]
print(  Quaternion(deltaR), " Result by SVD ")

import matplotlib.pyplot as plt

Ustack, Sstack, Vstackt = np.linalg.svd(newA)
Sstack = np.diag(Sstack)
Vstack = Vstackt.transpose()


def incrementalSVD2(S, V, A):
    """
    Given S and V of SVD, update S and V for the new rows A.
    all matrices should be 4x4.
    
    input
    S: diagonal matrix with singular values
    V: right singular matrix (not V^T)
    return
    S: updated S
    V: updated V
    """
    tmp = np.vstack((S, A @ V))
    _, S1, V1t = np.linalg.svd(tmp)
    V1 = V1t.transpose()
    S = np.diag(S1) # update S
    V = V @ V1 # update V

    return S, V

_, S, Vt = np.linalg.svd(A[0]) # initialize
S = np.diag(S)
V = Vt.transpose()

diff = []
for i in range(1, len(A)):
    S, V = incrementalSVD2(S, V, A[i])
    
    diff.append(min(np.linalg.norm(V[:, -1] - Vstack[:, -1]),
                    np.linalg.norm(V[:, -1] + Vstack[:, -1])))
    
print("solution: ", V[:, -1]) # min singular value vector 
plt.plot(diff)
plt.show()

# estimete rig translation (when the scale issue exists)
# [(I-R^masteri_master0), (deltaR*C^slavei_slave0)] * [deltaC, deltaS]^t = C^masteri_master0


print( "****************************")
B = [None]*numFrames
Cm = [None]*numFrames
cnt = 0
for i in range(0, singleframesize*numFrames , singleframesize):

    qm = Xinput[0, i:i+4]
#    qs = Xinput[0, i+7:i+11]
    tm = Xinput[0, i+4:i+7]
    ts = Xinput[0, i+11:i+14]
    
    qqm = Quaternion(qm)
#    qqs = Quaternion(qs)
    rm = qqm.rotation_matrix
#    rs = qqs.rotation_matrix
    rd = Quaternion(deltaR).rotation_matrix
    
    #euqation 12 in the paper
    B[cnt] = np.c_[ (np.eye(3)-rm), (np.dot(rd,ts)) ]
    Cm[cnt] = tm
    
    cnt += 1
    
print( Yinput[0][4:7], scalegt, "Translation&scale Groundtruth")

newB = np.array(B).reshape(-1, 4)
newCm = np.array(Cm).reshape(-1,)
x, res, rak, s = np.linalg.lstsq(newB, newCm)

print(x, "Result by np.linalg.lstsq")


xstack, _, _, _ = np.linalg.lstsq(newB, newCm)
def incrementalLSQ(x, C, A, b):
    """
    input
    x: initial solution, x_n-1
    C: param C_n-1
    A: A_i
    b: b_i
    return
    x: updated solution x_n
    C: updated C_n
    """
    
    Cn = C + A.transpose() @ A
    Cn_inv = np.linalg.inv(Cn)
    
    xn = x + Cn_inv @ A.transpose() @ (b - A @ x)

    return xn, Cn


x, _, _, _ = np.linalg.lstsq(B[0], Cm[0]) # initialize
C = A[0].transpose() @ A[0]

diff = []
for i in range(1, len(A)):
    x, C = incrementalLSQ(x, C, B[i], Cm[i])
    
    diff.append(np.linalg.norm(x - xstack) / np.linalg.norm(xstack))

axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([0,20])

plt.plot(diff)
plt.show()
print("check: ", np.linalg.norm(newB @ x - newCm))








