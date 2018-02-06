#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:52:54 2017

@author: ogawa
"""
#import matplotlib.pylab as plt3d
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import math
from pyquaternion import Quaternion

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrixold(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def eulerAnglesToRotationMatrix(theta) :
    
    ai, aj, ak = theta
    sx, sy, sz = math.sin(ai), math.sin(aj), math.sin(ak)
    cx, cy, cz = math.cos(ai), math.cos(aj), math.cos(ak)
    
    R = np.identity(3)
    
    R[0,0] = +cy*cz
    R[0,1] = -cy*sz
    R[0,2] = +sy
    
    R[1,0] = +sx*sy*cz+cx*sz
    R[1,1] = -sx*sy*sz+cx*cz
    R[1,2] = -sx*cy
    
    R[2,0] = -cx*sy*cz+sx*sz
    R[2,1] = +cx*sy*sz+sx*cz
    R[2,2] = +cx*cy
    
 
    return R


def MakeTrajectory3dnew(D, B, numFrames):
    patternid = np.random.randint(4, size=(4))
    A, b, Q, R = D[patternid[0]]
    mu_s, phi_s, mu_e, phi_e = B[patternid[1]]
    
    track = []
    track.append(np.random.multivariate_normal(mu_s, phi_s, 1)[0].tolist())

    
#    cnt = 0
    r = 1
    theta = 0
    unitdeg = 3.1415 * 1/360
    
    #deg = [unitdeg*2, -unitdeg*5, unitdeg*8, -unitdeg*10]
    #deg = [2, -5, 8, -10]
    deg = 3 * np.random.randn(1, 4) + 2
    deg = deg * unitdeg
    #rad = [1, 3, 5, 7]
    rad = 1.5 * np.random.randn(1, 4) + 3.5
#    while True:
    noise = np.random.multivariate_normal([0, 0, 0], Q, size = numFrames)
    for cnt in range(0, (numFrames-1)):
        theta += deg[0][patternid[2]]
        r = rad[0][patternid[3]]
        c = [math.cos(theta)*r, math.sin(theta)*r, 0]
        
        A, b, Q, R = D[patternid[patternid[0]]]
        mu_s, phi_s, mu_e, phi_e = B[patternid[patternid[1]]]
#        track.append((A.dot(np.array(track[-1])) + b + c + np.random.multivariate_normal([0, 0, 0], Q, 1)[0]).tolist())       #system model
        track.append((A.dot(np.array(track[-1])) + b + c).tolist())       #system model

#        cnt += 1
#        if cnt > 198:
#            break
        if cnt % 32 ==0:
            patternid = np.random.randint(4, size=(4))
            continue
    
    track = np.array(track) + noise
    
    return track

def MakeRotation3d(D, numFrames):
    
    patternid = np.random.randint(4, size=(4))
    
    unitdeg = 3.1415926 * 1 / 360
#    deg = [1, -5, 7, -3]
    deg = 2.0 * np.random.randn(1, 4) + 5
    deg = deg*unitdeg
    
    mu_s    = np.array([0, 0, 0])
    phi_s = np.array([[3*unitdeg,0,0],
                       [0,3*unitdeg,0],
                       [0,0,3*unitdeg]]) 
    
    rotation = []
    rot = []
    
    rot.append(np.random.multivariate_normal(mu_s, phi_s, 1)[0].tolist())
    
#    rotation.append(eulerAnglesToRotationMatrix(rot[-1]).tolist())
    
#    cnt = 0
#    while True:
    R = np.array([[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]])
    noise = np.random.multivariate_normal([0, 0, 0], R, numFrames)
    for cnt in range(0, (numFrames-1)):
        
        A, b, Q, R = D[patternid[0]]
        
        rot.append((np.array(rot[-1]) + b*deg[0][patternid[1]]).tolist())

        if cnt % 32 ==0:
            patternid = np.random.randint(4, size=(4))
            continue
        
    rot = np.array(rot) + noise
#    rot = np.array(rot)

        
#    for cnt in range(0, 199):
#        rotation.append(eulerAnglesToRotationMatrix(rot[cnt]).tolist())
    rotation = list(map(eulerAnglesToRotationMatrix, rot))

    
    rotation = np.array(rotation)

    return rotation


def MakeTraRot3d(D, B, E, K, numFrames):
    trajectories = []
    for k in range(K):

        track = MakeTrajectory3dnew(D, B, numFrames)
        rotation = MakeRotation3d(E, numFrames)
        
        if len(track) != 0:
            trajectories.append([track, rotation])
        
    return trajectories


def GenTrajectory(num, numFrames = 200):
#    Dynamics
    Q = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])


    A1 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    b1 = np.array([0., -1, 0])
    D1 = [A1, b1, Q, R]
    
    
    A2 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    b2 = np.array([1., 0, 0])
    D2 = [A2, b2, Q, R]
    
    
    A3 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    b3 = np.array([1., -1, 0])
    D3 = [A3, b3, Q, R]
    
    
    A4 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    b4 = np.array([-1., 1, 0])
    D4 = [A4, b4, Q, R]
    
    D = [D1, D2, D3, D4]
    
    R1 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]])
    r1 = np.array([0, 0, 1])
    E1 = [R1, r1, Q, R]
    
    
    R2 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    r2 = np.array([1, 0, -1])
    E2 = [R2, r2, Q, R]
    
    
    R3 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    r3 = np.array([0, 1, -1])
    E3 = [R3, r3, Q, R]
    
    
    R4 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    r4 = np.array([1, 1, 1])
    E4 = [R4, r4, Q, R]
    
    E = [E1, E2, E3, E4]
    
    
#    Beliefs
    mu_s1    = np.array([0, 0, 0])
    phi_s1 = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])  
    
    mu_s2    = np.array([0, 0, 0])
    phi_s2 = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])  
    
    mu_e1    = np.array([0, 0, 0])
    phi_e1 = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])   
    
    mu_e2    = np.array([0, 0,0])
    phi_e2 = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])
    
    B1 = [mu_s1, phi_s1, mu_e1, phi_e1]
    B2 = [mu_s2, phi_s2, mu_e2, phi_e2]
    B3 = [mu_s1, phi_s1, mu_e2, phi_e2]
    B4 = [mu_s2, phi_s2, mu_e1, phi_e1]
    
    B = [B1, B2, B3, B4]
            
    pi = np.random.random(4)
    pi /= pi.sum()
#    print("pi : ", end="")
#    print(pi)
    
    K = num

    trajectories = MakeTraRot3d(D, B, E, K, numFrames)
    
    return trajectories

#
#code for gen data
#
def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def aa_to_rotation(theta, phi, z):
    deflection = 1
    theta = theta * 2.0*deflection*np.pi # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def rand_translation_vector(low = 0, high = 1):
    
    t = np.random.uniform(low, high, 3)
    return t

def rt_to_homogeneous(R, t):
    H = np.eye(4)
    #rotation matrix R and translation vector t    
    H[0:3,  0:3] = R
    # rows  cols
    H[0:3,  3] = t
    return H
    
def gen_cameraset():
    #transformation to rig from first camera
    R = rand_rotation_matrix()    
    t = rand_translation_vector()
    H1 = rt_to_homogeneous(R, t)

    #transformation to rig from second camera
    R = rand_rotation_matrix()    
    t = rand_translation_vector()
    H2 = rt_to_homogeneous(R, t)

    #transformation to first from second camera
#    H3 = np.matrix(H1) * np.matrix(np.linalg.inv(np.matrix(H2)))
    H3 = H1.dot(np.linalg.inv(H2))
    
#    return np.matrix(H1), np.matrix(H2), H3
    return H1, H2, H3

######################################
#
# output the motion under each camera's cooridinate system's position
#
def gen_inputdata_v4(tranum = 1, numFrames = 128, scale = 1):

    #output is camera rig's position under WCS
    tra = GenTrajectory(tranum, numFrames)
    Y = np.zeros((tranum, 7))
    
    invec = np.zeros((tranum, 2 * numFrames * 7))
    
    
    for i in range(tranum):
        A, C = tra[i]

        inputmaster = np.zeros((numFrames, 7))
        inputslave = np.zeros((numFrames, 7))
        
        
        # gen different camera setting for each trajectory
        # H1 for rig to master camera
        # H2 for rig to slave camera
        # H3 for slave to master camera
        H1, H2, H3 = gen_cameraset()
        invH1 = np.linalg.inv(H1)
        #invH2 = np.linalg.inv(H2)
        qr = Quaternion(matrix=H3) #[0:3,0:3])
        qre = qr.elements
        groundtruth = np.hstack((qre, H3[0:3,3]))
#        groundtruth = qre
        Y[i] = groundtruth
        
        #gen scale diff
        scalemaster = scale
        scaleslave = 1
#        print(scalemaster, "scale")


        # T^si_m0 = T^s0_m0 T^si_s0 = T^mi_m0 T^si_mi
        # T^s0_m0 = T^mi_m0 = H3
        firstMaster = np.eye(4) #wcs to m0cs
        firstSlave = np.eye(4) #wcs to s0cs
        #firstRig = np.eye(4) #wcs to r0cs
        
        text_file = open("/home/zhao/Desktop/TrackVis/build/myfile0.txt", "w")
        #first master to rig, then slave to master, then rig to world(s)
        #all in homogeneous matrix, flatten to one row
        #text_file.write( np.array2string(H1.flatten(), max_line_width=999) + "\n") #invH1.flatten().tostring() )
        text_file.write( ' '.join(map(str, H1.flatten())) + "\n")
        text_file.write( ' '.join(map(str, H3.flatten())) + "\n") #H3.flatten().tostring() )
        
        
        for j in range(numFrames):
            R = C[j] # .reshape(3,3)
            t = A[j]
            
            # Trig rig-ith to world
            Trig = rt_to_homogeneous(R,t)
            text_file.write( ' '.join(map(str, Trig.flatten())) + "\n")
            # world to rig-ith
            Trig = np.linalg.inv(Trig)
            # 
#            Trig = np.dot(np.linalg.inv(firstRig), Trig)
            
            
            # transform from wcs to m(aster)cs-ith and scs-ith
            Tmaster = np.dot(H1, Trig)
            Tslave = np.dot(H2, Trig)
            
            if ( j == 0 ):
                # world to master 0-th
                firstMaster = Tmaster
                firstSlave = Tslave
                #firstRig = Trig
            
            qrm = Quaternion(matrix=np.dot(firstMaster, np.linalg.inv(Tmaster)))
            qrem = qrm.elements
            if (qrem[0] < 0):
                qrem = -qrem
            inputmaster[j] = np.hstack((qrem, np.dot(firstMaster, np.linalg.inv(Tmaster))[0:3,3]*scalemaster))
            
            
            qrs = Quaternion(matrix=np.dot(firstSlave, np.linalg.inv(Tslave)))
            qres = qrs.elements
            if (qres[0] < 0):
                qres = -qres
            inputslave[j] = np.hstack((qres, np.dot(firstSlave, np.linalg.inv(Tslave))[0:3,3]*scaleslave))

            
#        invec[i] = np.hstack(( inputmaster.flatten(), inputslave.flatten() ))
        invec[i] = np.hstack(( inputmaster, inputslave )).reshape(1,-1)
        
        
        text_file.close()

    X = invec

    return X, Y

######################################
#
# output the motion under each camera's cooridinate system's position
# as rotatin matrix and translation vector
#
def gen_inputdata_v5(tranum = 1, numFrames = 128, scale = 1):

    #output is camera rig's position under WCS
    tra = GenTrajectory(tranum, numFrames)
    Y = np.zeros((tranum, 12))
    
    invec = np.zeros((tranum, 2 * numFrames * 12))
#    outvec = np.zeros(tranum, numFrames, 2, 12)
    
    for i in range(tranum):
        A, C = tra[i]

        inputmaster = np.zeros((numFrames, 12))
        inputslave = np.zeros((numFrames, 12))
        
        
        # gen different camera setting for each trajectory
        # H1 for rig to master camera
        # H2 for rig to slave camera
        # H3 for slave to master camera
        H1, H2, H3 = gen_cameraset()
        invH1 = np.linalg.inv(H1)
#        invH2 = np.linalg.inv(H2)
#        qr = Quaternion(matrix=H3) #[0:3,0:3])
#        qre = qr.elements
        groundtruth = np.hstack((H3[0:3,0:3].flatten(), H3[0:3,3]))
#        groundtruth = qre
        Y[i] = groundtruth
        
        #gen scale diff
        scalemaster = scale
        scaleslave = 1
#        print(scalemaster, "scale")


        # T^si_m0 = T^s0_m0 T^si_s0 = T^mi_m0 T^si_mi
        # T^s0_m0 = T^mi_m0 = H3
        firstMaster = np.eye(4) #wcs to m0cs
        firstSlave = np.eye(4) #wcs to s0cs
        #firstRig = np.eye(4) #wcs to r0cs
        
        text_file = open("/home/zhao/Desktop/TrackVis/build/myfile0.txt", "w")
        #first master to rig, then slave to master, then rig to world(s)
        #all in homogeneous matrix, flatten to one row
        #text_file.write( np.array2string(H1.flatten(), max_line_width=999) + "\n") #invH1.flatten().tostring() )
        text_file.write( ' '.join(map(str, H1.flatten())) + "\n")
        text_file.write( ' '.join(map(str, H3.flatten())) + "\n") #H3.flatten().tostring() )
        
        
        for j in range(numFrames):
            R = C[j] # .reshape(3,3)
            t = A[j]
            
            # Trig rig-ith to world
            Trig = rt_to_homogeneous(R,t)
            text_file.write( ' '.join(map(str, Trig.flatten())) + "\n")
            # world to rig-ith
            Trig = np.linalg.inv(Trig)
            # 
#            Trig = np.dot(np.linalg.inv(firstRig), Trig)
            
            
            # transform from wcs to m(aster)cs-ith and scs-ith
            Tmaster = np.dot(H1, Trig)
            Tslave = np.dot(H2, Trig)
            
            if ( j == 0 ):
                # world to master 0-th
                firstMaster = Tmaster
                firstSlave = Tslave
                #firstRig = Trig
            
            mat = np.dot(firstMaster, np.linalg.inv(Tmaster))
#            qrm = Quaternion(matrix=np.dot(firstMaster, np.linalg.inv(Tmaster)))
#            qrem = qrm.elements
#            if (qrem[0] < 0):
#                qrem = -qrem
            inputmaster[j] = np.hstack((mat[0:3, 0:3].flatten(), mat[0:3,3]*scalemaster))
            
            mat = np.dot(firstSlave, np.linalg.inv(Tslave))
#            qrs = Quaternion(matrix=np.dot(firstSlave, np.linalg.inv(Tslave)))
#            qres = qrs.elements
#            if (qres[0] < 0):
#                qres = -qres
            inputslave[j] = np.hstack((mat[0:3, 0:3].flatten(), mat[0:3,3]*scaleslave))

            
#        invec[i] = np.hstack(( inputmaster.flatten(), inputslave.flatten() ))
        invec[i] = np.hstack(( inputmaster, inputslave )).reshape(1,-1)
        
        
        text_file.close()

    X = invec

    return X, Y

######################################
#
# output the motion under each camera's cooridinate system's position
# as rotatin matrix and translation vector
#
def gen_inputdata_v6(tranum = 1, numFrames = 128, scale = 1):

    #output is camera rig's position under WCS
    tra = GenTrajectory(tranum, numFrames)
    Y = np.zeros((tranum, 12))
    
#    invec = np.zeros((tranum, 2 * numFrames * 12))
    outvec = np.zeros( (tranum, numFrames, 2, 12) )
    
    for i in range(tranum):
        A, C = tra[i]

#        inputmaster = np.zeros((numFrames, 12))
#        inputslave = np.zeros((numFrames, 12))
        
        
        # gen different camera setting for each trajectory
        # H1 for rig to master camera
        # H2 for rig to slave camera
        # H3 for slave to master camera
        H1, H2, H3 = gen_cameraset()
#        invH1 = np.linalg.inv(H1)
#        invH2 = np.linalg.inv(H2)
#        qr = Quaternion(matrix=H3) #[0:3,0:3])
#        qre = qr.elements
        groundtruth = np.hstack((H3[0:3,0:3].flatten(), H3[0:3,3]))
#        groundtruth = qre
        Y[i] = groundtruth
        
        #gen scale diff
        scalemaster = scale
        scaleslave = 1
#        print(scalemaster, "scale")


        # T^si_m0 = T^s0_m0 T^si_s0 = T^mi_m0 T^si_mi
        # T^s0_m0 = T^mi_m0 = H3
        firstMaster = np.eye(4) #wcs to m0cs
        firstSlave = np.eye(4) #wcs to s0cs
        #firstRig = np.eye(4) #wcs to r0cs
        
        text_file = open("/home/zhao/Desktop/TrackVis/build/myfile0.txt", "w")
        #first master to rig, then slave to master, then rig to world(s)
        #all in homogeneous matrix, flatten to one row
        #text_file.write( np.array2string(H1.flatten(), max_line_width=999) + "\n") #invH1.flatten().tostring() )
        text_file.write( ' '.join(map(str, H1.flatten())) + "\n")
        text_file.write( ' '.join(map(str, H3.flatten())) + "\n") #H3.flatten().tostring() )
        
        
        for j in range(numFrames):
            R = C[j] # .reshape(3,3)
            t = A[j]
            
            # Trig rig-ith to world
            Trig = rt_to_homogeneous(R,t)
            text_file.write( ' '.join(map(str, Trig.flatten())) + "\n")
            # world to rig-ith
            Trig = np.linalg.inv(Trig)
            # 
#            Trig = np.dot(np.linalg.inv(firstRig), Trig)
            
            
            # transform from wcs to m(aster)cs-ith and scs-ith
            Tmaster = np.dot(H1, Trig)
            Tslave = np.dot(H2, Trig)
            
            if ( j == 0 ):
                # world to master 0-th
                firstMaster = Tmaster
                firstSlave = Tslave
                #firstRig = Trig
            
            mat = np.dot(firstMaster, np.linalg.inv(Tmaster))
#            qrm = Quaternion(matrix=np.dot(firstMaster, np.linalg.inv(Tmaster)))
#            qrem = qrm.elements
#            if (qrem[0] < 0):
#                qrem = -qrem
#            inputmaster[j] = np.hstack((mat[0:3, 0:3].flatten(), mat[0:3,3]*scalemaster))
            outvec[i][j][0] = np.hstack((mat[0:3, 0:3].flatten(), mat[0:3,3]*scalemaster))
            
            mat = np.dot(firstSlave, np.linalg.inv(Tslave))
#            qrs = Quaternion(matrix=np.dot(firstSlave, np.linalg.inv(Tslave)))
#            qres = qrs.elements
#            if (qres[0] < 0):
#                qres = -qres
#            inputslave[j] = np.hstack((mat[0:3, 0:3].flatten(), mat[0:3,3]*scaleslave))
            outvec[i][j][1] = np.hstack((mat[0:3, 0:3].flatten(), mat[0:3,3]*scaleslave))

            
#        invec[i] = np.hstack(( inputmaster.flatten(), inputslave.flatten() ))
#        invec[i] = np.hstack(( inputmaster, inputslave )).reshape(1,-1)
        
        
        text_file.close()

    X = outvec

    return X, Y
    

######################################
#
# output the motion under each camera's cooridinate system's position
#
def gen_inputdata_v7(tranum = 1, numFrames = 128, scale = 1):

    #output is camera rig's position under WCS
    tra = GenTrajectory(tranum, numFrames)
    Y = np.zeros((tranum, 7))
    
    invec = np.zeros((tranum, 2 * numFrames * 7))
    outvec = np.zeros( (tranum, numFrames, 2, 7) )
    
    
    for i in range(tranum):
        A, C = tra[i]

        inputmaster = np.zeros((numFrames, 7))
        inputslave = np.zeros((numFrames, 7))
        
        
        # gen different camera setting for each trajectory
        # H1 for rig to master camera
        # H2 for rig to slave camera
        # H3 for slave to master camera
        H1, H2, H3 = gen_cameraset()
        invH1 = np.linalg.inv(H1)
        #invH2 = np.linalg.inv(H2)
        qr = Quaternion(matrix=H3) #[0:3,0:3])
        qre = qr.elements
        groundtruth = np.hstack((qre, H3[0:3,3]))
#        groundtruth = qre
        Y[i] = groundtruth
        
        #gen scale diff
        scalemaster = scale
        scaleslave = 1
#        print(scalemaster, "scale")


        # T^si_m0 = T^s0_m0 T^si_s0 = T^mi_m0 T^si_mi
        # T^s0_m0 = T^mi_m0 = H3
        firstMaster = np.eye(4) #wcs to m0cs
        firstSlave = np.eye(4) #wcs to s0cs
        #firstRig = np.eye(4) #wcs to r0cs
        
        text_file = open("/home/zhao/Desktop/TrackVis/build/myfile0.txt", "w")
        #first master to rig, then slave to master, then rig to world(s)
        #all in homogeneous matrix, flatten to one row
        #text_file.write( np.array2string(H1.flatten(), max_line_width=999) + "\n") #invH1.flatten().tostring() )
        text_file.write( ' '.join(map(str, H1.flatten())) + "\n")
        text_file.write( ' '.join(map(str, H3.flatten())) + "\n") #H3.flatten().tostring() )
        
        
        for j in range(numFrames):
            R = C[j] # .reshape(3,3)
            t = A[j]
            
            # Trig rig-ith to world
            Trig = rt_to_homogeneous(R,t)
            text_file.write( ' '.join(map(str, Trig.flatten())) + "\n")
            # world to rig-ith
            Trig = np.linalg.inv(Trig)
            # 
#            Trig = np.dot(np.linalg.inv(firstRig), Trig)
            
            
            # transform from wcs to m(aster)cs-ith and scs-ith
            Tmaster = np.dot(H1, Trig)
            Tslave = np.dot(H2, Trig)
            
            if ( j == 0 ):
                # world to master 0-th
                firstMaster = Tmaster
                firstSlave = Tslave
                #firstRig = Trig
            
            qrm = Quaternion(matrix=np.dot(firstMaster, np.linalg.inv(Tmaster)))
            qrem = qrm.elements
            if (qrem[0] < 0):
                qrem = -qrem
#            inputmaster[j] = np.hstack((qrem, np.dot(firstMaster, np.linalg.inv(Tmaster))[0:3,3]*scalemaster))
            outvec[i][j][0] = np.hstack(( qrem, np.dot(firstMaster, np.linalg.inv(Tmaster))[0:3,3]*scalemaster ))
            
            
            qrs = Quaternion(matrix=np.dot(firstSlave, np.linalg.inv(Tslave)))
            qres = qrs.elements
            if (qres[0] < 0):
                qres = -qres
#            inputslave[j] = np.hstack((qres, np.dot(firstSlave, np.linalg.inv(Tslave))[0:3,3]*scaleslave))
            outvec[i][j][1] = np.hstack(( qres, np.dot(firstSlave, np.linalg.inv(Tslave))[0:3,3]*scaleslave ))

            
#        invec[i] = np.hstack(( inputmaster.flatten(), inputslave.flatten() ))
#        invec[i] = np.hstack(( inputmaster, inputslave )).reshape(1,-1)
        
        
        text_file.close()

#    X = invec
    X = outvec

    return X, Y

Xinput, Yinput = gen_inputdata_v7(2)




