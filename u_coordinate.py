from __future__ import print_function, division

import numpy as np

# print(np.__version__)

''' Rotation Matrix (implementation)

 ref: https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions

   Xnew = [[ cos(q),-sin(q), 0 ],
           [ sin(q), cos(q), 0 ],
           [      0,      0, 1 ]]    *   X


 ref: https://math.stackexchange.com/questions/1137745/proof-of-the-extrinsic-to-intrinsic-rotation-transform

    Extrinsic : T R
        xyz_world = TR * xyz_body
    
    Intrinsic : Rinv Tinv
        xyz_body = Rinv Tinv * xyz_world
  

 ref: http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
  
'''

''' cf) Rotation of Axes 

 ref: https://en.wikipedia.org/wiki/Rotation_of_axes

    Xnew = [[ cos(q), sin(q), 0 ],
            [-sin(q), cos(q), 0 ],
            [      0,      0, 1 ]]    *   X
'''




''' New Interfaces 
    2017.12.08
    TODO: add test funtions
'''

def RotationMatrixFromAxisAngle(axis, theta):
    ''' 
    parameter:

        axis  [ string ] - 'x', 'y', 'z'
        theta [ float  ] - angle in radian

    return:

        R [ numpy array 3x3 ] - rotation matrix

    description:
        
        Rotation matrix 'R' means as below

        v_world = R * v_body

        [x]              =  [         ]                   [x]
        [y]              =  [    R    ]                *  [y]
        [z] world frame  =  [         ] body to world     [z] body frame

    '''

    if   axis=='x':
        return np.array([
            [  1,              0,              0 ],
            [  0,  np.cos(theta), -np.sin(theta) ],
            [  0,  np.sin(theta),  np.cos(theta) ]])
    elif axis=='y':
        return np.array([
            [  np.cos(theta), 0,  np.sin(theta) ],
            [              0, 1,              0 ],
            [ -np.sin(theta), 0,  np.cos(theta) ]])
    elif axis=='z':
        return np.array([
            [  np.cos(theta), -np.sin(theta), 0 ],
            [  np.sin(theta),  np.cos(theta), 0 ],
            [              0,              0, 1 ]])
    else:
        raise Exception('invalid axis value : '+str(axis))


def RotationMatrixFromEulerAngles(r,p,y):
    ''' 
    parameter:

        r [ float ] - roll  in radian
        p [ float ] - pitch in radian
        y [ float ] - yaw   in radian

    return:

        R [ numpy array 3x3 ] - rotation matrix

    description:
        
        Rotation matrix 'R' means as below

        v_world = R * v_body

        [x]              =  [         ]                   [x]
        [y]              =  [    R    ]                *  [y]
        [z] world frame  =  [         ] body to world     [z] body frame

    '''
    return np.dot( RotationMatrixFromAxisAngle('z', y),
            np.dot( RotationMatrixFromAxisAngle('y', p),
                RotationMatrixFromAxisAngle('x', r)))

def RotationMatrixFromQuaternion(q):
    ''' 
    parameter:

        q [ array 4 ] - quaternion in (w,x,y,z) order

    return:

        R [ numpy array 3x3 ] - rotation matrix

    description:

        https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions

    '''
    r,i,j,k = q

    return np.array([
        [ 1-2*j*j-2*k*k, 2*(i*j-k*r), 2*(i*k+j*r) ],
        [ 2*(i*j+k*r), 1-2*i*i-2*k*k, 2*(j*k-i*r) ],
        [ 2*(i*k-j*r), 2*(i*r+j*k), 1-2*i*i-2*j*j ]])


def QuaternionFromRotationMatrix(R):
    ''' 
    parameter:

        R [ numpy array 3x3 ] - rotation matrix

    return:

        q [ array 4 ] - quaternion in (w,x,y,z) order


    description:

        https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
    '''

    r = 0.5*np.sqrt(1.0+R[0,0]+R[1,1]+R[2,2])
    i = (R[2,1]-R[1,2])/(4.0*r)
    j = (R[0,2]-R[2,0])/(4.0*r)
    k = (R[1,0]-R[0,1])/(4.0*r)

    return np.array([r,i,j,k])


def QuaternionFromAxisAngle(v, theta):
    ''' 
    parameter:

        v     [ array 3 ] - rotation axis as unit vector
        theta [ float   ] - rotation angle in radian

    return:

        q [ array 4 ] - quaternion in (w,x,y,z) order

    '''
    q = np.zeros(4)

    q[0] = np.cos(theta/2)
    q[1] = np.sin(theta/2) * v[0]
    q[2] = np.sin(theta/2) * v[1]
    q[3] = np.sin(theta/2) * v[2]

    return q


def AxisAngleFromQuaternion(q):
    ''' 
    parameter:

        q [ array 4 ] - quaternion in (w,x,y,z) order

    return:

        unit_vector [ array 3 ] - rotation axis as unit vector
        theta       [ float   ] - rotation angle in radian

    '''
    if abs(q[0]-1.0) < 1e-8:
        unit_vector = np.array([1.0,0.0,0.0])
        theta       = 0.0
    else:
        theta_d2 = np.arccos(q[0])
        v = q[1:]/np.sin(theta_d2)

        unit_vector = v / np.linalg.norm(v)
        theta       = theta_d2*2.0

    return unit_vector, theta


def EulerAnglesFromRotationMatrix(R):
    ''' 
    parameter:

        R [ numpy array 3x3 ] - rotation matrix

    return:

        rpy [ numpy array 3 ] - (roll, pitch, yaw) order in radian

    description:
    
        https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    '''
 
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        r = np.arctan2(R[2,1] , R[2,2])
        p = np.arctan2(-R[2,0], sy)
        y = np.arctan2(R[1,0], R[0,0])
    else :
        r = np.arctan2(-R[1,2], R[1,1])
        p = np.arctan2(-R[2,0], sy)
        y = 0
 
    return np.array([r, p, y])


def TransformationMatrix(r,p,y, tx,ty,tz):
    ''' 
    parameter:

        r  [ float ] - roll  in radian
        p  [ float ] - pitch in radian
        y  [ float ] - yaw   in radian
        tx [ float ] - x translation
        ty [ float ] - y translation
        tz [ float ] - z translation

    return:

        T [ numpy array 4x4 ] - transformation matrix

            [ R11 R12 R13   tx]
            [ R21 R22 R23   ty]
            [ R31 R32 R33   tz]
            [   0   0   0   1 ]


    '''
    T = np.eye(4)
    T[0,3] = tx
    T[1,3] = ty
    T[2,3] = tz
    T[0:3,0:3] = RotationMatrixFromEulerAngles(r,p,y)

    return T


def TransformationMatrixFromRotationMatrixTranslation(R,t):
    ''' 
    parameter:

        R [ numpy array 3x3 ] - rotation matrix
        t [ numpy array 3   ] - translation vector

    return:

        T [ numpy array 4x4 ] - transformation matrix

            [ R11 R12 R13   tx]
            [ R21 R22 R23   ty]
            [ R31 R32 R33   tz]
            [   0   0   0   1 ]


    '''
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t

    return T


def TransformationMatrixFromEulerAngleTranslation(rpy, tsl):
    ''' 
    parameter:
        rpy [ numpy array 3] - roll, pitch, yaw
        tsl [ numpy array 3] - translation x, y, z

    return:

        T [ numpy array 4x4 ] - transformation matrix

            [ R11 R12 R13   tx]
            [ R21 R22 R23   ty]
            [ R31 R32 R33   tz]
            [   0   0   0   1 ]


    '''
    T = np.eye(4)
    T[:3,:3] = RotationMatrixFromEulerAngles(*rpy)
    T[:3, 3] = tsl
    return T



def SphericalLinearInterpolation(q0, q1, t):

    '''
    spherical linear interpolation
    parameter:

        q0 [ w, x, y, z] - intial rotation in quaternion 
        q1 [ w, x, y, z] - final rotation in quaternion 
        t  [ float, 0<=t<=1] - amount of interpolation

    return:
        qi [ w, x, y, z] - interpolated quaternion


    ref: https://en.wikipedia.org/wiki/Slerp
    '''

    # normalize
    v0 = np.array(q0)/np.linalg.norm(q0)
    v1 = np.array(q1)/np.linalg.norm(q1)

    # amount of rotation
    dot = np.dot(v0, v1)
    if dot < 0:
        v1 = -v1
        dot = -dot

    if dot > 0.9995:
        qi = v0 + t*(v1 - v0)        # linear interpolation

    else:
        theta = np.arccos(dot)
        theta_i = theta*t
        sin_theta = np.sin(theta)
        sin_theta_i = np.sin(theta_i)

        s0 = np.cos(theta_i) - dot * sin_theta_i / sin_theta
        s1 = sin_theta_i / sin_theta
        
        qi = s0*v0 + s1*v1

    return qi/np.linalg.norm(qi)





def Rxyz(theta, axis):

    '''Deprecated ...
       Rotation matrix
       theta [radian]
       axis - 'x','y','z'
    '''

    if   axis=='x':
        return np.array([
            [  1,              0,              0 ],
            [  0,  np.cos(theta), -np.sin(theta) ],
            [  0,  np.sin(theta),  np.cos(theta) ]])
    elif axis=='y':
        return np.array([
            [  np.cos(theta), 0,  np.sin(theta) ],
            [              0, 1,              0 ],
            [ -np.sin(theta), 0,  np.cos(theta) ]])
    elif axis=='z':
        return np.array([
            [  np.cos(theta), -np.sin(theta), 0 ],
            [  np.sin(theta),  np.cos(theta), 0 ],
            [              0,              0, 1 ]])
    else:
        raise Exception('invalid axis value : '+str(axis))


def Rrpy(r,p,y):

    '''Deprecated ...
    Rotation matrix for Euler angle (intrinsic)
    v_world = Rrpy * v_body


    explain 1) rotaion of axes
       -> first, rotate axis z with amount of yaw
        v1 = Rz(yaw).T * v_world
       -> next, rotate axis y with amount of pitch
        v2 = Ry(pitch).T * v_world
       -> last, rotate axis x with amount of roll
        v_body = Rx(roll).T * v_world

       => v_body = Rx(roll).T * Ry(pitch).T * Rz(yaw).T * v_world


    explain 2) rotaion of points

       inverse of explain 1

       => v_world = Rz(yaw) * Ry(pitch) * Rx(roll) * v_body

    '''

    return np.dot( Rxyz(y, 'z'), np.dot( Rxyz(p, 'y'), Rxyz(r, 'x') ) )


###def Rz(psi):
###    ''' Axis Roataion Matrix for Z '''
###    return np.array([
###        [  np.cos(psi), np.sin(psi), 0 ],
###        [ -np.sin(psi), np.cos(psi), 0 ],
###        [            0,           0, 1 ]])
###
###def Ry(theta):
###    ''' Axis Roataion Matrix for Y '''
###    return np.array([
###        [ np.cos(theta), 0, -np.sin(theta) ],
###        [             0, 1,              0 ],
###        [ np.sin(theta), 0,  np.cos(theta) ]])
###
###def Rx(phi):
###    ''' Axis Roataion Matrix for Z '''
###    return np.array([
###        [ 1,            0,           0 ],
###        [ 0,  np.cos(phi), np.sin(phi) ],
###        [ 0, -np.sin(phi), np.cos(phi) ]])
###
###
###def Rxyz(r,p,y):
###    ''' Axes Rotation Matrix for XYZ (Euler Angle)
###        v_body = R_xyz * v_global  (x for roll, y for pitch, z for yaw)
###        Rxyz : C^b_g ( g to b )
###    '''
###    return np.dot(np.dot(Rx(r), Ry(p)), Rz(y))
###

def Cb2g(r,p,y):
    return Rrpy(r,p,y)


def T(r,p,y, tx,ty,tz):
    '''Deprecated ...
    transform matrix  (rotation + translation)
    xyz_world = T * xyz_body
    '''

    T = np.eye(4)
    T[0,3] = tx
    T[1,3] = ty
    T[2,3] = tz
    T[0:3,0:3] = Rrpy(r,p,y)
    return T


def Q2R(q):
    '''Deprecated ...
    Quaternion to rotation matrix
    q - ( w, x, y, z)

    ref: https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
    '''

    r,i,j,k = q

    return np.array([
        [ 1-2*j*j-2*k*k, 2*(i*j-k*r), 2*(i*k+j*r) ],
        [ 2*(i*j+k*r), 1-2*i*i-2*k*k, 2*(j*k-i*r) ],
        [ 2*(i*k-j*r), 2*(i*r+j*k), 1-2*i*i-2*j*j ]])


def R2Q(R):
    '''Deprecated ...
    Rotation matrix to quaternion

    R - 3x3 rotation matrix

    return quaternion (w,x,y,z)

    ref: https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
    '''

    r = 0.5*np.sqrt(1.0+R[0,0]+R[1,1]+R[2,2])
    i = (R[2,1]-R[1,2])/(4.0*r)
    j = (R[0,2]-R[2,0])/(4.0*r)
    k = (R[1,0]-R[0,1])/(4.0*r)

    return np.array([r,i,j,k])


def R2RPY(R):
    '''Deprecated ...
    Roataion matrix to roll, pitch, yaw (X-Y-Z)

    return numpy array [r,p,y]

    ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    '''
 
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def Q2ThetaV(q):
    '''Deprecated ...
    extract angle, unit vector from quaternion (w,x,y,z)
    return theta(rad), unit_vector
    '''
    if abs(q[0]-1.0) < 1e-8:
        return 0, np.array([0.0,0.0,1.0])
    else:
        theta_d2 = np.arccos(q[0])

        v = q[1:]/np.sin(theta_d2)
        unit_vector = v / np.linalg.norm(v)

        return theta_d2*2, unit_vector

def ThetaV2Q(theta,v):
    '''Deprecated ...
    quaternion from angle and unit vector
    return quaternion (w,x,y,z)
    '''
    q = np.zeros(4)

    q[0] = np.cos(theta/2)
    q[1] = np.sin(theta/2) * v[0]
    q[2] = np.sin(theta/2) * v[1]
    q[3] = np.sin(theta/2) * v[2]

    return q








def SPH2XYZ(rng,rho,phi):
    ''' Spherical to XYZ '''
    return np.array([
        rng*np.cos(rho)*np.cos(phi),
        rng*np.cos(rho)*np.sin(phi),
        rng*np.sin(rho)             ])

def SPH2XYZ1(rng,rho,phi):
    ''' Spherical to XYZ1 (corresponding to matrix T) '''
    return np.array([
        rng*np.cos(rho)*np.cos(phi),
        rng*np.cos(rho)*np.sin(phi),
        rng*np.sin(rho),
        1.0])

def SPH2XYZ1A(rng,rho,phi):
    ''' Spherical to XYZ1 Array (corresponding to matrix T) '''
    return np.vstack((
        rng*np.cos(rho)*np.cos(phi),
        rng*np.cos(rho)*np.sin(phi),
        rng*np.sin(rho),
        np.ones_like(rng)))



def get_ground_range(rho, phi, T, ground_height=0.0):
    return (ground_height-T[2,3])/(
        np.cos(rho)*np.cos(phi)*T[2,0] +
        np.cos(rho)*np.sin(phi)*T[2,1] +
        np.sin(rho)            *T[2,2])



def param_form_string_from_transformation_matrix(T):
    eu = EulerAnglesFromRotationMatrix(T[:3,:3])
    s1 = ' -o %.6f %.6f %.6f' % (eu[0], eu[1], eu[2])
    s2 = ' -p %.6f %.6f %.6f' % (T[0,3], T[1,3], T[2,3])
    return s1 + s2



if __name__=='__main__':

    ### TODO add unit test

    epsilon = 1e-10

    R1 = Rxyz(np.pi/2, 'x')
    R2 = np.array([[ 1,0,0], [0,0,-1], [0,1,0]], dtype=np.float)
    assert np.all(np.abs(R1-R2)<epsilon)

    R1 = Rxyz(np.pi/2, 'y')
    R2 = np.array([[ 0,0,1], [0,1,0], [-1,0,0]], dtype=np.float)
    assert np.all(np.abs(R1-R2)<epsilon)

    R1 = Rxyz(np.pi/2, 'z')
    R2 = np.array([[ 0,-1,0], [1,0,0], [0,0,1]], dtype=np.float)
    assert np.all(np.abs(R1-R2)<epsilon)


    R1 = Rxyz(np.pi/2, 'x')
    # print(R1)
    # print(R1-Q2R(R2Q(R1)))
    assert np.all(np.abs(R1-Q2R(R2Q(R1)))<epsilon)


    theta = np.radians(10)
    unitv = np.array([1,0,0])
    q = ThetaV2Q(theta, unitv)
    t,u = Q2ThetaV(q)
    # print(theta, t)
    # print(unitv, u)
    assert abs(theta-t)<epsilon
    assert np.all(np.abs(unitv-u)<epsilon)




    ''' New Interfaces '''
    R1 = RotationMatrixFromAxisAngle('x', np.pi/2)
    R2 = np.array([[ 1,0,0], [0,0,-1], [0,1,0]], dtype=np.float)
    assert np.all(np.abs(R1-R2)<epsilon)

    R1 = RotationMatrixFromAxisAngle('y', np.pi/2)
    R2 = np.array([[ 0,0,1], [0,1,0], [-1,0,0]], dtype=np.float)
    assert np.all(np.abs(R1-R2)<epsilon)

    R1 = RotationMatrixFromAxisAngle('z', np.pi/2)
    R2 = np.array([[ 0,-1,0], [1,0,0], [0,0,1]], dtype=np.float)
    assert np.all(np.abs(R1-R2)<epsilon)

    R1 = RotationMatrixFromAxisAngle('x', np.pi/2)
    assert np.all(np.abs(R1-RotationMatrixFromQuaternion(QuaternionFromRotationMatrix(R1)))<epsilon)

    theta = np.radians(10)
    unitv = np.array([1,0,0])
    q = QuaternionFromAxisAngle(unitv, theta)
    u,t = AxisAngleFromQuaternion(q)
    assert abs(theta-t)<epsilon
    assert np.all(np.abs(unitv-u)<epsilon)


    R1 = RotationMatrixFromEulerAngles(np.pi/2, np.pi/2, np.pi/2)
    Rx = np.array([[ 1,0,0], [0,0,-1], [0,1,0]], dtype=np.float)
    Ry = np.array([[ 0,0,1], [0,1,0], [-1,0,0]], dtype=np.float)
    Rz = np.array([[ 0,-1,0], [1,0,0], [0,0,1]], dtype=np.float)
    R2 = np.dot(Rz, np.dot(Ry, Rx))
    assert np.all(np.abs(R1-R2)<epsilon)


    rpy1 = [0.1, 0.2, 0.3]
    R1 = RotationMatrixFromEulerAngles(*rpy1)
    rpy2 = EulerAnglesFromRotationMatrix(R1)
    assert np.all(np.abs(rpy2-rpy1)<epsilon)


    # TransformationMatrix(r,p,y, tx,ty,tz):
    rpy = [0.1, 0.2, 0.3]
    tsl = [0.5, 0.6, 0.7]
    R = RotationMatrixFromEulerAngles(*rpy)
    p   = np.array([1,1,1,1])
    p1  = np.dot(R,p[:3])+tsl
    p2  = np.dot(TransformationMatrixFromEulerAngleTranslation(rpy,tsl), p)[:3]
    assert np.all(np.abs(p1-p2)<epsilon)
    p3  = np.dot(TransformationMatrix(rpy[0],rpy[1],rpy[2],tsl[0],tsl[1],tsl[2]), p)[:3]
    assert np.all(np.abs(p1-p3)<epsilon)
    p4  = np.dot(TransformationMatrixFromRotationMatrixTranslation(R,tsl), p)[:3]
    assert np.all(np.abs(p1-p4)<epsilon)


    # SphericalLinearInterpolation (SLERP)
    u = [1,0,0]
    thu = 10*np.pi/180
    N = 8
    vv = [0,1,0]
    for i in range(N):
        qr = QuaternionFromAxisAngle(u, thu*i)
        q0 = QuaternionFromAxisAngle(u, thu*N)
        qs = SphericalLinearInterpolation([1,0,0,0], q0, i/N)
            
        assert np.all(np.abs(
            np.dot(RotationMatrixFromQuaternion(qr), vv)
            -np.dot(RotationMatrixFromQuaternion(qs), vv))<epsilon)


