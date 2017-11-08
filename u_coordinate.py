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
  
  
'''

''' cf) Rotation of Axes 

 ref: https://en.wikipedia.org/wiki/Rotation_of_axes

    Xnew = [[ cos(q), sin(q), 0 ],
            [-sin(q), cos(q), 0 ],
            [      0,      0, 1 ]]    *   X
'''


def Rxyz(theta, axis):

    ''' Rotation matrix
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

    ''' Rotation matrix for Euler angle (intrinsic)
    x_world = Rrpy * r_body
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
    ''' transform matrix  (rotation + translation)
    xyz_world = T * xyz_body
    '''

    T = np.zeros((4,4))
    T[0,3] = tx
    T[1,3] = ty
    T[2,3] = tz
    T[3,3] = 1.0
    T[0:3,0:3] = Rrpy(r,p,y)
    return T


def Q2R(q):
    ''' Quaternion to rotation matrix
    q - ( w, x, y, z)

    ref: https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
    '''

    r,i,j,k = q

    return np.array([
        [ 1-2*j*j-2*k*k, 2*(i*j-k*r), 2*(i*k+j*r) ],
        [ 2*(i*j+k*r), 1-2*i*i-2*k*k, 2*(j*k-i*r) ],
        [ 2*(i*k-j*r), 2*(i*r+j*k), 1-2*i*i-2*j*j ]])


def R2Q(R):
    ''' Rotation matrix to quaternion

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

    '''
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


