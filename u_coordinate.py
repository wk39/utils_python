import numpy as np

## https://en.wikipedia.org/wiki/Rotation_of_axes

def Rz(psi):
    ''' Axis Roataion Matrix for Z '''
    return np.array([
        [  np.cos(psi), np.sin(psi), 0 ],
        [ -np.sin(psi), np.cos(psi), 0 ],
        [            0,           0, 1 ]])

def Ry(theta):
    ''' Axis Roataion Matrix for Y '''
    return np.array([
        [ np.cos(theta), 0, -np.sin(theta) ],
        [             0, 1,              0 ],
        [ np.sin(theta), 0,  np.cos(theta) ]])

def Rx(phi):
    ''' Axis Roataion Matrix for Z '''
    return np.array([
        [ 1,            0,           0 ],
        [ 0,  np.cos(phi), np.sin(phi) ],
        [ 0, -np.sin(phi), np.cos(phi) ]])


def Rxyz(r,p,y):
    ''' Axes Rotation Matrix for XYZ (Euler Angle)
        v_body = R_xyz * v_global  (x for roll, y for pitch, z for yaw)
        Rxyz : C^b_g ( g to b )
    '''
    return np.dot(np.dot(Rx(r), Ry(p)), Rz(y))


def Cb2g(r,p,y):
    return Rxyz(r,p,y).T


def Tb2g(r,p,y,tx,ty,tz):
    T = np.zeros((4,4))
    T[0,3] = tx
    T[1,3] = ty
    T[2,3] = tz
    T[3,3] = 1.0
    T[0:3,0:3] = Cb2g(r,p,y)
    return T



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




def get_ground_range(rho, phi, Tbs, ground_height=0.0):
    return (ground_height-Tbs[2,3])/(
        np.cos(rho)*np.cos(phi)*Tbs[2,0] +
        np.cos(rho)*np.sin(phi)*Tbs[2,1] +
        np.sin(rho)            *Tbs[2,2])

