#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
from __future__ import print_function, division 

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# https://en.wikipedia.org/wiki/B%C3%A9zier_curve
# https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-837-computer-graphics-fall-2012/lecture-notes/MIT6_837F12_Lec01.pdf
# https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-837-computer-graphics-fall-2012/lecture-notes/MIT6_837F12_Lec02.pdf



def linear_bezier(p0, p1, ts):

    tsm = 1-ts

    pts = p0[None,:]*tsm[:,None] \
            + p1[None,:]*ts[:,None]

    return pts


def quadratic_bezier(p0, p1, p2, ts):

    pts = p0[None,:]*((1-ts)*(1-ts))[:,None] \
            + p1[None,:]*2*((1-ts)*ts)[:,None] \
            + p2[None,:]*(ts*ts)[:,None]

    return pts


def cubic_bezier(p0, p1, p2, p3, ts):

    pts = p0[None,:]*((1-ts)**3)[:,None] \
            + p1[None,:]*3*(((1-ts)**2)*ts)[:,None] \
            + p2[None,:]*3*((1-ts)*(ts**2))[:,None] \
            + p3[None,:]*(ts**3)[:,None]

    return pts


def cubic_bezier_in_matrix_form(ptc, ts):

    '''
    pt - n x dim
    ts - 1d - [0 ... 1]
    '''

    G = ptc.T                                               # Geometry (control points)
    B = np.array([
        [ 1,-3, 3,-1],
        [ 0, 3,-6, 3],
        [ 0, 0, 3,-3],
        [ 0, 0, 0, 1]])                                     # Spline Basis - Bernstein
    T = np.vstack( (np.ones_like(ts), ts, ts**2, ts**3) )   # Power basis

    Q = np.dot(G,np.dot(B,T))                               # Q = GBT

    return Q.T



def cubic_bspline(ptc, ts):

    '''
    pt - n x dim
    ts - 1d - [0 ... 1]
    '''

    G = ptc.T                                               # Geometry (control points)
    B = np.array([
        [ 1,-3, 3,-1],
        [ 4, 0,-6, 3],
        [ 1, 3, 3,-3],
        [ 0, 0, 0, 1]])/6.0                                 # Spline Basis - Bernstein
    T = np.vstack( (np.ones_like(ts), ts, ts**2, ts**3) )   # Power basis

    Q = np.dot(G,np.dot(B,T))                               # Q = GBT

    return Q.T


def cubic_bsplines(ptc, ts):

    '''
    pt - 4 x dim
    ts - 1d - [0 ... 1]
    '''

    n_dim = ptc.shape[1]
    n_ts  = len(ts)

    B = np.array([
        [ 1,-3, 3,-1],
        [ 4, 0,-6, 3],
        [ 1, 3, 3,-3],
        [ 0, 0, 0, 1]])/6.0                                 # Spline Basis - Bernstein
    T = np.vstack( (np.ones_like(ts), ts, ts**2, ts**3) )   # Power basis


    nw = len(ptc)-3
    # print(nw)
    ps = []
    for i in range(nw):
        Gi = ptc[i:i+4].T                  # Geometry (control points)
        # print(Gi)
        Qi = np.dot(Gi,np.dot(B,T))      # Q = GBT  [ dim x 4]x[4x4]x[4xlen(ts)] ==> [dim x len(ts)]
        ps.append(Qi.T)

    return np.vstack(ps)


if __name__=='__main__':

    ts = np.linspace(0,1,100)

    ps = np.array([[0,1], [1,0]])
    #
    ptbz = linear_bezier(ps[0], ps[1], ts)

    plt.figure()
    ax = plt.subplot(1,1,1)
    ax.plot(ps[:,0], ps[:,1], 's-', label='control points')
    ax.plot(ptbz[:,0], ptbz[:,1], label='bezier curve')
    ax.axis('equal')
    ax.grid()
    ax.legend()


    ps = np.array([[0,1], [0.2,0], [1,0]])
    #
    ptbz = quadratic_bezier(ps[0], ps[1], ps[2], ts)

    plt.figure()
    ax = plt.subplot(1,1,1)
    ax.plot(ps[:,0], ps[:,1], 's-', label='control points')
    ax.plot(ptbz[:,0], ptbz[:,1], label='bezier curve')
    ax.axis('equal')
    ax.grid()
    ax.legend()


    ps = np.array([[0,1], [0.5,1], [0.5,0], [1,0]])
    #
    ptbz = cubic_bezier(ps[0], ps[1], ps[2], ps[3], ts)

    plt.figure()
    ax = plt.subplot(1,1,1)
    ax.plot(ps[:,0], ps[:,1], 's-', label='control points')
    ax.plot(ptbz[:,0], ptbz[:,1], label='bezier curve')
    ax.axis('equal')
    ax.grid()
    ax.legend()


    ps = np.array([[0,1], [0.5,1], [0.5,0], [1,0]])
    #
    ptbz = cubic_bezier_in_matrix_form(ps, ts)
    ptbs = cubic_bspline(ps, ts)

    plt.figure()
    ax = plt.subplot(1,1,1)
    ax.plot(ps[:,0], ps[:,1], 's-', label='control points')
    ax.plot(ptbz[:,0], ptbz[:,1], label='bezier curve')
    ax.plot(ptbs[:,0], ptbs[:,1], label='b-spline curve')
    ax.axis('equal')
    ax.grid()
    ax.legend()


    ps = np.array([
        [0,1],
        [0,1],
        [0.5,1],
        [0.5,0],
        [1,0],
        [1,0],
        # [2,1],
        # [3,0],
        # [4,1],
        # [4,1],
        ])
    #
    ptbs = cubic_bsplines(ps, ts)
    # print(ptbs)

    plt.figure()
    ax = plt.subplot(1,1,1)
    ax.plot(ps[:,0], ps[:,1], 's-', label='control points')
    # ax.plot(ptbz[:,0], ptbz[:,1], label='bezier curve')
    ax.plot(ptbs[:,0], ptbs[:,1], label='b-spline curve')
    ax.axis('equal')
    ax.grid()
    ax.legend()




    ''' show and close all together '''
    for i in plt.get_fignums():
        plt.figure(i).canvas.mpl_connect('close_event', lambda event: plt.close('all') )
    plt.show()
    


