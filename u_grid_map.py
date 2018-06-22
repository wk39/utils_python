from __future__ import print_function, division

import numpy as np



class HistogramGrid:

    def __init__(self,
            xmin, xmax, xres,
            ymin, ymax, yres,
            zmin, zmax, zres,
            ):
        self.__dict__.update( { (k,v) for k,v in locals().items() if k!='self'} )

        self.nx = int((xmax-xmin)/xres)
        self.ny = int((ymax-ymin)/yres)
        self.nz = int((zmax-zmin)/zres)

        self.xmax = self.xmin + self.nx * self.xres
        self.ymax = self.ymin + self.ny * self.yres
        self.zmax = self.zmin + self.nz * self.zres

        self.grid = np.zeros((self.nx,self.ny,self.nz))

    def p2i(self,x,y,z=None):

        if z==None:
            ret = None,None
        else:
            ret = None,None,None

        if not self.xmin < x < self.xmax: 
            print('warning - out of bound x', self.xmin, '<=', x, '<', self.xmax);
            return ret
        if not self.ymin < y < self.ymax:
            print('warning - out of bound y', self.ymin, '<=', y, '<', self.ymax);
            return ret
        if z!=None and not self.zmin < z < self.zmax:
            print('warning - out of bound z', self.zmin, '<=', z, '<', self.zmax);
            return ret

        if z==None:
            return int((x-self.xmin)/self.xres), int((y-self.ymin)/self.yres)
        else:
            return int((x-self.xmin)/self.xres), int((y-self.ymin)/self.yres), int((z-self.zmin)/self.zres)


    def i2p(self,i,j,k=None):
        if k==None:
            return self.xmin+self.xres*i, self.ymin+self.yres*j
        else:
            return self.xmin+self.xres*i, self.ymin+self.yres*j, self.zmin+self.zres*k

    def is_filled_xy(self,x,y):
        return self.is_filled_ij( *self.p2i(x,y) )

    def is_filled_ij(self,i,j):
        if i==None:
            return False
        else:
            return self.grid[i,j].max()>0

    def add_point(self,x,y,z):

        i,j,k = self.p2i(x,y,z)

        if i is not None:
            self.grid[i,j,k] += 1;

    def clear(self):
        self.grid.fill(0)

    def to_points(self, count_threshold=0):
        xl = np.arange(self.xmin, self.xmax, self.xres)
        yl = np.arange(self.ymin, self.ymax, self.yres)
        zl = np.arange(self.zmin, self.zmax, self.zres)
        Y,X,Z = np.meshgrid(yl,xl,zl)
        idx = (self.grid.flatten()>count_threshold)

        x = X.flatten()[idx]
        y = Y.flatten()[idx]
        z = Z.flatten()[idx]
        return np.c_[x,y,z]



    def line_search(self, xys, radius):

        ps = np.array(xys)

        s = np.zeros(len(ps))
        for i in range(1,len(s)):
            d = np.linalg.norm(ps[i]-ps[i-1])
            s[i] = s[i-1]+d

        rres = np.linalg.norm([self.xres, self.yres])
        t = np.arange(s.min(), s.max(), rres)
        xn = np.interp(t, s, ps[:,0])
        yn = np.interp(t, s, ps[:,1])

        ires = int(radius/rres+0.5)

        idx_list = []
        for x,y in zip(xn,yn):
            i,j = self.p2i(x,y)
            print(x,y, i,j)
            if i!=None:
                for ii in range(-ires, ires+1):
                    for jj in range(-ires, ires+1):
                        idx = (i+ii, j+jj)
                        if self.is_filled_ij(*idx):
                            if idx not in idx_list:
                                idx_list.append(idx)

        return idx_list


if __name__=='__main__':

    import time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    import sys


    hi = HistogramGrid(
            -2,2,1.0, 
            -5,5,1.0, 
             0,10,0.5)

    hi.add_point(0,0,1)
    hi.add_point(1,0,5)
    hi.add_point(0,1,6)
    hi.add_point(1,1,9)
    hi.add_point(1.5,-2,1)
    hi.add_point(1,-1,5)
    hi.add_point(0,0,6)
    hi.add_point(-1,1,9)

    pp = hi.to_points()
    print(pp)

    plt.figure();
    ax = plt.subplot(1,1,1, projection='3d');
    ax.scatter(pp[:,0], pp[:,1], pp[:,2], c='blue', marker='o')
    ax.grid()


    plt.show()
    
