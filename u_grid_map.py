from __future__ import print_function, division

import numpy as np



class GridMap:

    def __init__(self,
            xmin, xmax, xres,
            ymin, ymax, yres):
        self.__dict__.update( { (k,v) for k,v in locals().items() if k!='self'} )

        self.nx = int((xmax-xmin)/xres)
        self.ny = int((ymax-ymin)/yres)

        self.xmax = self.xmin + self.nx * self.xres
        self.ymax = self.ymin + self.ny * self.yres

        self.grid = np.zeros((self.nx,self.ny))

    def p2i(self,x,y):

        ret = None,None

        if not self.xmin < x < self.xmax: 
            print('warning - out of bound x', self.xmin, '<=', x, '<', self.xmax);
            return ret
        if not self.ymin < y < self.ymax:
            print('warning - out of bound y', self.ymin, '<=', y, '<', self.ymax);
            return ret

        return np.array([int((x-self.xmin)/self.xres),
            int((y-self.ymin)/self.yres)])

    def is_filled_xy(self,x,y, threshold=0):
        return self.is_filled_ij( *self.p2i(x,y, threshold) )

    def is_filled_ij(self,i,j, threshold=0):
        if i==None:
            return False
        else:
            return self.grid[i,j]>threshold

    def i2p(self,i,j):
        return np.array([self.xmin+self.xres*i, self.ymin+self.yres*j])

    def set(self,x,y,z=1):

        i,j = self.p2i(x,y)
        self.grid[i,j] = z;

    def clear(self):
        self.grid.fill(0)

    def fill(self, z):
        self.grid.fill(z)

    def to_points(self):
        xl = np.arange(self.xmin, self.xmin+self.xres*self.nx, self.xres)
        yl = np.arange(self.ymin, self.ymin+self.yres*self.ny, self.yres)
        Y,X = np.meshgrid(yl,xl)
        # print('x',X)
        # print('y',Y)
        idx = (self.grid.flatten()>0)

        x = X.flatten()[idx]
        y = Y.flatten()[idx]
        z = self.grid.flatten()[idx]
        return np.c_[x,y,z]

    def to_meshgrid(self):
        xl = np.arange(self.xmin, self.xmin+self.xres*self.nx, self.xres)
        yl = np.arange(self.ymin, self.ymin+self.yres*self.ny, self.yres)
        Y,X = np.meshgrid(yl,xl)
        # print('x',X)
        # print('y',Y)
        # idx = (self.grid.flatten()>0)

        # x = X.flatten()[idx]
        # y = Y.flatten()[idx]
        # z = self.grid.flatten()[idx]
        return X,Y,self.grid

    def line_search(self, xys, radius, threshold=0):

        ps = np.array(xys)

        s = np.zeros(len(ps))
        for i in range(1,len(s)):
            d = np.linalg.norm(ps[i]-ps[i-1])
            s[i] = s[i-1]+d

        res_min = min(self.xres, self.yres)
        t = np.arange(s.min(), s.max()+res_min/2, res_min)
        xn = np.interp(t, s, ps[:,0])
        yn = np.interp(t, s, ps[:,1])

        ires = int(radius/res_min+0.5)
        # print('x', xn)
        # print('y', yn)
        # print(ires)

        idx_list = []
        for x,y in zip(xn,yn):
            i,j = self.p2i(x,y)
            # print(x,y, i,j)
            if i!=None:
                for ii in range(-ires, ires+1):
                    for jj in range(-ires, ires+1):
                        idx = (i+ii, j+jj)
                        # print(idx, self.is_filled_ij(idx[0],idx[1], threshold))
                        if self.is_filled_ij(idx[0],idx[1], threshold):
                            if idx not in idx_list:
                                idx_list.append(idx)

        return idx_list


    def line_interpolation(self, lines):


        lines = np.array(lines)
        res_min = min(self.xres, self.yres)
        res_max = max(self.xres, self.yres)

        for i in range(1,len(lines)):
            line = lines[i-1:i+1]
            idx = np.array(
                    self.line_search(line, res_max*2)
                    )
            # print(i, len(line), len(idx))#, self.i2p(*idx[0]), self.i2p(*idx[-1]))

            idx0 = idx[0]
            idx1 = idx[-1]
            p0 = self.i2p(*idx0)
            p1 = self.i2p(*idx1)
            z0 = self.grid[idx0[0], idx0[1]]
            z1 = self.grid[idx1[0], idx1[1]]

            l = np.linalg.norm(p1-p0)
            n = l//res_min

            ts = np.linspace(0,1,n)

            for t in ts:
                s = t*l
                p = p0 + (p1-p0)*t
                z = z0 + (z1-z0)*t

                self.set(p[0],p[1],z)


    def interpolate(self):

        gridx = self.grid.copy()
        for i in range(gridx.shape[0]):
            idx = gridx[i,:]>0
            y = np.arange(self.ymin, self.ymax, self.yres)
            # print(idx.shape, y.shape, gridx.shape )
            ya = y[idx]
            if len(ya) < 2:
                continue
            else:
                z = np.interp(y, ya, gridx[i,idx], left=0, right=0)
                gridx[i,:] = z
                # print('interp...', z)

        gridy = self.grid.copy()
        for j in range(gridy.shape[1]):
            idx = gridy[:,j]>0
            x = np.arange(self.xmin, self.xmax, self.xres)
            # print(idx.shape, x.shape, gridy.shape )
            xa = x[idx]
            if len(xa) < 2:
                continue
            else:
                z = np.interp(x, xa, gridy[idx,j], left=0, right=0)
                gridy[:,j] = z
                # print('interp...', z)

        idxx = gridx>0
        idxy = gridy>0
        idx = np.logical_and(idxx, idxy)

        self.grid[idx] = (gridx[idx] + gridy[idx])/2



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



    def line_search(self, xys, radius, threshold=0):

        ps = np.array(xys)

        s = np.zeros(len(ps))
        for i in range(1,len(s)):
            d = np.linalg.norm(ps[i]-ps[i-1])
            s[i] = s[i-1]+d

        rres = np.linalg.norm([self.xres, self.yres])
        t = np.arange(s.min(), s.max()+rres/2, rres)
        xn = np.interp(t, s, ps[:,0])
        yn = np.interp(t, s, ps[:,1])

        ires = int(radius/rres+0.5)

        idx_list = []
        for x,y in zip(xn,yn):
            i,j = self.p2i(x,y)
            # print(x,y, i,j)
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


    g2d = GridMap(
            -2,2,0.1, 
            -5,5,0.1)

    for x in np.arange(-2,2,0.1):
        g2d.add_point(x, np.sin(x))


    pp = g2d.to_points()
    print(pp)

    plt.figure();
    ax = plt.subplot(1,1,1);
    ax.scatter(pp[:,0], pp[:,1], c='blue', marker='o')
    ax.grid()



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
    
