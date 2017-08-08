import numpy as np
import cv2

def draw_lines_pair(img, pts, color):
    ''' draw lines between pairs of points 
    ex) for 2*n points input, draw n lines
    '''
    for i in range(0,pts.shape[1],2):
        p0 = tuple(pts[0:2,i  ].astype(np.float32))
        p1 = tuple(pts[0:2,i+1].astype(np.float32))
        img = cv2.line(img, p0, p1, color, 1)
    return img


def draw_cross_point(img, pts, mark_size, color):
    ''' draw cross mark line (+) not (x)
    '''
    w = [mark_size/2,0]
    h = [0,mark_size/2]
    pl = pts[0:2,0]-w
    pr = pts[0:2,0]+w
    pt = pts[0:2,0]-h
    pb = pts[0:2,0]+h

    cv2.line(img, tuple(pl.astype(np.float32)), tuple(pr.astype(np.float32)), color, 1)
    cv2.line(img, tuple(pt.astype(np.float32)), tuple(pb.astype(np.float32)), color, 1)


def generate_grid_lines( xi, yi):#, A):
    ''' generate grid lines (numpy)
    xi - x-dir [min, max, res]
    yi - y-dir [min, max, res]
    '''
    # grid lines
    lines = []
    x0,x1,xr = xi; #[ 0, 8,0.5] [ min, max, res]
    y0,y1,yr = yi; #[-5, 5,0.5] [ min, max, res]
    for x in np.arange(x0,x1+xr/2,xr):
        lines.append([x,y0,0,1])
        lines.append([x,y1,0,1])
    for y in np.arange(y0,y1+yr/2,yr):
        lines.append([x0,y,0,1])
        lines.append([x1,y,0,1])

    pg = np.array(lines).T
    # dg = get_image_points(A,pg)
    return pg#, dg


def convert_world_to_image_points(K,p):
    ''' convert xyz coordinate to uv(image) coordinate
    K - camera matrix ( K = P * [R|t] )
    p - 4xn vectors - ex) one of columns has [x,y,z,1].T

    return - 4xn vectors - ex one of columns has [u,v,1].T
    '''
    pp = np.matmul(K,p)
    return pp/pp[2]



