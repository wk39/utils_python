from __future__ import division, print_function

import numpy as np


def rect_to_points(r):
    return (r[0],r[1]), (r[0]+r[2], r[1]+r[3])

def points_to_rect(p1,p2):
    return (p1[0],p1[1], p2[0]-p1[0], p2[1]-p1[1])

def union_rect(r1, r2):

    (a1,b1),(c1,d1) = rect_to_points(r1)
    (a2,b2),(c2,d2) = rect_to_points(r2)

    a = min(a1,a2)
    b = min(b1,b2)
    c = max(c1,c2)
    d = max(d1,d2)

    return points_to_rect((a,b),(c,d))

def intersection_rect(r1,r2, return_none=False):

    #                  a1   c1
    #                  x----x
    #                  .    .
    # case1)      x----x    .
    #                 c2    .
    #                       .
    # case2)                x----x     
    #                       a2

    (a1,b1),(c1,d1) = rect_to_points(r1)
    (a2,b2),(c2,d2) = rect_to_points(r2)

    if a1>c2 or c1<a2 or b1>d2 or d1<b2 :
        # no intersection
        if return_none:
            return None
        else:
            return (0,0,0,0)
    else:
        a = max(a1,a2)
        b = max(b1,b2)
        c = min(c1,c2)
        d = min(d1,d2)
        return points_to_rect((a,b),(c,d))

def area_rect(r):
    return r[2]*r[3]


def iou_rect(r1,r2):
    ai = area_rect(intersection_rect(r1,r2))
    return ai / (area_rect(r1)+area_rect(r2)-ai)




''' line segment intersection

    ref: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

'''

def points_to_ppr(p1, p2):
    '''
    points to p, r vector

    [in]
        p1 - start point of the line [ array like ]
        p2 - end point of the line [ array like ]
    [out]
        p  - p1 [numpy array]
        r  - vector (p1-->p2) [numpy array]
    '''
    return np.array(p1), np.array(p2)-np.array(p1)



def intersect_line_segments(p1, p2, p3, p4):
    '''
    check two line segments have intersection or not

    [in]
        p1 - start point of line1
        p2 - end point of line1
        p3 - start point of line2
        p4 - end point of line2

    [out]
        r  - False : no intersection
           - True  : intersect
    '''
    p,r = points_to_ppr(p1, p2)
    q,s = points_to_ppr(p3, p4)

    # prepare data
    #
    crs = np.cross(r,s)
    cqmpr = np.cross(q-p,r)
    # print('r x s     =', crs)
    # print('(q-p) x r =', cqmpr)
    if crs==0.:                 # collinear or parallel
        t,u = 10.0, 10.0        # meaning less value
    else:
        t = np.cross(q-p,s)/crs
        u = np.cross(q-p,r)/crs
    #
    # decision
    if crs==0. and cqmpr==0.:
        ''' collinear '''
        t0 = np.dot(q-p,r)/np.dot(r,r)
        t1 = np.dot(q+s-p,r)/np.dot(r,r)
        print('collinear', t0, t1)
        # t1 = t0 + np.dot(s,r)/np.dot(r,r)
        # print('t0 =', t0)
        # print('t1 =', t0)
        if max(t0,t1)<0.0 or 1.0<min(t0,t1):
            pass
        else:
        # if 0.0<=t0<=1.0 or 0.0<=t1<=1.0:
            ''' collinear and intersecting '''
            print('collinear and intersect ...', t0, t1)
            return True
    elif crs==0. and cqmpr!=0.:
        ''' parallel and not intersecting '''
        return False
    elif crs!=0. and 0.<=t<=1. and 0.<=u<=1.:
        ''' not parallel and intersecting '''
        return True
    else:
        ''' not intersecting '''
        pass

    return False



def intersection_point(p1, p2, p3, p4):
    '''
    get intersection point of two lines

    [in]
        p1 - start point of line1
        p2 - end point of line1
        p3 - start point of line2
        p4 - end point of line2

    [out]
        p  - point or None


    ref
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    '''
    
    c12 = p1[0]*p2[1] - p2[0]*p1[1]
    c34 = p3[0]*p4[1] - p4[0]*p3[1]

    deno = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0])

    

    if deno==0:
        return None
    else:
        return np.array([
            c12*(p3[0]-p4[0])-c34*(p1[0]-p2[0]),
            c12*(p3[1]-p4[1])-c34*(p1[1]-p2[1])])/deno


def intersection_tu(p1, p2, p3, p4):
    '''
    get intersection t, u of two lines

    [in]
        p1 - start point of line1
        p2 - end point of line1
        p3 - start point of line2
        p4 - end point of line2

    [out]
        t  - parameter
        u  - parameter


    ref
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    '''
    
    deno = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0])

    if deno==0:
        return None, None
    else:
        return np.array([
            (p1[0]-p3[0])*(p3[1]-p4[1])-(p1[1]-p3[1])*(p3[0]-p4[0]),
           -(p1[0]-p2[0])*(p1[1]-p3[1])+(p1[1]-p2[1])*(p1[0]-p3[0])])/deno



if __name__ == '__main__':

    if False:

        import numpy as np
        import cv2
        import u_opencv as ucv

        rects = [
            ( (10,10,200,200), (90,90,200,200) ),
            ( (90,90,200,200), (10,10,200,200) ),
            ( (90,10,200,200), (10,90,200,200) ),
            ( (10,90,200,200), (90,10,200,200) ),
            ( (100,100,200,50), (200,50,50,200) ),
            ( (200,50,50,200), (100,100,200,50) ),
            ( (100,100,200,200), (150,150,50,50) ),
            ( (150,150,50,50), (100,100,200,200) ),
            ( (100,100,50,50), (300,300,50,50) ),
            ( (300,300,50,50), (100,100,50,50) ),
        ]


        for i, (r1,r2) in enumerate(rects):

            print('rectangles',i)
            print(' area  1:', area_rect(r1))
            print(' area  2:', area_rect(r2))
            print(' area  int:', area_rect(intersection_rect(r1,r2)))
            print(' area  iou:', iou_rect(r1,r2))

            img = np.zeros((512,512,3),dtype=np.uint8)

            ucv.rectangle( img, r1, ucv.color('darkblue'),thickness=3)
            ucv.rectangle( img, r2, ucv.color('darkgreen'),thickness=3)
            ucv.rectangle( img, intersection_rect(r1,r2), ucv.color('magenta'))
            ucv.rectangle( img, union_rect(r1,r2), ucv.color('cyan'))

            cv2.imshow( 'test', img)

            cv2.waitKey(0)


    if True:

        ps = np.array([
            [ [0,0], [1,0] ],
            [ [1,1], [2,1] ],
            [ [2,2], [3,2] ],
            [ [3,3], [4,3] ] ])

        print('//', intersect_line_segments( ps[0,0], ps[2,0], ps[0,1], ps[2,1]) )
        print()
        print('x ', intersect_line_segments( ps[0,0], ps[2,1], ps[0,1], ps[2,0]) )
        print()
        print('./', intersect_line_segments( ps[0,0], ps[1,0], ps[0,1], ps[2,1]) )
        print()
        print('/.', intersect_line_segments( ps[0,0], ps[2,0], ps[0,1], ps[1,1]) )
        print()
        print('/<', intersect_line_segments( ps[0,0], ps[2,0], ps[0,1], ps[1,0]) )
        print()
            
        print('V ', intersect_line_segments( ps[0,1], ps[2,0], ps[0,1], ps[2,1]) )
        print()
        print('^ ', intersect_line_segments( ps[0,0], ps[2,0], ps[0,1], ps[2,0]) )
        print()
        print('/-', intersect_line_segments( ps[0,0], ps[2,0], ps[1,0], ps[1,1]) )
        print()
        print('-/', intersect_line_segments( ps[1,0], ps[1,1], ps[2,1], ps[0,1]) )
        print()
        print('= ', intersect_line_segments( ps[0,0], ps[0,1], ps[2,0], ps[2,1]) )
        print()

        print(': ', intersect_line_segments( ps[0,0], ps[1,0], ps[1,0], ps[3,0]) )
        print()
        print(': ', intersect_line_segments( ps[0,0], ps[2,0], ps[2,0], ps[3,0]) )
        print()
        print(': ', intersect_line_segments( ps[0,0], ps[2,0], ps[1,0], ps[2,0]) )
        print()
        print('/_', intersect_line_segments( ps[0,0], ps[2,0], ps[0,0], ps[0,1]) )
        print()

        # print('error generated')
        # print('/_', intersect_line_segments( ps[0,0], ps[0,0], ps[0,0], ps[0,1]) )
        # print()



    ''' intersection_point '''
    p1 = [-1, 0]
    p2 = [ 1, 0]
    p3 = [ 0,-1]
    p4 = [ 0, 1]
    pc = [ 0, 0]
    assert np.allclose(pc, intersection_point(p1,p2,p3,p4))

    p1 = [-1, 1]
    p2 = [ 1, 1]
    p3 = [ 0,-2]
    p4 = [ 0, 2]
    pc = [ 0, 1]
    assert np.allclose(pc, intersection_point(p1,p2,p3,p4))

    dx=2.5; dy=1.2;
    pc = [ 1, 1]
    p1 = [pc[0]-dx,pc[1]-dy]
    p2 = [pc[0]+dx,pc[1]+dy]
    p3 = [pc[0]-dx,pc[1]+dy]
    p4 = [pc[0]+dx,pc[1]-dy]
    try:
        assert np.allclose(pc, intersection_point(p1,p2,p3,p4))
    except Exception as e:
        print('pc', pc)
        print('intersection point', intersection_point(p1,p2,p3,p4))
        raise(e)

    dx=2.5; dy=1.2;
    pc = [ 1, 1]
    p1 = [pc[0]-dx,pc[1]-dy]
    p2 = [pc[0]+dx,pc[1]+dy]
    p3 = [pc[0]-dx,pc[1]-dy]
    p4 = [pc[0]+dx,pc[1]+dy]
    try:
        assert intersection_point(p1,p2,p3,p4) is None
    except Exception as e:
        print('pc', pc)
        print('intersection point', intersection_point(p1,p2,p3,p4))
        raise(e)



