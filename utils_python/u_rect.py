# import numpy as np

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



if __name__ == '__main__':

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

