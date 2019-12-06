import numpy as np
import cv2

def label(img, text, pt, color_fg=(255,255,255), color_bg=(0,0,0),
        font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, thickness=1):
    ''' put text label (text with box)
    pt - bottom left point of label box

    return None
    '''

    size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    # print(size, baseline)

    if color_bg:
        cv2.rectangle(img, (pt[0], pt[1]-size[1]-baseline), (pt[0]+size[0],pt[1]) , color_bg, -1)

    cv2.putText(img, text, (pt[0], pt[1]-baseline), font_face, font_scale, color_fg, thickness, cv2.LINE_AA)

def rectangle(img, rect, color_fg, color_bg=None, thickness=1):
    ''' draw rectangle using rect

    return None
    '''

    p1, p2 = rect_to_points(rect)

    if color_bg:
        cv2.rectangle(img, p1, p2, color_bg, cv2.FILLED)

    cv2.rectangle(img, p1, p2, color_fg, thickness)


def lines_pair(img, pts, color, linewidth=1):            # draw_lines_pair
    ''' draw lines between pairs of points 
    (for 2*n points input, draw n lines)

    img - input image (numpy array)
    pts - image points numpy array sized 2+xN
    color - bgr integer tuple

    return - image
    '''
    for i in range(0,pts.shape[1],2):
        p0 = tuple(pts[0:2,i  ].astype(np.float32))
        p1 = tuple(pts[0:2,i+1].astype(np.float32))
        img = cv2.line(img, p0, p1, color, linewidth)


def cross_mark(img, pts, mark_size, color, linewidth=1): # draw_cross_point
    ''' draw cross mark line (+)

    img - input image (numpy array)
    pts - image points numpy array sized 2+xN

    return - image

    '''
    w = [mark_size/2,0]
    h = [0,mark_size/2]
    pl = pts[0:2,0]-w
    pr = pts[0:2,0]+w
    pt = pts[0:2,0]-h
    pb = pts[0:2,0]+h

    img = cv2.line(img, tuple(pl.astype(np.float32)), tuple(pr.astype(np.float32)), color, linewidth)
    img = cv2.line(img, tuple(pt.astype(np.float32)), tuple(pb.astype(np.float32)), color, linewidth)

    return img


def rect_to_points(r, as_int=True):

    x,y,w,h = r
    if as_int:
        p1 = (int(x  ),int(y  ))
        p2 = (int(x+w),int(y+h))
    else:
        p1 = (x  ,y  )
        p2 = (x+w,y+h)

    return p1, p2


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
    return pg.astype(np.float32)#, dg


def xyz1_to_uv1(K,xyz1): #convert_world_to_image_points
    ''' convert xyz coordinate to uv(image) coordinate
    K    - camera matrix ( K = P * [R|t] )
    xyz1 - 4xn vectors - ex) one of columns has [x,y,z,1].T

    return - 3xn vectors - ex one of columns has [u,v,1].T
    '''
    pp = np.matmul(K,xyz1)
    return (pp/pp[2]).astype(np.float32)


def uv1_to_xy01(Kmi,uv1):  # convert_image_to_world_in_ground_point
    ''' convert image points (uv) to world coordinate in ground (xy0) 
    (assume image points are on the ground

    Kmi - modified inverse camera matrix
         Km - modified camera matrix K with 3rd column removed, hence 3x3
         kmi - inverse of Km
    uv1 - 3xn vectors - ex) one of columns has [u,v,1].T

    return - 4xn vectors - ex one of columns has [x,y,0,1]
    '''
    S = np.array([[1,0,0],[0,1,0],[0,0,0],[0,0,1]])
    K = np.matmul( S, Kmi)
    pp = np.matmul(K,uv1)
    return (pp/pp[3]).astype(np.float32)



def resize(src, size):
    W, H = size
    h, w, _ = src.shape
    src = cv2.resize(src, (0, 0), fx=W/w, fy=H/h)
    return src

def overlay(src, overlay, x, y, scale=1):

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    assert overlay.shape[-1] == src.shape[-1], 'color depth must be equal; overlay {} != src {}'.format(overlay.shape[-1], src.shape[-1])
 
    # loop over all pixels and apply the blending equation
    src[y:y+h, x:x+w, :] = overlay
    return src



colors = {
        'pink'                :	(203, 192, 255),
        'lightpink'           :	(193, 182, 255),
        'hotpink'             :	(180, 105, 255),
        'deeppink'            :	(147,  20, 255),
        'palevioletred'       :	(147, 112, 219),
        'mediumvioletred'     :	(133,  21, 199),
        'lightsalmon'         :	(122, 160, 255),
        'salmon'              :	(114, 128, 250),
        'darksalmon'          :	(122, 150, 233),
        'lightcoral'          :	(128, 128, 240),
        'indianred'           :	( 92,  92, 205),
        'crimson'             :	( 60,  20, 220),
        'firebrick'           :	( 34,  34, 178),
        'darkred'             :	(  0,   0, 139),
        'red'                 :	(  0,   0, 255),
        'orangered'           :	(  0,  69, 255),
        'tomato'              :	( 71,  99, 255),
        'coral'               :	( 80, 127, 255),
        'darkorange'          :	(  0, 140, 255),
        'orange'              :	(  0, 165, 255),
        'yellow'              :	(  0, 255, 255),
        'lightyellow'         :	(224, 255, 255),
        'lemonchiffon'        :	(205, 250, 255),
        'lightgoldenrodyellow':	(210, 250, 250),
        'papayawhip'          :	(213, 239, 255),
        'moccasin'            :	(181, 228, 255),
        'peachpuff'           :	(185, 218, 255),
        'palegoldenrod'       :	(170, 232, 238),
        'khaki'               :	(140, 230, 240),
        'darkkhaki'           :	(107, 183, 189),
        'gold'                :	(  0, 215, 255),
        'cornsilk'            :	(220, 248, 255),
        'blanchedalmond'      :	(205, 235, 255),
        'bisque'              :	(196, 228, 255),
        'navajowhite'         :	(173, 222, 255),
        'wheat'               :	(179, 222, 245),
        'burlywood'           :	(135, 184, 222),
        'tan'                 :	(140, 180, 210),
        'rosybrown'           :	(143, 143, 188),
        'sandybrown'          :	( 96, 164, 244),
        'goldenrod'           :	( 32, 165, 218),
        'darkgoldenrod'       :	( 11, 134, 184),
        'peru'                :	( 63, 133, 205),
        'chocolate'           :	( 30, 105, 210),
        'saddlebrown'         :	( 19,  69, 139),
        'sienna'              :	( 45,  82, 160),
        'brown'               :	( 42,  42, 165),
        'maroon'              :	(  0,   0, 128),
        'darkolivegreen'      :	( 47, 107,  85),
        'olive'               :	(  0, 128, 128),
        'olivedrab'           :	( 35, 142, 107),
        'yellowgreen'         :	( 50, 205, 154),
        'limegreen'           :	( 50, 205,  50),
        'lime'                :	(  0, 255,   0),
        'lawngreen'           :	(  0, 252, 124),
        'chartreuse'          :	(  0, 255, 127),
        'greenyellow'         :	( 47, 255, 173),
        'springgreen'         :	(127, 255,   0),
        'mediumspringgreen'   :	(154, 250,   0),
        'lightgreen'          :	(144, 238, 144),
        'palegreen'           :	(152, 251, 152),
        'darkseagreen'        :	(143, 188, 143),
        'mediumaquamarine'    :	(170, 205, 102),
        'mediumseagreen'      :	(113, 179,  60),
        'seagreen'            :	( 87, 139,  46),
        'forestgreen'         :	( 34, 139,  34),
        'green'               :	(  0, 128,   0),
        'darkgreen'           :	(  0, 100,   0),
        'aqua'                :	(255, 255,   0),
        'cyan'                :	(255, 255,   0),
        'lightcyan'           :	(255, 255, 224),
        'paleturquoise'       :	(238, 238, 175),
        'aquamarine'          :	(212, 255, 127),
        'turquoise'           :	(208, 224,  64),
        'mediumturquoise'     :	(204, 209,  72),
        'darkturquoise'       :	(209, 206,   0),
        'lightseagreen'       :	(170, 178,  32),
        'cadetblue'           :	(160, 158,  95),
        'darkcyan'            :	(139, 139,   0),
        'teal'                :	(128, 128,   0),
        'lightsteelblue'      :	(222, 196, 176),
        'powderblue'          :	(230, 224, 176),
        'lightblue'           :	(230, 216, 173),
        'skyblue'             :	(235, 206, 135),
        'lightskyblue'        :	(250, 206, 135),
        'deepskyblue'         :	(255, 191,   0),
        'dodgerblue'          :	(255, 144,  30),
        'cornflowerblue'      :	(237, 149, 100),
        'steelblue'           :	(180, 130,  70),
        'royalblue'           :	(225, 105,  65),
        'blue'                :	(255,   0,   0),
        'mediumblue'          :	(205,   0,   0),
        'darkblue'            :	(139,   0,   0),
        'navy'                :	(128,   0,   0),
        'midnightblue'        :	(112,  25,  25),
        'lavender'            : (250, 230, 230),
        'thistle'             : (216, 191, 216),
        'plum'                : (221, 160, 221),
        'violet'              : (238, 130, 238),
        'orchid'              : (214, 112, 218),
        'fuchsia'             : (255,   0, 255),
        'magenta'             : (255,   0, 255),
        'mediumorchid'        : (211,  85, 186),
        'mediumpurple'        : (219, 112, 147),
        'blueviolet'          : (226,  43, 138),
        'darkviolet'          : (211,   0, 148),
        'darkorchid'          : (204,  50, 153),
        'darkmagenta'         : (139,   0, 139),
        'purple'              : (128,   0, 128),
        'indigo'              : (130,   0,  75),
        'darkslateblue'       : (139,  61,  72),
        'slateblue'           : (205,  90, 106),
        'mediumslateblue'     : (238, 104, 123),
        'white'               : (255, 255, 255),
        'snow'                : (250, 250, 255),
        'honeydew'            : (240, 255, 240),
        'mintcream'           : (250, 255, 245),
        'azure'               : (255, 255, 240),
        'aliceblue'           : (255, 248, 240),
        'ghostwhite'          : (255, 248, 248),
        'whitesmoke'          : (245, 245, 245),
        'seashell'            : (238, 245, 255),
        'beige'               : (220, 245, 245),
        'oldlace'             : (230, 245, 253),
        'floralwhite'         : (240, 250, 255),
        'ivory'               : (240, 255, 255),
        'antiquewhite'        : (215, 235, 250),
        'linen'               : (230, 240, 250),
        'lavenderblush'       : (245, 240, 255),
        'mistyrose'           : (225, 228, 255),
        'gainsboro'           : (220, 220, 220),
        'lightgray'           : (211, 211, 211),
        'silver'              : (192, 192, 192),
        'darkgray'            : (169, 169, 169),
        'gray'                : (128, 128, 128),
        'dimgray'             : (105, 105, 105),
        'lightslategray'      : (153, 136, 119),
        'slategray'           : (144, 128, 112),
        'darkslategray'       : ( 79,  79,  47),
        'black'               : (  0,   0,   0),
        }







def color(name):
    return colors[name.lower()]




if __name__=='__main__':

    import numpy as np

    img = np.ones((480,640,3), dtype=np.uint8)/2

    ''' test label '''
    label(img, 'label - AbcdefghijklmnopqrsTuvwxyz', (30,30))



    cv2.imshow('Test',img)
    cv2.waitKey(0)

