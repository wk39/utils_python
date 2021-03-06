import numpy as np

# /**
 # * http://godsnotwheregodsnot.blogspot.kr/2012/09/color-distribution-methodology.html
 # * */
distinct64 = np.array([
    [0.203000, 0.203000, 0.203000],
    [0.003922, 0.000000, 0.403922],
    [0.835294, 1.000000, 0.000000],
    [1.000000, 0.000000, 0.337255],
    [0.619608, 0.000000, 0.556863],
    [0.054902, 0.298039, 0.631373],
    [1.000000, 0.898039, 0.007843],
    [0.000000, 0.372549, 0.223529],
    [0.000000, 1.000000, 0.000000],
    [0.584314, 0.000000, 0.227451],
    [1.000000, 0.576471, 0.494118],
    [0.643137, 0.141176, 0.000000],
    [0.000000, 0.082353, 0.266667],
    [0.568627, 0.815686, 0.796078],
    [0.384314, 0.054902, 0.000000],
    [0.419608, 0.407843, 0.509804],
    [0.000000, 0.000000, 1.000000],
    [0.000000, 0.490196, 0.709804],
    [0.415686, 0.509804, 0.423529],
    [0.000000, 0.682353, 0.494118],
    [0.760784, 0.549020, 0.623529],
    [0.745098, 0.600000, 0.439216],
    [0.000000, 0.560784, 0.611765],
    [0.372549, 0.678431, 0.305882],
    [1.000000, 0.000000, 0.000000],
    [1.000000, 0.000000, 0.964706],
    [1.000000, 0.007843, 0.615686],
    [0.407843, 0.239216, 0.231373],
    [1.000000, 0.454902, 0.639216],
    [0.588235, 0.541176, 0.909804],
    [0.596078, 1.000000, 0.321569],
    [0.654902, 0.341176, 0.250980],
    [0.003922, 1.000000, 0.996078],
    [1.000000, 0.933333, 0.909804],
    [0.996078, 0.537255, 0.000000],
    [0.741176, 0.776471, 1.000000],
    [0.003922, 0.815686, 1.000000],
    [0.733333, 0.533333, 0.000000],
    [0.458824, 0.266667, 0.694118],
    [0.647059, 1.000000, 0.823529],
    [1.000000, 0.650980, 0.996078],
    [0.466667, 0.301961, 0.000000],
    [0.478431, 0.278431, 0.509804],
    [0.149020, 0.203922, 0.000000],
    [0.000000, 0.278431, 0.329412],
    [0.262745, 0.000000, 0.172549],
    [0.709804, 0.000000, 1.000000],
    [1.000000, 0.694118, 0.403922],
    [1.000000, 0.858824, 0.400000],
    [0.564706, 0.984314, 0.572549],
    [0.494118, 0.176471, 0.823529],
    [0.741176, 0.827451, 0.576471],
    [0.898039, 0.435294, 0.996078],
    [0.870588, 1.000000, 0.454902],
    [0.000000, 1.000000, 0.470588],
    [0.000000, 0.607843, 1.000000],
    [0.000000, 0.392157, 0.003922],
    [0.000000, 0.462745, 1.000000],
    [0.521569, 0.662745, 0.000000],
    [0.000000, 0.725490, 0.090196],
    [0.470588, 0.509804, 0.192157],
    [0.000000, 1.000000, 0.776471],
    [1.000000, 0.431373, 0.254902],
    [0.909804, 0.368627, 0.745098],
    [0.003922, 0.000000, 0.403922],
    [0.835294, 1.000000, 0.000000],
    [1.000000, 0.000000, 0.337255],
    [0.619608, 0.000000, 0.556863],
    [0.054902, 0.298039, 0.631373],
    [1.000000, 0.898039, 0.007843],
    [0.000000, 0.372549, 0.223529],
    [0.000000, 1.000000, 0.000000],
    [0.584314, 0.000000, 0.227451],
    [1.000000, 0.576471, 0.494118],
    [0.643137, 0.141176, 0.000000],
    [0.000000, 0.082353, 0.266667],
    [0.568627, 0.815686, 0.796078],
    [0.384314, 0.054902, 0.000000],
    [0.419608, 0.407843, 0.509804],
    [0.000000, 0.000000, 1.000000],
    [0.000000, 0.490196, 0.709804],
    [0.415686, 0.509804, 0.423529],
    [0.000000, 0.682353, 0.494118],
    [0.760784, 0.549020, 0.623529],
    [0.745098, 0.600000, 0.439216],
    [0.000000, 0.560784, 0.611765],
    [0.372549, 0.678431, 0.305882],
    [1.000000, 0.000000, 0.000000],
    [1.000000, 0.000000, 0.964706],
    [1.000000, 0.007843, 0.615686],
    [0.407843, 0.239216, 0.231373],
    [1.000000, 0.454902, 0.639216],
    [0.588235, 0.541176, 0.909804],
    [0.596078, 1.000000, 0.321569],
    [0.654902, 0.341176, 0.250980],
    [0.003922, 1.000000, 0.996078],
    [1.000000, 0.933333, 0.909804],
    [0.996078, 0.537255, 0.000000],
    [0.741176, 0.776471, 1.000000],
    [0.003922, 0.815686, 1.000000],
    [0.733333, 0.533333, 0.000000],
    [0.458824, 0.266667, 0.694118],
    [0.647059, 1.000000, 0.823529],
    [1.000000, 0.650980, 0.996078],
    [0.466667, 0.301961, 0.000000],
    [0.478431, 0.278431, 0.509804],
    [0.149020, 0.203922, 0.000000],
    [0.000000, 0.278431, 0.329412],
    [0.262745, 0.000000, 0.172549],
    [0.709804, 0.000000, 1.000000],
    [1.000000, 0.694118, 0.403922],
    [1.000000, 0.858824, 0.400000],
    [0.564706, 0.984314, 0.572549],
    [0.494118, 0.176471, 0.823529],
    [0.741176, 0.827451, 0.576471],
    [0.898039, 0.435294, 0.996078],
    [0.870588, 1.000000, 0.454902],
    [0.000000, 1.000000, 0.470588],
    [0.000000, 0.607843, 1.000000],
    [0.000000, 0.392157, 0.003922],
    [0.000000, 0.462745, 1.000000],
    [0.521569, 0.662745, 0.000000],
    [0.000000, 0.725490, 0.090196],
    [0.470588, 0.509804, 0.192157],
    [0.000000, 1.000000, 0.776471],
    [1.000000, 0.431373, 0.254902],
    [0.909804, 0.368627, 0.745098],
    [0.003922, 0.000000, 0.403922],
    [0.835294, 1.000000, 0.000000],
    [1.000000, 0.000000, 0.337255],
    [0.619608, 0.000000, 0.556863],
    [0.054902, 0.298039, 0.631373],
    [1.000000, 0.898039, 0.007843],
    [0.000000, 0.372549, 0.223529],
    [0.000000, 1.000000, 0.000000],
    [0.584314, 0.000000, 0.227451],
    [1.000000, 0.576471, 0.494118],
    [0.643137, 0.141176, 0.000000],
    [0.000000, 0.082353, 0.266667],
    [0.568627, 0.815686, 0.796078],
    [0.384314, 0.054902, 0.000000],
    [0.419608, 0.407843, 0.509804],
    [0.000000, 0.000000, 1.000000],
    [0.000000, 0.490196, 0.709804],
    [0.415686, 0.509804, 0.423529],
    [0.000000, 0.682353, 0.494118],
    [0.760784, 0.549020, 0.623529],
    [0.745098, 0.600000, 0.439216],
    [0.000000, 0.560784, 0.611765],
    [0.372549, 0.678431, 0.305882],
    [1.000000, 0.000000, 0.000000],
    [1.000000, 0.000000, 0.964706],
    [1.000000, 0.007843, 0.615686],
    [0.407843, 0.239216, 0.231373],
    [1.000000, 0.454902, 0.639216],
    [0.588235, 0.541176, 0.909804],
    [0.596078, 1.000000, 0.321569],
    [0.654902, 0.341176, 0.250980],
    [0.003922, 1.000000, 0.996078],
    [1.000000, 0.933333, 0.909804],
    [0.996078, 0.537255, 0.000000],
    [0.741176, 0.776471, 1.000000],
    [0.003922, 0.815686, 1.000000],
    [0.733333, 0.533333, 0.000000],
    [0.458824, 0.266667, 0.694118],
    [0.647059, 1.000000, 0.823529],
    [1.000000, 0.650980, 0.996078],
    [0.466667, 0.301961, 0.000000],
    [0.478431, 0.278431, 0.509804],
    [0.149020, 0.203922, 0.000000],
    [0.000000, 0.278431, 0.329412],
    [0.262745, 0.000000, 0.172549],
    [0.709804, 0.000000, 1.000000],
    [1.000000, 0.694118, 0.403922],
    [1.000000, 0.858824, 0.400000],
    [0.564706, 0.984314, 0.572549],
    [0.494118, 0.176471, 0.823529],
    [0.741176, 0.827451, 0.576471],
    [0.898039, 0.435294, 0.996078],
    [0.870588, 1.000000, 0.454902],
    [0.000000, 1.000000, 0.470588],
    [0.000000, 0.607843, 1.000000],
    [0.000000, 0.392157, 0.003922],
    [0.000000, 0.462745, 1.000000],
    [0.521569, 0.662745, 0.000000],
    [0.000000, 0.725490, 0.090196],
    [0.470588, 0.509804, 0.192157],
    [0.000000, 1.000000, 0.776471],
    [1.000000, 0.431373, 0.254902],
    [0.909804, 0.368627, 0.745098],
    [0.003922, 0.000000, 0.403922],
    [0.835294, 1.000000, 0.000000],
    [1.000000, 0.000000, 0.337255],
    [0.619608, 0.000000, 0.556863],
    [0.054902, 0.298039, 0.631373],
    [1.000000, 0.898039, 0.007843],
    [0.000000, 0.372549, 0.223529],
    [0.000000, 1.000000, 0.000000],
    [0.584314, 0.000000, 0.227451],
    [1.000000, 0.576471, 0.494118],
    [0.643137, 0.141176, 0.000000],
    [0.000000, 0.082353, 0.266667],
    [0.568627, 0.815686, 0.796078],
    [0.384314, 0.054902, 0.000000],
    [0.419608, 0.407843, 0.509804],
    [0.000000, 0.000000, 1.000000],
    [0.000000, 0.490196, 0.709804],
    [0.415686, 0.509804, 0.423529],
    [0.000000, 0.682353, 0.494118],
    [0.760784, 0.549020, 0.623529],
    [0.745098, 0.600000, 0.439216],
    [0.000000, 0.560784, 0.611765],
    [0.372549, 0.678431, 0.305882],
    [1.000000, 0.000000, 0.000000],
    [1.000000, 0.000000, 0.964706],
    [1.000000, 0.007843, 0.615686],
    [0.407843, 0.239216, 0.231373],
    [1.000000, 0.454902, 0.639216],
    [0.588235, 0.541176, 0.909804],
    [0.596078, 1.000000, 0.321569],
    [0.654902, 0.341176, 0.250980],
    [0.003922, 1.000000, 0.996078],
    [1.000000, 0.933333, 0.909804],
    [0.996078, 0.537255, 0.000000],
    [0.741176, 0.776471, 1.000000],
    [0.003922, 0.815686, 1.000000],
    [0.733333, 0.533333, 0.000000],
    [0.458824, 0.266667, 0.694118],
    [0.647059, 1.000000, 0.823529],
    [1.000000, 0.650980, 0.996078],
    [0.466667, 0.301961, 0.000000],
    [0.478431, 0.278431, 0.509804],
    [0.149020, 0.203922, 0.000000],
    [0.000000, 0.278431, 0.329412],
    [0.262745, 0.000000, 0.172549],
    [0.709804, 0.000000, 1.000000],
    [1.000000, 0.694118, 0.403922],
    [1.000000, 0.858824, 0.400000],
    [0.564706, 0.984314, 0.572549],
    [0.494118, 0.176471, 0.823529],
    [0.741176, 0.827451, 0.576471],
    [0.898039, 0.435294, 0.996078],
    [0.870588, 1.000000, 0.454902],
    [0.000000, 1.000000, 0.470588],
    [0.000000, 0.607843, 1.000000],
    [0.000000, 0.392157, 0.003922],
    [0.000000, 0.462745, 1.000000],
    [0.521569, 0.662745, 0.000000],
    [0.000000, 0.725490, 0.090196],
    [0.470588, 0.509804, 0.192157],
    [0.000000, 1.000000, 0.776471],
    [1.000000, 0.431373, 0.254902],
    [0.909804, 0.368627, 0.745098],
    [0.003922, 0.000000, 0.403922],
    [0.835294, 1.000000, 0.000000],
    [1.000000, 0.000000, 0.337255]]);


AliceBlue         = np.array([0xF0,0xF8,0xFF])/255
AntiqueWhite      = np.array([0xFA,0xEB,0xD7])/255
Aqua              = np.array([0x00,0xFF,0xFF])/255
Aquamarine        = np.array([0x7F,0xFF,0xD4])/255
Azure             = np.array([0xF0,0xFF,0xFF])/255
Beige             = np.array([0xF5,0xF5,0xDC])/255
Bisque            = np.array([0xFF,0xE4,0xC4])/255
Black             = np.array([0x00,0x00,0x00])/255
BlanchedAlmond    = np.array([0xFF,0xEB,0xCD])/255
Blue              = np.array([0x00,0x00,0xFF])/255
BlueViolet        = np.array([0x8A,0x2B,0xE2])/255
Brown             = np.array([0xA5,0x2A,0x2A])/255
BurlyWood         = np.array([0xDE,0xB8,0x87])/255
CadetBlue         = np.array([0x5F,0x9E,0xA0])/255
Chartreuse        = np.array([0x7F,0xFF,0x00])/255
Chocolate         = np.array([0xD2,0x69,0x1E])/255
Coral             = np.array([0xFF,0x7F,0x50])/255
CornflowerBlue    = np.array([0x64,0x95,0xED])/255
Cornsilk          = np.array([0xFF,0xF8,0xDC])/255
Crimson           = np.array([0xDC,0x14,0x3C])/255
Cyan              = np.array([0x00,0xFF,0xFF])/255
DarkBlue          = np.array([0x00,0x00,0x8B])/255
DarkCyan          = np.array([0x00,0x8B,0x8B])/255
DarkGoldenRod     = np.array([0xB8,0x86,0x0B])/255
DarkGray          = np.array([0xA9,0xA9,0xA9])/255
DarkGrey          = np.array([0xA9,0xA9,0xA9])/255
DarkGreen         = np.array([0x00,0x64,0x00])/255
DarkKhaki         = np.array([0xBD,0xB7,0x6B])/255
DarkMagenta       = np.array([0x8B,0x00,0x8B])/255
DarkOliveGreen    = np.array([0x55,0x6B,0x2F])/255
DarkOrange        = np.array([0xFF,0x8C,0x00])/255
DarkOrchid        = np.array([0x99,0x32,0xCC])/255
DarkRed           = np.array([0x8B,0x00,0x00])/255
DarkSalmon        = np.array([0xE9,0x96,0x7A])/255
DarkSeaGreen      = np.array([0x8F,0xBC,0x8F])/255
DarkSlateBlue     = np.array([0x48,0x3D,0x8B])/255
DarkSlateGray     = np.array([0x2F,0x4F,0x4F])/255
DarkSlateGrey     = np.array([0x2F,0x4F,0x4F])/255
DarkTurquoise     = np.array([0x00,0xCE,0xD1])/255
DarkViolet        = np.array([0x94,0x00,0xD3])/255
DeepPink          = np.array([0xFF,0x14,0x93])/255
DeepSkyBlue       = np.array([0x00,0xBF,0xFF])/255
DimGray           = np.array([0x69,0x69,0x69])/255
DimGrey           = np.array([0x69,0x69,0x69])/255
DodgerBlue        = np.array([0x1E,0x90,0xFF])/255
FireBrick         = np.array([0xB2,0x22,0x22])/255
FloralWhite       = np.array([0xFF,0xFA,0xF0])/255
ForestGreen       = np.array([0x22,0x8B,0x22])/255
Fuchsia           = np.array([0xFF,0x00,0xFF])/255
Gainsboro         = np.array([0xDC,0xDC,0xDC])/255
GhostWhite        = np.array([0xF8,0xF8,0xFF])/255
Gold              = np.array([0xFF,0xD7,0x00])/255
GoldenRod         = np.array([0xDA,0xA5,0x20])/255
Gray              = np.array([0x80,0x80,0x80])/255
Grey              = np.array([0x80,0x80,0x80])/255
Green             = np.array([0x00,0x80,0x00])/255
GreenYellow       = np.array([0xAD,0xFF,0x2F])/255
HoneyDew          = np.array([0xF0,0xFF,0xF0])/255
HotPink           = np.array([0xFF,0x69,0xB4])/255
IndianRed         = np.array([0xCD,0x5C,0x5C])/255
Indigo            = np.array([0x4B,0x00,0x82])/255
Ivory             = np.array([0xFF,0xFF,0xF0])/255
Khaki             = np.array([0xF0,0xE6,0x8C])/255
Lavender          = np.array([0xE6,0xE6,0xFA])/255
LavenderBlush     = np.array([0xFF,0xF0,0xF5])/255
LawnGreen         = np.array([0x7C,0xFC,0x00])/255
LemonChiffon      = np.array([0xFF,0xFA,0xCD])/255
LightBlue         = np.array([0xAD,0xD8,0xE6])/255
LightCoral        = np.array([0xF0,0x80,0x80])/255
LightCyan         = np.array([0xE0,0xFF,0xFF])/255
LightGoldenRodYellow  = np.array([0xFA,0xFA,0xD2])/255
LightGray         = np.array([0xD3,0xD3,0xD3])/255
LightGrey         = np.array([0xD3,0xD3,0xD3])/255
LightGreen        = np.array([0x90,0xEE,0x90])/255
LightPink         = np.array([0xFF,0xB6,0xC1])/255
LightSalmon       = np.array([0xFF,0xA0,0x7A])/255
LightSeaGreen     = np.array([0x20,0xB2,0xAA])/255
LightSkyBlue      = np.array([0x87,0xCE,0xFA])/255
LightSlateGray    = np.array([0x77,0x88,0x99])/255
LightSlateGrey    = np.array([0x77,0x88,0x99])/255
LightSteelBlue    = np.array([0xB0,0xC4,0xDE])/255
LightYellow       = np.array([0xFF,0xFF,0xE0])/255
Lime              = np.array([0x00,0xFF,0x00])/255
LimeGreen         = np.array([0x32,0xCD,0x32])/255
Linen             = np.array([0xFA,0xF0,0xE6])/255
Magenta           = np.array([0xFF,0x00,0xFF])/255
Maroon            = np.array([0x80,0x00,0x00])/255
MediumAquaMarine  = np.array([0x66,0xCD,0xAA])/255
MediumBlue        = np.array([0x00,0x00,0xCD])/255
MediumOrchid      = np.array([0xBA,0x55,0xD3])/255
MediumPurple      = np.array([0x93,0x70,0xDB])/255
MediumSeaGreen    = np.array([0x3C,0xB3,0x71])/255
MediumSlateBlue   = np.array([0x7B,0x68,0xEE])/255
MediumSpringGreen = np.array([0x00,0xFA,0x9A])/255
MediumTurquoise   = np.array([0x48,0xD1,0xCC])/255
MediumVioletRed   = np.array([0xC7,0x15,0x85])/255
MidnightBlue      = np.array([0x19,0x19,0x70])/255
MintCream         = np.array([0xF5,0xFF,0xFA])/255
MistyRose         = np.array([0xFF,0xE4,0xE1])/255
Moccasin          = np.array([0xFF,0xE4,0xB5])/255
NavajoWhite       = np.array([0xFF,0xDE,0xAD])/255
Navy              = np.array([0x00,0x00,0x80])/255
OldLace           = np.array([0xFD,0xF5,0xE6])/255
Olive             = np.array([0x80,0x80,0x00])/255
OliveDrab         = np.array([0x6B,0x8E,0x23])/255
Orange            = np.array([0xFF,0xA5,0x00])/255
OrangeRed         = np.array([0xFF,0x45,0x00])/255
Orchid            = np.array([0xDA,0x70,0xD6])/255
PaleGoldenRod     = np.array([0xEE,0xE8,0xAA])/255
PaleGreen         = np.array([0x98,0xFB,0x98])/255
PaleTurquoise     = np.array([0xAF,0xEE,0xEE])/255
PaleVioletRed     = np.array([0xDB,0x70,0x93])/255
PapayaWhip        = np.array([0xFF,0xEF,0xD5])/255
PeachPuff         = np.array([0xFF,0xDA,0xB9])/255
Peru              = np.array([0xCD,0x85,0x3F])/255
Pink              = np.array([0xFF,0xC0,0xCB])/255
Plum              = np.array([0xDD,0xA0,0xDD])/255
PowderBlue        = np.array([0xB0,0xE0,0xE6])/255
Purple            = np.array([0x80,0x00,0x80])/255
RebeccaPurple     = np.array([0x66,0x33,0x99])/255
Red               = np.array([0xFF,0x00,0x00])/255
RosyBrown         = np.array([0xBC,0x8F,0x8F])/255
RoyalBlue         = np.array([0x41,0x69,0xE1])/255
SaddleBrown       = np.array([0x8B,0x45,0x13])/255
Salmon            = np.array([0xFA,0x80,0x72])/255
SandyBrown        = np.array([0xF4,0xA4,0x60])/255
SeaGreen          = np.array([0x2E,0x8B,0x57])/255
SeaShell          = np.array([0xFF,0xF5,0xEE])/255
Sienna            = np.array([0xA0,0x52,0x2D])/255
Silver            = np.array([0xC0,0xC0,0xC0])/255
SkyBlue           = np.array([0x87,0xCE,0xEB])/255
SlateBlue         = np.array([0x6A,0x5A,0xCD])/255
SlateGray         = np.array([0x70,0x80,0x90])/255
SlateGrey         = np.array([0x70,0x80,0x90])/255
Snow              = np.array([0xFF,0xFA,0xFA])/255
SpringGreen       = np.array([0x00,0xFF,0x7F])/255
SteelBlue         = np.array([0x46,0x82,0xB4])/255
Tan               = np.array([0xD2,0xB4,0x8C])/255
Teal              = np.array([0x00,0x80,0x80])/255
Thistle           = np.array([0xD8,0xBF,0xD8])/255
Tomato            = np.array([0xFF,0x63,0x47])/255
Turquoise         = np.array([0x40,0xE0,0xD0])/255
Violet            = np.array([0xEE,0x82,0xEE])/255
Wheat             = np.array([0xF5,0xDE,0xB3])/255
White             = np.array([0xFF,0xFF,0xFF])/255
WhiteSmoke        = np.array([0xF5,0xF5,0xF5])/255
Yellow            = np.array([0xFF,0xFF,0x00])/255
YellowGreen       = np.array([0x9A,0xCD,0x32])/255

