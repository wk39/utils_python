from __future__ import print_function, division

''' utils for command line in/output
    2018.05.10

    ref:
        https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
'''

# TODO
# 1. print(*args) prints one space ' ' at the start of line
#    why: multiple arguments in print function are joined togather with ' '
#

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_warning(*args):
    args = (bcolors.WARNING,) + args + (bcolors.ENDC,)
    print(*args)

def print_fail(*args):
    args = (bcolors.FAIL,) + args + (bcolors.ENDC,)
    print(*args)

def print_bold(*args):
    args = (bcolors.BOLD,) + args + (bcolors.ENDC,)
    print(*args)



