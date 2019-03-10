#!/usr/bin/env python 
# -*- coding: utf-8 -*- 

from __future__ import print_function, division 

'''
    reference 

        https://www.python.org/doc/essays/graphs/

    license

        PSF
        [ Copyright (c) 1998, 2000, 2003 Python Software Foundation. ]
        [ All rights reserved.                                       ]
        [ Licensed under the PSF license.                            ]


'''


def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath: return newpath
    return None


def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if not graph.has_key(start):
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest



if __name__=='__main__':

    graph = {
            'A': ['B', 'C'],
            'B': ['C', 'D'],
            'C': ['D'],
            'D': ['C'],
            'E': ['F'],
            'F': ['C']}

    print()
    print('graph')
    print(graph)

    print()
    print('path first match')
    print(find_path(graph, 'A', 'D'))

    print()
    print('path all')
    print(find_all_paths(graph, 'A', 'D'))

    print()
    print('path shortest')
    print(find_shortest_path(graph, 'A', 'D'))





