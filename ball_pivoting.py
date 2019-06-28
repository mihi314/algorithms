#!/usr/bin/env python3
from collections import namedtuple

import numpy as np
from scipy.spatial import KDTree
from pytest import approx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# based on "The ball-pivoting algorithm for surface reconstruction":
# https://ieeexplore.ieee.org/document/817351

class Edge(namedtuple("Edge", ["i", "j", "o", "c"])):
    def key(self):
        # edges with the same i and j, irrespective of order, are considered the same
        i = min(self.i, self.j)
        j = max(self.i, self.j)
        return (i, j)

    def triangle(self):
        """Return the triangle represented by that edge."""
        return [self.i, self.j, self.o]


NOT_USED = -10
PART_OF_MESH = 0

class State:
    active, boundary = 1, 2

def norm(vec):
    return np.sqrt(np.sum(vec*vec, axis=-1))

def normalize(vec):
    return vec / np.expand_dims(norm(vec), -1)

def isclose(x, y, rtol=1e-5, atol=1e-8):
    return abs(x-y) <= atol + rtol * abs(y)


def construct_surface(points, normals, rho):
    kdtree = KDTree(points)

    # find seed triangle
    c = np.array([2.5, 3.5, 2.22474487])
    seed1 = Edge(i=0, j=1, o=3, c=c)
    seed2 = Edge(i=1, j=3, o=0, c=c)
    seed3 = Edge(i=3, j=0, o=1, c=c)



    # count of how often a point is included in a front edge (>0), part of the mesh (==0), or not yet used (==-10)
    point_state = np.zeros(len(points), dtype=int)
    point_state[:] = NOT_USED
    point_state[seed1.i] = 2
    point_state[seed1.j] = 2
    point_state[seed1.o] = 2

    front_active = {seed1.key(): seed1, seed2.key(): seed2, seed3.key(): seed3}
    front_boundary = {}
    mesh = [seed1]

    while front_active:
        # remove a edge from the active front
        edge = pop_edge(front_active, None, point_state)

        k, c = ball_pivot(edge, rho, kdtree, points, normals)
        if point_state[k] == PART_OF_MESH:
            add_edge(front_boundary, edge, point_state)
            continue

        # normal of triangle vs normal of k
        triangle_normal = np.cross(points[edge.j] - points[k], points[edge.i] - points[k])
        if np.dot(triangle_normal, normals[k]) < 0:
            add_edge(front_boundary, edge, point_state)
            continue

        new_edge1 = Edge(i=edge.i, j=k, o=edge.j, c=c)
        new_edge2 = Edge(i=k, j=edge.j, o=edge.i, c=c)

        mesh.append(new_edge1) # doesn't matter which one, so maybe output triangle (with ball center) here, instead of edge?

        # point not yet used, initialize usage to 0
        if point_state[k] == NOT_USED:
            point_state[k] = 0

        for new_edge in [new_edge1, new_edge2]:
            if new_edge.key() in front_active or new_edge.key() in front_boundary:
                # edge already exits, "glue" the surface along the edge by removing it
                try:
                    pop_edge(front_active, new_edge, point_state)
                except KeyError:
                    pop_edge(front_boundary, new_edge, point_state)
            else:
                # edge does not alreay exist, so add to front
                add_edge(front_active, new_edge, point_state)

            assert(not new_edge.key() in front_boundary)

    return mesh

def add_edge(edge_dict, edge, point_state):
    edge_dict[edge.key()] = edge
    point_state[edge.i] += 1
    point_state[edge.j] += 1

def pop_edge(edge_dict, edge, point_state):
    if edge:
        popped_edge = edge_dict.pop(edge.key())
    else:
        _, popped_edge = edge_dict.popitem()
    point_state[popped_edge.i] -= 1
    point_state[popped_edge.j] -= 1
    assert(point_state[popped_edge.i] >= 0 and point_state[popped_edge.i] >= 0)
    return popped_edge

def ball_pivot(edge, rho, kdtree, points, normals):
    """
    Return the index of the point and ball center that forms the next triangle with 'edge'.
    (Always returns sómething, as in the extreme case it just returns edge.o)
    """
    sigma_i = points[edge.i]
    sigma_j = points[edge.j]
    m = (sigma_i + sigma_j) / 2
    gamma = norm(edge.c - m)
    
    neighbors = kdtree.query_ball_point(m, r=2*rho)
    next_angle = np.inf
    next_c = None
    next_point = None

    for neighbor in neighbors:
        n_gamma = normalize(sigma_j - m)
        cs = sphere_circle_intersect(points[neighbor], rho, m, gamma, n_gamma)
        # print(neighbor, cs)
        if not cs or cs == np.inf:
            continue

        # how much the sphere has to rotate to reach this point (neighbor)
        # angles[0] is hit first, then angles[1] (todo: could remove second calculation)
        angles = list(vec_vec_angle(edge.c - m, c - m, n_gamma) % (2*np.pi) for c in cs)
        # due to numeric instability, points already touching might end up at 2*pi
        if isclose(angles[0], 2*np.pi):
            angles[0] = 0
        # print("{:.2f}, {:.2f}".format(angles[0]/np.pi*180, angles[1]/np.pi*180))
        if len(angles) == 2:
            assert(angles[0] < angles[1] or isclose(angles[1], 0))
        if angles[0] < next_angle:
            next_angle = angles[0]
            next_c = cs[0]
            next_point = neighbor

    return next_point, next_c

def sphere_circle_intersect(m_sphere, r_sphere, m_circ, r_circ, n_circ):
    # see also: https://gamedev.stackexchange.com/a/75775
    assert(norm(n_circ) == approx(1))

    # check if the plane of the circle intersects the sphere
    d = np.dot(n_circ, m_circ - m_sphere)
    if abs(d) > r_sphere:
        return []

    # center and radius of intersection of plane and sphere
    m_intersect = m_sphere + d * n_circ
    r_intersect = np.sqrt(r_sphere**2 - d**2)
    
    return circle_circle_intersect(m_intersect, r_intersect, m_circ, r_circ, n_circ)

def circle_circle_intersect(m1, r1, m2, r2, n):
    assert(np.dot(m2 - m1, n) == approx(0))

    d = norm(m2 - m1)
    # too far apart
    if d > r1 + r2:
        return []
    # same circles
    if isclose(d, 0) and isclose(r1, r2):
        return np.inf
    # within each other
    if d + min(r1, r2) < max(r1, r2):
        return []
    
    # base and perpendicular directions of the triangle
    u = (m2 - m1) / d
    v = np.cross(n, u)

    # the point where the two intersections (p1 and p2) and the connection of the centers intersect
    p = d/2 + (r1**2 - r2**2) / (2 * d)
    pm = m1 + u * p
    if isclose(p, r1):
        return [pm]
    
    h = np.sqrt(r1**2 - p**2)
    p1 = pm + v * h
    p2 = pm - v * h
    return [p1, p2]

def vec_vec_angle(v1, v2, plane_normal):
    # https://stackoverflow.com/a/33920320
    assert(isclose(norm(plane_normal), 1))
    perp = np.cross(v1, v2)
    assert(np.allclose(np.cross(perp, plane_normal), [0, 0, 0]))
    return np.arctan2(np.dot(perp, plane_normal), np.dot(v1, v2))



def check_circle_circle_intersect(args_res):
    args = list(map(np.asarray, args_res[:-1]))
    np.testing.assert_allclose(
        circle_circle_intersect(*args),
        args_res[-1])

def check_sphere_circle_intersect(args_res):
    args = list(map(np.asarray, args_res[:-1]))
    np.testing.assert_allclose(
        sphere_circle_intersect(*args),
        args_res[-1])

def test_circle_circle_intersect():
    # two intersections
    args = [
        (1, 1, 1), # m1
        1, # r1
        (3, 1, 1), # m2
        2, # r2
        (0, 0, 1), # n
        [(1.25, 1+np.sqrt(15)/4, 1), (1.25, 1-np.sqrt(15)/4, 1)] # expected result
    ]
    check_circle_circle_intersect(args)

    # no intersection, radius
    args = [
        (1, 1, 1), # m1
        1, # r1
        (1, 1, 1), # m2
        2, # r2
        (0, 0, 1), # n
        [] # expected result
    ]
    check_circle_circle_intersect(args)

    # no intersection, distance
    args = [
        (1, 1, 1), # m1
        1, # r1
        (6, 1, 1), # m2
        2, # r2
        (0, 0, 1), # n
        [] # expected result
    ]
    check_circle_circle_intersect(args)

    # touch
    args = [
        (1, 1, 1), # m1
        1, # r1
        (4, 1, 1), # m2
        2, # r2
        (0, 0, 1), # n
        [(2, 1, 1)] # expected result
    ]
    check_circle_circle_intersect(args)

    # same circles
    args = [
        (1, 1, 1), # m1
        1, # r1
        (1, 1, 1), # m2
        1, # r2
        (0, 0, 1), # n
        [np.inf] # expected result
    ]
    check_circle_circle_intersect(args)

def test_sphere_circle_intersect():
    # no intersection due to large radius
    args = [
        (1, 1, 1), # m_sphere
        1, # r_sphere
        (1, 1, 1), # m_circ
        2, # r_circ
        (0, 0, -1), # n_circ
        [] # expected result
    ]
    check_sphere_circle_intersect(args)
    # no intersection due to far away
    args = [
        (1, 1, 1), # m_sphere
        1, # r_sphere
        (5, 1, 1), # m_circ
        2, # r_circ
        (0, 0, -1), # n_circ
        [] # expected result
    ]
    check_sphere_circle_intersect(args)
    # no intersection due to far away (in n_circ dir)
    args = [
        (1, 1, 1), # m_sphere
        1, # r_sphere
        (1, 1, 10), # m_circ
        2, # r_circ
        (0, 0, -1), # n_circ
        [] # expected result
    ]
    check_sphere_circle_intersect(args)

    # touch frontal
    args = [
        (1, 1, 1), # m_sphere
        1, # r_sphere
        (3, 1, 1), # m_circ
        1, # r_circ
        (0, 0, 1), # n_circ
        [(2, 1, 1)] # expected result
    ]
    check_sphere_circle_intersect(args)

    # touch side
    args = [
        (1, 1, 1), # m_sphere
        1, # r_sphere
        (2, 1, 0), # m_circ
        1, # r_circ
        (0, 0, 1), # n_circ
        [(1, 1, 0)] # expected result
    ]
    check_sphere_circle_intersect(args)

    # two intersections with plane of circ cutting m_sphere
    args = [
        (1, 1, 1), # m_sphere
        1, # r_sphere
        (3, 1, 1), # m_circ
        2, # r_circ
        (0, 0, 1), # n_circ
        [(1.25, 1+np.sqrt(15)/4, 1), (1.25, 1-np.sqrt(15)/4, 1)] # expected result
    ]
    check_sphere_circle_intersect(args)
    # normal of circ reversed
    args = [
        (1, 1, 1), # m_sphere
        1, # r_sphere
        (3, 1, 1), # m_circ
        2, # r_circ
        (0, 0, -1), # n_circ
        [(1.25, 1-np.sqrt(15)/4, 1), (1.25, 1+np.sqrt(15)/4, 1)] # expected result
    ]
    check_sphere_circle_intersect(args)


def test_ball_pivot_simple():
    points = np.array([(2, 5, 1), (2, 2, 1), (0, 3, 1), (4, 4, 1), (4, 4, 0)])
    kdtree = KDTree(points)
    normals = np.array([(0, 0, 1)] * len(points))
    c = np.array([2.5, 3.5, 2.22474487])
    edge = Edge(i=0, j=1, o=3, c=c)

    rho = 2
    k, next_c = ball_pivot(edge, rho, kdtree, points, normals)
    assert(k == 2)

    # sigma_k (p[2]) already touches
    points = np.array([(2, 5, 1), (2, 2, 1), (1, 3, 1), (4, 4, 1), (4, 4, 0)])
    kdtree = KDTree(points)
    normals = np.array([(0, 0, 1)] * len(points))
    c = np.array([2.5, 3.5, 2.22474487])
    edge = Edge(i=0, j=1, o=3, c=c)

    rho = 2
    k, next_c = ball_pivot(edge, rho, kdtree, points, normals)
    assert(k == 2)



def main():
    points = np.array([(2, 5, 1), (2, 2, 1), (1, 3, 1), (4, 4, 1), (4, 4, 0)])
    ps = np.random.uniform(0, 2, size=(100, 3)) + np.array([4, 4, -2]).reshape((1,3))
    points = np.concatenate([points, ps])
    # ball pivoting needs surface normals, for testing just take the vector pointing away from the origin for each point here
    normals = normalize(points)
    edges = construct_surface(points, normals, 2)
    # print(edges)

    triangles = list([e.i, e.j, e.o] for e in edges)
    cs = np.asarray([e.c for e in edges])
    # print(cs)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.plot_trisurf(*points.T, triangles=triangles)
    ax.plot(*points.T, "b.")
    ax.plot(*cs.T, "r.")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    X, Y, Z = points.T
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


if __name__ == '__main__':
    main()
