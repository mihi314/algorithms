import math
import numpy as np
import matplotlib.pyplot as plt
import operator
from contexttimer import Timer

class Node(object):
    def __init__(self, p, left, right, dir):
        """
        p is the location of the node.
        left and right can be an instance of Node or None.
        dir is the direction along which they were split.
        """
        self.p = p
        self.left = left
        self.right = right
        self.dir = dir

    def __str__(self):
        if not self.left and not self.right:
            return str(self.p)

        left = lpad_lines(str(self.left),   "─┬─",
                                            " │ ",
                                            " │ ")
        right = lpad_lines(str(self.right), " └─",
                                            "   ",
                                            "   ")
        p = str(self.p)
        return lpad_lines("{}\n{}".format(left, right), p, " "*len(p), " "*len(p))


def lpad_lines(string, first, mid, last):
    lines = string.splitlines()
    if len(lines) == 1:
        return first + lines[0]
    lines[0] = first + lines[0]
    lines[1:-1] = map(lambda l: mid + l, lines[1:-1])
    lines[-1] = last + lines[-1]
    return "\n".join(lines)
    
def dist(p1, p2):
    s = sum((x-y)**2 for x, y in zip(p1, p2))
    return math.sqrt(s)


class Kdtree(object):
    def __init__(self, points, k):
        self.k = k
        self.root = self._construct(points, 0)
    
    def _construct(self, points, dir):
        # could also implement quickselect for median finding, or use median of a few random points
        points = sorted(points, key=lambda p: p[dir])
        if not points:
            return None

        median = len(points) // 2
        # make sure that all points to the right of median are strictly greater
        while median+1 < len(points) and points[median][dir] == points[median+1][dir]:
            median += 1
        new_dir = (dir + 1) % self.k
        left = self._construct(points[:median], new_dir)
        right = self._construct(points[median+1:], new_dir)
        return Node(points[median], left, right, dir)

    def add(self, p):
        if not self.root:
            self.root = Node(p, None, None, 0)
            return

        curr_node = self.root
        while True:
            dir = curr_node.dir
            side = "left" if p[dir] <= curr_node.p[dir] else "right"

            node = getattr(curr_node, side)
            if node:
                curr_node = node
            else:
                new_dir = (dir + 1) % self.k
                setattr(curr_node, side, Node(p, None, None, new_dir))
                break

    def remove(self, p):
        """Remove all occurences of p."""
        found = self.remove_first(p)
        if not found:
            return False
        while found:
            found = self.remove_first(p)
        return True
        
    def remove_first(self, p):
        """Remove the first occurence of p."""
        node, parent, side = self.find_first(p)
        if not node:
            return False
        
        children = self._get_all_points(node.left) + self._get_all_points(node.right)
        new_tree = self._construct(children, node.dir)
        if parent:
            setattr(parent, side, new_tree)
        else:
            self.root = new_tree
        return True

    def find_first(self, p):
        if not self.root:
            return None, None, None

        node = self.root
        parent = None
        side = None

        while True:
            if node.p == p:
                return node, parent, side
            next_side = "left" if p[node.dir] <= node.p[node.dir] else "right"
            next_node = getattr(node, next_side)
            if next_node:
                parent = node
                node = next_node
                side = next_side
            else:
                return None, None, None
        assert(False)

    def _get_all_points(self, node):
        if not node:
            return []
        left = self._get_all_points(node.left)
        right = self._get_all_points(node.right)
        return left + [node.p] + right

    def nearest_neighbor(self, p):
        """Return the coordinates of the nearest neighbor of p in the tree."""
        self.vist_count = 0
        return self._nearest_neighbor(self.root, p)

    def _nearest_neighbor(self, node, p):
        self.vist_count += 1

        nearest_global = node.p
        dist_nearest_global = dist(node.p, p)

        # go down the left or right branch first depding on which side of the hyperplane p falls
        if p[node.dir] <= node.p[node.dir]:
            branches = [(node.left, "L"), (node.right, "R")]
        else:
            branches = [(node.right, "R"), (node.left, "L")]

        for branch, side in branches:
            if not branch:
                continue

            nearest = self._nearest_neighbor(branch, p)
            distance = dist(nearest, p)
            
            if distance < dist_nearest_global:
                nearest_global = nearest
                dist_nearest_global = distance
                # if the sphere around p with radius of the nearest found neighbor
                # in branch does not intersect the dividing hyperlane
                # defined by node.p, the nearest neigher has to be in this branch
                if (side == "L" and p[node.dir] + distance <= node.p[node.dir] or
                    side == "R" and p[node.dir] - distance > node.p[node.dir]):
                    return nearest
        return nearest_global

    # def query_radius(self, p, r):
    #     """Returns all points within r of p."""

    def __str__(self):
        return str(self.root)

# todos:
#   consistency check function
#   k nearest


def check_is_leaf(node, p):
    assert(node.p == p)
    assert(node.left == None)
    assert(node.right == None)

def get_nearest(points, p):
    return min(points, key=lambda x: dist(x, p))


def test_one_point():
    ps = [(1, 2)]
    tree = Kdtree(ps, 2)
    check_is_leaf(tree.root, ps[0])

def test_two_point():
    ps = [(1, 2), (0, 4)]
    tree = Kdtree(ps, 2)
    assert(tree.root.p == ps[0])
    assert(tree.root.left.p == ps[1])
    assert(tree.root.right == None)
    
def test_three_points():
    ps = [(4, 2), (1, 3), (3, 3)]
    tree = Kdtree(ps, 2)
    assert(tree.root.p == ps[2])
    check_is_leaf(tree.root.left, ps[1])
    check_is_leaf(tree.root.right, ps[0])

def test_three_points2():
    ps = [(4, 2), (1, 3), (3, 3)]
    tree = Kdtree(ps, 2)
    assert(tree.root.p == ps[2])
    check_is_leaf(tree.root.left, ps[1])
    check_is_leaf(tree.root.right, ps[0])

def test_nearest_neighbor():
    ps = list(map(tuple, np.random.uniform(-1, 1, (100, 2))))
    p = (0.5, -0.5)
    tree = Kdtree(ps, 2)
    assert(tree.nearest_neighbor(p) == get_nearest(ps, p))

def test_nearest_neighbor2():
    ps = [(4, 2), (1, 3), (3, 3)] * 3
    p = (1, 3)
    tree = Kdtree(ps, 2)
    assert(tree.nearest_neighbor(p) == get_nearest(ps, p))

def test_add_point_empty():
    tree = Kdtree([], 2)
    p = (1, 1)
    tree.add(p)
    check_is_leaf(tree.root, p)

def test_add_point():
    ps = [(5, 2), (1, 3), (4, 3)]
    tree = Kdtree(ps, 2)

    p = (2, 5)
    tree.add(p)
    check_is_leaf(tree.root.left.right, p)
    assert(tree.root.left.left == None)

    p = (3, 4)
    tree.add(p)
    check_is_leaf(tree.root.left.right.right, p)
    assert(tree.root.left.right.left == None)

def test_remove_empty():
    tree = Kdtree([], 2)
    assert(tree.remove((1,2)) == False)

def test_remove_not_found():
    ps = [(4, 2), (1, 3), (3, 3)]
    tree = Kdtree(ps, 2)
    assert(tree.remove((10,10)) == False)
    assert(tree.root.p == ps[2])
    check_is_leaf(tree.root.left, ps[1])
    check_is_leaf(tree.root.right, ps[0])

def test_remove_root():
    ps = [(4, 2), (1, 3), (3, 3)]
    tree = Kdtree(ps, 2)
    assert(tree.remove(ps[2]) == True)
    assert(tree.root.p == ps[0])
    assert(tree.root.right == None)
    check_is_leaf(tree.root.left, ps[1])

def test_remove_leaf():
    ps = [(4, 2), (1, 3), (3, 3)]
    tree = Kdtree(ps, 2)
    assert(tree.remove((1, 3)) == True)
    assert(tree.root.p == ps[2])
    assert(tree.root.left == None)
    check_is_leaf(tree.root.right, ps[0])

def test_remove_multiple():
    ps = [(4, 2), (1, 3), (3, 3)] * 2
    tree = Kdtree(ps, 2)
    tree.remove((1,3))
    check_is_leaf(tree.root.left, (3, 3))


if __name__ == "__main__":

    Ns = list(range(1, 600, 2))
    counts = []
    for N in Ns:
        p = tuple(np.random.uniform(-1, 1, 3))
        # p = (0, 0)
        ps = list(map(tuple, np.random.uniform(-1, 1, (N, 3))))

        tree = Kdtree(ps, 3)
        with Timer() as t_tree:
            nearest_tree = tree.nearest_neighbor(p)
        with Timer() as t_linear:
            nearest_linear = get_nearest(ps, p)
        assert(nearest_tree == nearest_linear)

        # print("tree+search: {:.5f}s, linear: {:.5f}s".format(t_tree.elapsed, t_linear.elapsed))

        counts.append(tree.vist_count)
    
    plt.plot(Ns, Ns)
    plt.plot(Ns, counts)
    plt.plot(Ns, np.log(Ns))#/np.log(Ns[-1])*counts[-1])
    plt.show()
