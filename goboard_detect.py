#!/usr/bin/env python3
from pathlib import Path

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, value_and_grad
# from contexttimer import Timer
from pytest import approx
from scipy.optimize import basinhopping, minimize
from sklearn.cluster import DBSCAN
import cv2 as cv


def detect_lines(img):
    edges = cv.Canny(img, 10, 150, apertureSize=3)
    kernel = np.ones((5,5), np.uint8)
    # difference of closing and image (i.e. only the stuff that was "closed")
    blackhat = cv.morphologyEx(edges, cv.MORPH_BLACKHAT, kernel)
    # w = cv.morphologyEx(edges, cv.MORPH_TOPHAT, kernel)
    # cv.imshow("edge", edges)
    # cv.imshow("img", img)
    # cv.waitKey()
    edges = blackhat
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=60, maxLineGap=30)
    # lines.shape == (-1, 1, 4)
    return lines.reshape(-1, 2, 2).astype(float)

def detect_circles(img):
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, 20,
                              param1=10, param2=150, minRadius=0, maxRadius=20)
    circles = np.uint16(np.around(circles))
    for circle in circles[0,:]:
        # draw the outer circle
        cv.circle(img, (circle[0], circle[1]), circle[2], (0,255,0), 1, lineType=cv.LINE_AA)

def plot_lines(img, lines, color=(0,255,0)):
    for a, b in np.around(lines):
        cv.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1, lineType=cv.LINE_AA)

def threshold_test(img):
    """Figure out parameters for the adaptive threshold."""
    cv.namedWindow('window')
    sizes = range(3,201,2)
    cv.createTrackbar('filter size', 'window', 4, len(sizes)-1, nothing)
    
    cs = range(-20,20)
    cv.createTrackbar('c', 'window', len(cs)//2+4, len(cs)-1, nothing)

    while True:
        size = sizes[cv.getTrackbarPos('filter size', 'window')]
        c = cs[cv.getTrackbarPos('c', 'window')]
        print(size, c)
        thresh1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, size, c)
        cv.imshow("img", thresh1)
        k = cv.waitKey(100) & 0xFF
        if k == 27:
            break

def nothing(*arg):
    pass

# def board_point(corners, i, j, boardsize):
#     """
#       3--2
#     ^ |  |
#     j 0--1
#      i >
#     """
#     assert(corners.shape == (4, 2))
#     assert(0 <= i < boardsize)
#     assert(0 <= j < boardsize)
#     c0, c1, c2, c3 = corners
#     i /= (boardsize - 1)
#     j /= (boardsize - 1)
#     lower = c0*(1-i) + c1*i
#     upper = c3*(1-i) + c2*i
#     return lower*(1-j) + upper*j

def affine_to_euclidian(vecs):
    return vecs[...,:-1] / vecs[...,-1,np.newaxis]

def get_perspective_transform(pa, pb):
    # see: https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.array(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(
        np.dot(np.linalg.inv(np.dot(A.T, A)), A.T),
        B)
    # doing some strange things because of the limitations autograd
    return np.concatenate([np.array(res), np.array([1])]).reshape((3,3))

def get_board_lines(corners, boardsize):
    #     """
    #       3--2
    #     ^ |  |
    #     j 0--1
    #      i >
    #     """
    assert(corners.shape == (4, 2))
    # find the affine transformation between board coords and pixel coords
    coords = (boardsize - 1) * np.array([[0, 0],
                                         [1, 0],
                                         [1, 1],
                                         [0, 1]])
    affine_trafo = get_perspective_transform(coords, corners)

    lines = []
    for k in range(boardsize):
        a = np.dot(affine_trafo, [k, 0, 1])
        b = np.dot(affine_trafo, [k, boardsize-1, 1])
        lines.append([a, b])
        a = np.dot(affine_trafo, [0, k, 1])
        b = np.dot(affine_trafo, [boardsize-1, k, 1])
        lines.append([a, b])
    return affine_to_euclidian(np.array(lines))

def get_board_lines_mat(affine_trafo, boardsize):
    #     """
    #       3--2
    #     ^ |  |
    #     j 0--1
    #      i >
    #     """
    assert(affine_trafo.shape == (3, 3))

    lines = []
    for k in range(boardsize):
        a = np.dot(affine_trafo, [k, 0, 1])
        b = np.dot(affine_trafo, [k, boardsize-1, 1])
        lines.append([a, b])
        a = np.dot(affine_trafo, [0, k, 1])
        b = np.dot(affine_trafo, [boardsize-1, k, 1])
        lines.append([a, b])
    return affine_to_euclidian(np.array(lines))


def norm(vec):
    return np.sqrt(np.sum(vec*vec, axis=-1))

def cross(a, b):
    return a[...,0]*b[...,1] - a[...,1]*b[...,0]

def smooth_min(x, characteristic_length, axis=None):
    k = characteristic_length
    return np.sum(x**(-1/k), axis=axis)**(-k)

def line_point_dist(lines, ps):
    """
    Closest distance of a point to a line segment defined by two points (a, b).
    The arguments can also be lists of lines and points, in that case the distance for
    each combination is returned, with shape lines.shape[:-2] + ps.shape[:-1].
    """

    assert(lines.shape[-2:] == (2, 2))
    assert(ps.shape[-1] == 2)
    a = lines[...,0,:]
    b = lines[...,1,:]
    for _ in range(max(len(ps.shape)-1, 1)):
        a = np.expand_dims(a, -2)
        b = np.expand_dims(b, -2)
    # ps = np.expand_dims(ps, 0)

    v_hat = (b - a) / np.expand_dims(norm(b - a), -1)
    
    # d_along.shape == (v_hat.shape[0], ps.shape[0])
    # i.e. one scalar product for each line-point combination
    d_along = np.sum(v_hat*(ps - a), axis=-1)
    d_normal = np.abs(cross(v_hat, ps - a))
    assert(d_along.shape == d_normal.shape)

    d_ends = np.min(np.array([norm(ps-a), norm(ps-b)]), axis=0)

    # if p lies along the sides of the line use the normal distance,
    # else the distance to one of the ends
    mask = (0 <= d_along) & (d_along <= norm(b - a))
    return np.where(mask, d_normal, d_ends)


def line_line_dist_old(line1, line2):
    """Assymetric line line distance. line1 is base."""
    a1, b1 = line1
    a2, b2 = line2
    v1 = b1 - a1
    v2 = b2 - a2
    d1 = line_point_dist(line1, a2)
    d2 = line_point_dist(line1, (a2+b2)/2)
    d3 = line_point_dist(line1, b2)
    return (d1 + 3*d2 + d3)/3

def line_line_dist(lines1, lines2):
    """Assymetric line line distance. line1 is base."""
    assert(lines1.shape[-2:] == (2, 2))
    assert(lines2.shape[-2:] == (2, 2))

    # if single line, reshape to be a list containing single line
    if len(lines1.shape) == 2:
        lines1 = lines1.reshape((-1, 2, 2))
    if len(lines2.shape) == 2:
        lines2 = lines2.reshape((-1, 2, 2))

    a1 = lines1[...,0,:]
    b1 = lines1[...,1,:]
    a2 = lines2[...,0,:]
    b2 = lines2[...,1,:]

    # combine into one large array with all points to speed things up
    # points = np.stack([a2, (a2+b2)/2, b2], axis=0)
    # ds = line_point_dist(lines1, points)
    # assert(ds.shape[-2] == 3)
    # d1 = ds[...,0,:]
    # d2 = ds[...,1,:]
    # d3 = ds[...,2,:]

    d1 = line_point_dist(lines1, a2)
    d2 = line_point_dist(lines1, (a2+b2)/2)
    d3 = line_point_dist(lines1, b2)

    lengths1 = np.expand_dims(norm(b1-a1), -1)
    lenghts2 = norm(b2-a2)
    length_diff = np.abs(lengths1 - lenghts2) / lengths1

    # return np.max(np.array([d1, d3]), axis=0)
    return (d1 + d2 + d3)/3

def cluster_angles(angles):
    def angle_dist(a1, a2):
        return np.min([(a2 - a1) % 360, (a1 - a2) % 360], axis=0)

    X = np.array(angles).reshape((-1,1))
    db = DBSCAN(eps=6, min_samples=3, metric=angle_dist).fit(X)
    labels = db.labels_
    print(list(zip(angles, labels)))
    # print(labels)


def fit_board(corners, lines_detected, boardsize, img):
    def cost_fun(corners_flat):
        corners = corners_flat.reshape((4, 2))
        lines_board = get_board_lines(corners, boardsize)
        dists_ = line_line_dist(lines_board, lines_detected)
        # for each detected line, find the closes board line
        dists = np.min(dists_, axis=1)
        dists2 = smooth_min(dists_, 1, axis=1)
        # print(dists)
        # print(dists2)
        dists = dists2
        print(dists.shape)
        # print(dists-dists2)
        # for d, l in zip(dists, lines_board):
        #     print(d)
        #     print(l)
        #     print()
        # print(lines_board)
        print(dists.mean())
        # dists = np.where(dists > 10, 10, dists)
        # import IPython; IPython.embed()
        return np.mean(dists)
    
    value_grad_cost = value_and_grad(cost_fun)

    cs = corners.copy()
    rate = 10**2

    alpha = 0.05
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = 0
    v = 0

    for i in range(1, 1000):
        cost, grad = value_grad_cost(cs)
        print(i, cost, grad)
        print(cs)

        img_new = img.copy()
        lines_board_new = get_board_lines(cs, boardsize)
        plot_lines(img_new, lines_board_new, color=(128,0,0))
        cv.imshow("img", img_new)
        cv.waitKey(1)

        # if i == 500:
        #     rate /= 10
        # cs -= grad*rate
        # m = beta1 * m + (1 - beta1) * grad
        # v = beta2 * v + (1 - beta2) * grad**2
        # m_hat = m / (1 - beta1**i)
        # v_hat = v / (1 - beta2**i)
        # alpha_hat = alpha# / np.sqrt(i)
        # cs -= alpha_hat * m_hat / (np.sqrt(v_hat) + eps)

        cs -= alpha * grad / norm(grad.reshape(-1))**2


    # result = minimize(value_grad_cost, corners, jac=True, method='Nelder-Mead',
    #                   options={'maxiter': 1000, 'disp': True})
    # result = basinhopping(value_grad_cost, corners, T=np.sqrt(4*25), minimizer_kwargs={"method": "Nelder-Mead", "jac": True, "options": {'maxiter': 500}}, niter=50)
    # print(result)
    # cs = result.x.reshape((4,2))
    return cs

def fit_board_mat(initial_mat, lines_detected, boardsize, img):
    def cost_fun(mat_flat):
        mat = mat_flat.reshape((3, 3))
        lines_board = get_board_lines_mat(mat, boardsize)
        dists = line_line_dist(lines_board, lines_detected)
        # for each detected line, find the closes board line
        dists = np.min(dists, axis=1)
        print(dists.shape)
        # for d, l in zip(dists, lines_board):
        #     print(d)
        #     print(l)
        #     print()
        # print(lines_board)
        print(dists.mean())
        # dists = np.where(dists > 10, 10, dists)
        # import IPython; IPython.embed()
        return np.mean(dists)
    

    print(cost_fun(initial_mat))
    value_grad_cost = value_and_grad(cost_fun)

    mat = initial_mat.copy()
    rate = 0.0000001
    for i in range(1000):
        cost, grad = value_grad_cost(mat)
        print(i, cost, grad)
        print(mat)

        img_new = img.copy()
        lines_board_new = get_board_lines_mat(mat, boardsize)
        plot_lines(img_new, lines_board_new, color=(128,0,0))
        cv.imshow("img", img_new)
        cv.waitKey(1)

        if i == 500:
            rate /= 5

        mat -= grad*rate

    return mat


def test_line_point_dist():
    lines_points_dists = [
        ([[0, 0], [1, 1]], [1, 0.5], np.sqrt(2)/4),
        ([[0, 0], [1, 1]], [0.5, 1], np.sqrt(2)/4),
        ([[0, 0], [1, 1]], [2, 2], np.sqrt(2)),
        ([[0, 0], [1, 1]], [0.5, 0.5], 0),
        ([[1, 1], [3, 3]], [2, 3], np.sqrt(2)/2),
        ([[1, 1], [3, 3]], [2, 2], 0)]

    for line, p, d_true in lines_points_dists:
        d = line_point_dist(np.array(line), np.array(p))
        # print(d)
        assert(d == approx(d_true))

def test_line_point_dist_vector():
    lines = np.asarray([
        [[0, 0], [1, 1]],
        [[0, 0], [1, -1]],
        [[0, 0], [-1, 1]],
        [[0, 1], [1, 2]]],
        [[1, 1], [3, 3]])

    points = np.asarray([
        [1, 0.5],
        [0.5, 1],
        [2, 2],
        [0.5, 0.5]])

    dists = line_point_dist(lines, points)
    assert(dists.shape == (lines.shape[0], points.shape[0]))
    for i, line in enumerate(lines):
        for j, point in enumerate(points):
            assert(dists[i,j] == line_point_dist(line, point))

# def test_line_line_dist():
#     lines1 = np.asarray([
#         [[0, 0], [1, 0]],
#         [[0, 1], [0, 0]],
#         [[0, 0], [1, 0]]])

#     lines1 = np.asarray([
#         [[0, 1], [1, 1]],
#         [[0, 0], [1, -1]],
#         [[2, 1], [3, 1]]])

#     dists = [1, , np.sqrt(2)]

#     line1 = [[0, 1], [1, 1]]
#     line2 = [[1, 0], np.array([0, 1])]
#     print(line_line_dist(line1, line2))
#     assert(False)

def test_line_line_dist_vector():
    lines1 = np.asarray([
        [[0, 0], [1, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]]])

    lines2 = np.asarray([
        [[0, 1], [1, 1]],
        [[0, 0], [1, -5]]])

    dists = line_line_dist(lines1, lines2)
    assert(dists.shape == (lines1.shape[0], lines2.shape[0]))
    for i, line1 in enumerate(lines1):
        for j, line2 in enumerate(lines2):
            assert(dists[i,j] == line_line_dist_old(line1, line2))


def plot_line_line_dist():
    lines1 = np.array([
        [[0, 1], [1, 2]],
        [[0, 0], [1, 1]]
        ])

    def make_line(x, y, x0, y0):
        return np.array([[x0, y0], [x, y]])

    def plot(ax, x0, y0):
        xs = np.linspace(-2, 3, 50)
        ys = np.linspace(-2, 3, 51)
        X, Y = np.meshgrid(xs, ys)
        lines2 = np.zeros((ys.shape[0], xs.shape[0], 2, 2))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                lines2[j,i,:,:] = make_line(x, y, x0, y0)

        dists = line_line_dist(lines1, lines2.reshape(-1, 2, 2))
        dists_combined = np.min(np.sqrt(dists), axis=0)
        dists_combined = dists_combined.reshape(lines2.shape[:2])

        ax.clear()
        ax.plot(*np.array(lines1).T)
        ax.plot(x0, y0, "o")
        a = ax.contourf(X, Y, dists_combined, 50, cmap='RdGy', vmin=0, vmax=3)
        if not plot.cbar:
            plot.cbar = plt.colorbar(a)
        # plot.cbar.set_clim(vmin=0,vmax=5)
        plot.cbar.draw_all() 
        plt.draw()
    plot.cbar = None

    fig, ax = plt.subplots()
    plot(ax, 0.5, 0.5)

    def onclick(event):
        x, y = event.xdata, event.ydata
        print(x, y)
        plot(ax, x, y)
        plt.show()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def main():
    np.random.seed(1)

    for f in Path("go").glob("*.jpg"):
        boardsize = int(f.name[:2])
        img = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
        # img = cv.equalizeHist(img)
        lines_detected = detect_lines(img)
        np.random.shuffle(lines_detected)
        lines_detected = lines_detected
        plot_lines(img, lines_detected)
        
        c0 = [200., 50.0]
        c1 = [500., 50.0]
        c2 = [500., 400.0]
        c3 = [200., 400.0]
        cs = np.array([c0, c1, c2, c3], dtype=float)
        # cs = np.array([
        #     [244, 74],
        #     [490, 81],
        #     [512, 351],
        #     [227, 345],
        # ], dtype=float)

        lines_board = get_board_lines(cs, boardsize)
        
        # cs_new = fit_board(cs, lines_detected, boardsize, img)
        # lines_board_new = get_board_lines(cs_new, boardsize)


        # coords = (boardsize - 1) * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        # affine_trafo = get_perspective_transform(coords, cs)
        # mat_new = fit_board_mat(affine_trafo, lines_detected, boardsize, img)
        # lines_board_new = get_board_lines_mat(mat_new, boardsize)

        # plot_lines(img, lines_board, color=(255,0,0))
        # plot_lines(img, lines_board_new, color=(128,0,0))

        # threshold_test(img)
        # thresh1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 4)
        # thresh2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 4)

        cv.imshow("img", img)
        cv.waitKey()

        break



if __name__ == "__main__":
    # test_line_line_dist_vector()
    # plot_line_line_dist()
    # test_line_point_dist_vector()
    main()
    # cluster_angles([-2,3,4,6,365,40,90,153,155,157])

# try local thresholding techniques?: https://scikit-image.org/docs/0.13.x/api/skimage.filters.rank.html#skimage.filters.rank.otsu
