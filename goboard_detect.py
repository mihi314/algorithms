#!/usr/bin/env python3
from pathlib import Path
import autograd.numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pytest import approx
from contexttimer import Timer
from autograd import grad, value_and_grad
from scipy.optimize import minimize


def detect_lines(img):
    edges = cv.Canny(img, 10, 150, apertureSize=3)
    # cv.imshow("edge", edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)
    return [(np.array([x1, y1], dtype=float), np.array([x2, y2], dtype=float)) for x1, y1, x2, y2 in lines[:,0,:]]

def detect_circles(img):
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, 20,
                              param1=150, param2=20, minRadius=0, maxRadius=20)
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

def board_point(corners, i, j, boardsize):
    """
      3--2
    ^ |  |
    j 0--1
     i >
    """
    assert(corners.shape == (4, 2))
    assert(0 <= i < boardsize)
    assert(0 <= j < boardsize)
    c0, c1, c2, c3 = corners
    i /= (boardsize - 1)
    j /= (boardsize - 1)
    lower = c0*(1-i) + c1*i
    upper = c3*(1-i) + c2*i
    return lower*(1-j) + upper*j

def get_board_lines(corners, boardsize):
    assert(corners.shape == (4, 2))
    lines = []
    for k in range(boardsize):
        a = board_point(corners, k, 0, boardsize)
        b = board_point(corners, k, boardsize-1, boardsize)
        lines.append([a, b])
        a = board_point(corners, 0, k, boardsize)
        b = board_point(corners, boardsize-1, k, boardsize)
        lines.append([a, b])
    return lines

def norm(vec):
    return np.sqrt(np.tensordot(vec, vec, axes=1))

def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

def line_point_dist(line, p):
    a, b = line
    v_hat = (b - a) / norm(b - a)
    d_along = np.dot(v_hat, p)
    d_normal = np.abs(cross(v_hat, p - a))
    if 0 <= d_along <= norm(b - a):
        return d_normal
    else:
        return np.min([norm(p-a), norm(p-b)])

def line_line_dist(line1, line2):
    """Assymetric line line distance. line1 is base."""
    a1, b1 = line1
    a2, b2 = line2
    v1 = b1 - a1
    v2 = b2 - a2
    d1 = line_point_dist(line1, a2)
    d2 = line_point_dist(line1, (a2+b2)/2)
    d3 = line_point_dist(line1, b2)
    return (d1 + 3*d2 + d3)/3
    
def fit_board(corners, lines_detected, boardsize, img):
    def cost_fun(corners_flat):
        corners = corners_flat.reshape((4, 2))
        lines_board = get_board_lines(corners, boardsize)
        dists = []
        for line in lines_detected:
            ds = [line_line_dist(line_b, line) for line_b in lines_board]
            dists.append(np.sqrt(min(ds)))
        
        return sum(dists)
    
    # with Timer() as t:
    #     print(cost(corners))
    # print(t.elapsed)
    # print(cost(corners))
    value_grad_cost = value_and_grad(cost_fun)
    print(value_grad_cost(corners))
    cs = corners.copy()
    rate = 10
    for i in range(1000):
        cost, grad = value_grad_cost(cs)
        print(i, cost, grad)
        print(cs)

        img_new = img.copy()
        lines_board_new = get_board_lines(cs, boardsize)
        plot_lines(img_new, lines_board_new, color=(128,0,0))
        cv.imshow("img", img_new)
        cv.waitKey(100)

        if i == 500:
            rate = 1

        cs -= grad*rate



    # result = minimize(value_grad_cost, corners, jac=True, method='BFGS',
    #                   options={'maxiter': 10, 'disp': True})
    # print(result)
    return cs

def test_line_point_dist():
    line = [np.array([0, 0]), np.array([1, 1])]

    points_dists = [
        ([1, 0.5], np.sqrt(2)/4),
        ([0.5, 1], np.sqrt(2)/4),
        ([2, 2], np.sqrt(2)),
        ([0.5, 0.5], 0)]

    for p, d_true in points_dists:
        d = line_point_dist(line, np.array(p))
        assert(d == approx(d_true))

# def test_line_line_dist():
#     line1 = [np.array([0, 0]), np.array([1, 1])]
#     line2 = [np.array([1, 0]), np.array([0, 1])]
#     print(line_line_dist(line1, line2))
#     assert(False)

def plot_line_line_dist():
    line1 = [np.array([0, 0]), np.array([1, 1])]
    def dist(x, y, x0, y0):
        line2 = [np.array([x0, y0]), np.array([x, y])]
        return line_line_dist(line1, line2)

    def plot(ax, x0, y0):
        x = np.linspace(-2, 3, 50)
        y = np.linspace(-2, 3, 50)
        X, Y = np.meshgrid(x, y)
        dists = np.vectorize(dist)(X, Y, x0, y0)

        ax.clear()
        ax.plot(*np.array(line1).T)
        ax.plot(x0, y0, "o")
        a = ax.contourf(X, Y, dists, 50, cmap='RdGy', vmin=0, vmax=3)
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
    np.random.seed(0)

    for f in Path("go").glob("*.jpg"):
        boardsize = int(f.name[:2])
        img = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
        # img = cv.equalizeHist(img)
        lines_detected = detect_lines(img)
        np.random.shuffle(lines_detected)
        lines_detected = lines_detected[:20]
        plot_lines(img, lines_detected)
        
        c0 = [200., 50.0]
        c1 = [500., 50.0]
        c2 = [500., 400.0]
        c3 = [200., 400.0]
        cs = np.asarray([c0, c1, c2, c3], dtype=float)
        lines_board = get_board_lines(cs, boardsize)
        


        cs_new = fit_board(cs, lines_detected, boardsize, img)
        lines_board_new = get_board_lines(cs_new, boardsize)



        
        plot_lines(img, lines_board, color=(255,0,0))
        plot_lines(img, lines_board_new, color=(128,0,0))

        # threshold_test(img)
        # thresh1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 4)
        # thresh2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 4)

        cv.imshow("img", img)
        cv.waitKey()

        break
    
if __name__ == "__main__":
    # plot_line_line_dist()
    main()
