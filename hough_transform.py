from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def hough_lines(edges, threshold, rho_res, theta_res):
    h, w = edges.shape
    rho_max = np.sqrt(h**2 + w**2)
    n_rho = math.ceil(rho_max / rho_res)
    n_theta = math.ceil(2 * np.pi / theta_res)
    accumulator = np.zeros((n_theta, n_rho))


    # points: (n_nonzero, 2)
    points = np.asarray(np.nonzero(edges)).T
    # thetas: (n_thetas, 1)
    thetas = np.linspace(0, 2*np.pi, n_theta)
    thetas = thetas.reshape((-1, 1))
    # rhos: (n_thetas, n_nonzero)
    rhos = points[:,1] * np.cos(thetas) + points[:,0] * np.sin(thetas)

    rhos = np.around(rhos / rho_res, decimals=0).astype(np.int64)

    for i in range(n_theta):
        r = rhos[i,:]
        counts = np.bincount(r[r>=0])
        accumulator[i,:len(counts)] += counts

    found_thetas, found_rhos = np.where(accumulator >= threshold)

    plt.imshow(accumulator.T)
    plt.show()
    print(accumulator.mean())
    return np.asarray([found_thetas * theta_res, found_rhos * rho_res]).T

def plot_lines(img, lines, color=(0,255,0)):
    for a, b in np.around(lines):
        cv.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1, lineType=cv.LINE_AA)

def plot_lines_theta_rho(img, lines, color=(0,255,0)):
    for theta,rho in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img,(x1,y1),(x2,y2),color,1, lineType=cv.LINE_AA)

if __name__ == "__main__":
    img = cv.imread("go/13x13 sun.jpg", cv.IMREAD_GRAYSCALE)
    img = img
    edges = cv.Canny(img, 10, 150, apertureSize=3)
    lines = hough_lines(edges, 120, 0.5, 2*np.pi/1000)
    edges_inv = 255 - edges
    plot_lines_theta_rho(img, lines, color=(0, 128, 0))
    # print(lines)
    plt.imshow(img, cmap="gray")
    plt.show()
    # cv.imshow("img", img)
    # cv.waitKey()