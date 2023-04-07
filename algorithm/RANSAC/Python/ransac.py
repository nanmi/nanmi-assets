import random
import math

# Define a function to calculate the distance between two points
def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx*dx + dy*dy)

# Define a function to fit a line to a set of points using RANSAC
def ransac(points, iterations, threshold):
    best_inliers = []
    best_count = 0
    for i in range(iterations):
        # Randomly select two points
        p1, p2 = random.sample(points, 2)
        # Calculate the line between the two points
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - a * p1[0]
        # Count the number of inliers
        inliers = [p for p in points if distance(p, (p[0], a*p[0]+b)) < threshold]
        # Update the best model if this one has more inliers
        if len(inliers) > best_count:
            best_inliers = inliers
            best_count = len(inliers)
    return best_inliers

# ===画图

import numpy as np
import matplotlib.pyplot as plt

y = [-1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , 999 , 999]

x = np.linspace(0, len(y)-1, len(y))

points = [[x, y] for x, y in zip(x, y)]

# 原始数据
y = np.array(y)
plt.scatter(x, y, s=50, c='r')

best_inliers = ransac(points, 15, 50)

# 过滤后数据
y1 = np.array(best_inliers)
x1 = y1[:, 0].reshape(1, -1)
y1 = y1[:, 1].reshape(1, -1)

plt.scatter(x1, y1, s=50, marker='*', c='b')


plt.savefig("./test0.jpg")