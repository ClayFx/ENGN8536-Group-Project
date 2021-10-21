import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import util
from src.body import Body
import os
import math

def draw_one_person(canvas, candidate):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for keypoint in candidate:
            x, y , confidence, class_idx = keypoint # coord
            # Draw the key points
            cv2.circle(canvas, (int(x), int(y)), 4, colors[int(class_idx)], thickness=-1)
    for limb in limbSeq:
        index = np.array(limb) - 1
        cur_canvas = canvas.copy()
        Y = candidate[index.astype(int), 0]
        X = candidate[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        # Draw the lines between key points
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[int(candidate[index.astype(int)[0], 3])])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


body_estimation = Body('./ckpt/1_parameter_checkpoint.pth')

test_video = './images/video'
oriVideo = [cv2.imread(os.path.join(test_video, oriImg)) for oriImg in os.listdir(test_video)][:-1]

# oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriVideo)
canvas = copy.deepcopy(oriVideo[1])
canvas = util.draw_bodypose(canvas, candidate, subset)
# canvas = draw_one_person(canvas, candidate)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()