import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import os
from tqdm import tqdm
from src import model
from src.body import Body
from src.util import draw_bodypose

body_estimation = Body('../model/body_pose_model.pth')

dir_path = "E:/Textbook/Dataset/MPII/mpii_human_pose_v1_sequences_batch1.tar/mpii_human_pose_v1_sequences_batch1"
image_dir = os.path.join(dir_path, '1')
save_dir = "..\data\label"

# draw the body keypoint and lims
def draw_label(image, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    canvas = np.zeros_like(image)
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            # Draw the key points
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    # for i in range(17):
    #     for n in range(len(subset)):
    #         index = subset[n][np.array(limbSeq[i]) - 1]
    #         if -1 in index:
    #             continue
    #         cur_canvas = canvas.copy()
    #         Y = candidate[index.astype(int), 0]
    #         X = candidate[index.astype(int), 1]
    #         mX = np.mean(X)
    #         mY = np.mean(Y)
    #         length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
    #         angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
    #         # Draw the lines between key points
    #         polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
    #         cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
    #         canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


if __name__ == "__main__":
    # for root, dirs, files in os.walk(image_dir, topdown=True):
    #     print(dirs)
    video_dirs = os.listdir(image_dir)
    with tqdm(video_dirs, ncols=100, ascii=True) as tq:
        for video in tq:
            video_path = os.path.join(image_dir, video)
            for idx, name in enumerate(os.listdir(video_path)):
                image_path = os.path.join(video_path, name)
                oriImg = cv2.imread(image_path)
                candidate, subset = body_estimation(oriImg)
                canvas = copy.deepcopy(oriImg)
                canvas = draw_label(canvas, candidate, subset)
                # canvas = draw_bodypose(canvas, candidate, subset)
                if os.path.exists(os.path.join(save_dir, video)):
                    cv2.imwrite(os.path.join(save_dir, video, name), canvas)
                else:
                    os.mkdir(os.path.join(save_dir, video))
                    cv2.imwrite(os.path.join(save_dir, video, name), canvas)
                tq.set_description(f'Images:[{idx}/{len(os.listdir(video_path))} ]')
