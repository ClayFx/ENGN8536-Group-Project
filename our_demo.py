import cv2
import matplotlib.pyplot as plt
import copy

from src import util
from src.body import Body
import os


body_estimation = Body('./ckpt/6_parameter_checkpoint.pth')

test_video = './images/video'
oriVideo = [cv2.imread(os.path.join(test_video, oriImg)) for oriImg in os.listdir(test_video)][:-1]

# oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriVideo)
canvas = copy.deepcopy(oriVideo[-1])
canvas = util.draw_bodypose(canvas, candidate, subset)
# canvas = draw_one_person(canvas, candidate)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()