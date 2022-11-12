import cv2
import numpy as np

BOXES = [
    [502, 344, 502 + 61, 344 + 86],
    [627, 427, 627 + 193, 427 + 93],
    [522, 654, 522 + 90, 654 + 85],
    [704, 594, 704 + 90, 594 + 80],
    [545, 577, 545 + 81, 577 + 75],
    [460, 700, 460 + 52, 700 + 80]
]



image = cv2.imread('video_frames/frame_13449.jpg')[344:344 + 86 , 502:502 + 61]
cv2.imwrite('BOX1.jpg', image)

image = cv2.imread('video_frames/frame_1.jpg')[427:427 + 87 , 627:627 + 193]
cv2.imwrite('BOX2.jpg', image)

image = cv2.imread('video_frames/frame_1.jpg')[654:654 + 85 , 522:522 + 90]
cv2.imwrite('BOX3.jpg', image)

image = cv2.imread('video_frames/frame_6069.jpg')[594:594 + 80 , 704:704 + 90]
cv2.imwrite('BOX4.jpg', image)

image = cv2.imread('video_frames/frame_1.jpg')[577:577 + 75 , 545:545 + 81]
cv2.imwrite('BOX5.jpg', image)

image1 = cv2.imread('video_frames/frame_2064.jpg')[700:700 + 80 , 460:460 + 52]
cv2.imwrite('BOX6_1.jpg', image1)

image2 = cv2.imread('video_frames/frame_1.jpg')[700:700 + 80 , 460:460 + 52]
cv2.imwrite('BOX6_2.jpg', image2)