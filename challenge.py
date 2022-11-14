import cv2
import numpy as np
import os

def mse(imageA, imageB):
	err = np.sum((np.asarray(imageA) - np.asarray(imageB)))
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	return err

def preprocess(image):
    blur = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    return cv2.adaptiveThreshold(blur, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 4)

BOXES = [
    [502, 344, 502 + 61, 344 + 86],
    [627, 427, 627 + 193, 427 + 80],
    [532, 654, 532 + 90, 654 + 85],
    [704, 594, 704 + 90, 594 + 80],
    [545, 577, 545 + 81, 577 + 75],
    [460, 700, 460 + 60, 700 + 80],
]

BOXIMAGES = [
    preprocess(cv2.imread('BOX1.jpg')),
    preprocess(cv2.imread('BOX2.jpg')),
    preprocess(cv2.imread('BOX3.jpg')),
    preprocess(cv2.imread('BOX4.jpg')),
    preprocess(cv2.imread('BOX5.jpg')),
    preprocess(cv2.imread('BOX6_1.jpg')),
    preprocess(cv2.imread('BOX6_2.jpg'))
]

THRESHOLDS = [23, 14, 28, 6, 33, 27]

def evaluate():

    counts = [0, 0, 0, 0, 0, 0]
    
    directory = 'video_frames'

    # base images saved already

    # iterate through every frame
    p = 0
    for filename in os.listdir(directory):
        p += 1
        if p % 100 == 0:
            print(counts)

        f = os.path.join(directory, filename)
        # current frame
        new_img = cv2.imread(f)
        new_img = cv2.resize(new_img, (1024, 1024))

        # iterate through 6 boxes in the frame and compare to base images
        i = 0
        for x, y, w, h in BOXES:
            new_img_box = preprocess(new_img[y:h, x:w])
            mseval = mse(new_img_box, BOXIMAGES[i])

            # modified mseval
            if i==5:
                mseval = (mseval + mse(new_img_box, BOXIMAGES[i+1])) / 2
            
            if mseval > THRESHOLDS[i]:
                counts[i] += 1

            i += 1

    return counts


def video(path):

    video_frames_count = 0

    counts = [0] * 6

    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def onChange(trackbarValue):
        cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
        _, img = cap.read()
        updateImage(img)
        cv2.imshow("Video", img)
    
    def updateImage(image):
        i = 0
        for x, y, w, h in BOXES:
            preprocess_image = preprocess(image[y:h, x:w])
            mseval = mse(preprocess_image, BOXIMAGES[i])
            if i == 5:
                mseval = (mseval + mse(preprocess_image, BOXIMAGES[i+1])) / 2
            color = (0,255,0)
            if mseval > THRESHOLDS[i]:
                color = (0, 0, 255)
                counts[i] += 1
            cv2.imshow('ROI' + str(i+1), preprocess_image)
            cv2.rectangle(image, (x, y), (w, h), color, 2)
            cv2.putText(image,"{:.2f} Count: ".format(mseval) + str(counts[i]), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    cv2.namedWindow('Video')
    cv2.createTrackbar( 'start', 'Video', 0, length, onChange )

    onChange(0)
    cv2.waitKey()

    start = cv2.getTrackbarPos('start','Video')
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    while cap.isOpened():
        valid, image = cap.read()
        if not valid:
            break

        video_frames_count += 1

        image = cv2.resize(image, (1024, 1024))

        updateImage(image)
        cv2.imshow('Video',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(counts)
    print('Number of frames', video_frames_count)
        

video('video.avi')
