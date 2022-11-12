import cv2
import numpy as np

BOXES = [
    [502, 344, 502 + 61, 344 + 86],
    [627, 427, 627 + 193, 427 + 87],
    [522, 654, 522 + 90, 654 + 85],
    [704, 594, 704 + 90, 594 + 80],
    [545, 577, 545 + 81, 577 + 75],
    [460, 700, 460 + 52, 700 + 80],
]


BOXIMAGES = [
    cv2.imread('BOX1.jpg'),
    cv2.imread('BOX2.jpg'),
    cv2.imread('BOX3.jpg'),
    cv2.imread('BOX4.jpg'),
    cv2.imread('BOX5.jpg'),
    cv2.imread('BOX6_1.jpg'),
    cv2.imread('BOX6_2.jpg')
]

THRESHOLDS = [50, 63.5, 60, 30, 60, 62]

def mse(imageA, imageB):
	err = np.sum((np.asarray(imageA) - np.asarray(imageB)) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	return err

def preprocess(image):
    blur = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    return cv2.threshold(blur, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

def video(path):
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def onChange(trackbarValue):
        cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
        _, img = cap.read()
        cv2.imshow("Video", img)

    cv2.namedWindow('Video')
    cv2.createTrackbar( 'start', 'Video', 0, length, onChange )

    onChange(0)
    cv2.waitKey()

    start = cv2.getTrackbarPos('start','Video')
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    while cap.isOpened():
        _, image = cap.read()
        image = cv2.resize(image, (1024, 1024))

        i = 0
        for x, y, w, h in BOXES:
            preprocess_image = preprocess(image[y:h, x:w])
            preprocess_box = preprocess(BOXIMAGES[i])
            mseval = mse(preprocess_image, preprocess_box)
            if i == 5:
                mseval = (mseval + mse(preprocess_image, preprocess_box)) / 2
            color = (0,255,0)
            if mseval > THRESHOLDS[i]:
                color = (255, 0, 0)
            cv2.rectangle(image, (x, y), (w, h), color, 2)
            cv2.putText(image,"{:.2f}".format(mseval), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            i += 1
        cv2.imshow('Video',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

video('video.avi')