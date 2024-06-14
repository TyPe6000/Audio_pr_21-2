import numpy as np
import cv2
import matplotlib.pyplot as plt


def score() :
        imgpath = "C:/code/audio_pr/score/"
        imgfile = "emptyscore.jpg"

        img = cv2.imread(imgpath+imgfile, cv2.IMREAD_UNCHANGED)

        cv2.namedWindow("emptyscore", cv2.WINDOW_NORMAL)
        cv2. imshow("emptyscore", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

score()

