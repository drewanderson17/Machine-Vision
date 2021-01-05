import cv2
import numpy as np
from matplotlib import pyplot as plt


def showImage(Img, window_name='image'):
    cv2.imshow(window_name, Img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


comp_left = cv2.imread("computers_left.png")
comp_right = cv2.imread("computers_right.png")


# 1.1 Feature matching
def feature_matching(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(2)
    matches = bf.knnMatch(des1, des2, k=2)

    left_points = []
    right_points = []
    left_points_pt = []
    right_points_pt = []

    listA = []
    for m, n in matches:

        if m.distance < 0.8 * n.distance:  # ratio test
            left_points.append(kp1[m.queryIdx])
            right_points.append(kp2[m.trainIdx])
            left_points_pt.append(kp1[m.queryIdx].pt)
            right_points_pt.append(kp2[m.trainIdx].pt)
            listA.append(m)
    left_points_pt = np.int32(left_points_pt)
    right_points_pt = np.int32(right_points_pt)
    return left_points, right_points, left_points_pt, right_points_pt


left_points, right_points, left_points_pt, right_points_pt = feature_matching(comp_left, comp_right)

left_img = cv2.drawKeypoints(comp_left, left_points, None)
right_img = cv2.drawKeypoints(comp_right, right_points, None)
# showImage(left_img, "left Key points")
# showImage(right_img, "right Key points")


# 1.2 Epipolar Lines Calculation
m, _ = cv2.findFundamentalMat(left_points_pt, right_points_pt, cv2.RANSAC, 5)

lines1 = cv2.computeCorrespondEpilines(left_points_pt.reshape(-1, 1, 2), 2, m)
lines1 = lines1.reshape(-1, 3)

lines2 = cv2.computeCorrespondEpilines(right_points_pt.reshape(-1, 1, 2), 2, m)
lines2 = lines2.reshape(-1, 3)


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1.shape
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        # img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
    # img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# find points in the right image and drawing its line on left image
img5, img6 = drawlines(left_img, right_img, lines1, left_points_pt, right_points_pt)
img3, img4 = drawlines(left_img, right_img, lines2, left_points_pt, right_points_pt)

numpy_horizontal = np.hstack((img5, img3))
numpy_horizontal_concat = np.concatenate((img5, img3), axis=1)
# result = cv2.hconcat(img5,img3)
showImage(numpy_horizontal_concat)

''''
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
'''


# 1.3
def callback(event, x, y, param):
    if event == cv2.EVENT_FLAG_MBUTTON:
        x, y


'''
load and display both images

if event= cv2.EV


def callback(event, x, y, param):
    if event == cv2.EVENT_FLAG_MBUTTON:
        x, y
        call compute eppilines 
        if 
'''
