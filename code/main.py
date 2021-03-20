import cv2 as cv
import numpy as np
import os
# import matplotlib.pyplot as plt

initial_img_path = ''
result = []


def Sort(final_list):
    return sorted(final_list,
                  reverse=True,
                  key=lambda x: len(x[4]))


def BruteForceMatcher(img1, kp1, des1, img2, kp2, des2):
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    findHomography(img1, kp1, img2, kp2, matches)


def FlannMatcher(img1, kp1, des1, img2, kp2, des2):
    # create FLANNMatcher object
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                        trees=5)
    search_params = dict(checks=50)
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    findHomography(img1, kp1, img2, kp2, matches)


def findHomography(img1, kp1, img2, kp2, matches):
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 50

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # if (M is not None):
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        result.append([img1, kp1, img2, kp2, good, draw_params, initial_img_path])
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))


def SIFTMatch(img1, kp1, des1, img_path2, matcher):
    img2 = cv.imread(img_path2, cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)
    if matcher == '1':
        BruteForceMatcher(img1, kp1, des1, img2, kp2, des2)
    else:
        FlannMatcher(img1, kp1, des1, img2, kp2, des2)


def ORBMatch(img1, kp1, des1, img_path2, matcher):
    img2 = cv.imread(img_path2, cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp2, des2 = orb.detectAndCompute(img2, None)
    if matcher == '1':
        BruteForceMatcher(img1, kp1, des1, img2, kp2, des2)
    else:
        FlannMatcher(img1, kp1, des1, img2, kp2, des2)


if __name__ == "__main__":

    descriptorsFinder = input('Choose a descriptor finder:\nORB (1)\nSIFT (2)\n')
    matcher = input('Choose a descriptor matcher:\nBrute-Force (1)\nFLANN (2)\n')
    query_img = input('A path of a query image\n')

    fds = sorted(os.listdir('oxbuild_images/'))

    if ((matcher == '1') or (matcher == '2')) and ((descriptorsFinder == '1') or (descriptorsFinder == '2')):
        if descriptorsFinder == '1':
            detector = cv.ORB_create()
            match = ORBMatch
        else:
            detector = cv.SIFT_create()
            match = SIFTMatch
        img1 = cv.imread(query_img, cv.IMREAD_GRAYSCALE)  # queryImage
        detector.detectAndCompute(img1, None)
        kp1, des1 = detector.detectAndCompute(img1, None)
        for img in fds[:200]:
            if img.endswith('.jpg'):
                print(img)
                initial_img_path = img
                match(img1, kp1, des1, 'oxbuild_images/' + img, matcher)

        result = Sort(result)
        output_file = open('ranked_list.txt', 'w')
        for index in range(len(result)):
            output_file.write(result[index][6] + '\n')
        output_file.close()