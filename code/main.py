import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
from typing import List

MIN_MATCH_COUNT = 30
initial_img_path = ''
result = []



def load_list(fname: str):
    """Plain text list loader. Reads from file separated by newlines, and returns a
    list of the file with whitespaces stripped.
    Args:
        fname (str): Name of file to be read.
    Returns:
        List[str]: A stripped list of strings, using newlines as a seperator from file.
    """

    return [e.strip() for e in open(fname, 'r').readlines()]


def compute_ap(pos: List[str], amb: List[str], ranked_list: List[str]):
    """Compute average precision against a retrieved list of images. There are some bits that
    could be improved in this, but is a line-to-line port of the original C++ benchmark code.
    Args:
        pos (List[str]): List of positive samples. This is normally a conjugation of
        the good and ok samples in the ground truth data.
        amb (List[str]): List of junk samples. This is normally the junk samples in
        the ground truth data. Omitting this makes no difference in the AP.
        ranked_list (List[str]): List of retrieved images from query to be evaluated.
    Returns:
        float: Average precision against ground truth - range from 0.0 (worst) to 1.0 (best).
    """

    intersect_size, old_recall, ap = 0.0, 0.0, 0.0
    old_precision, j = 1.0, 1.0

    for e in ranked_list:
        if e in amb:
            continue

        if e in pos:
            intersect_size += 1.0

        recall = intersect_size / len(pos)
        precision = intersect_size / j
        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

        old_recall = recall
        old_precision = precision
        j += 1.0

    return ap


def compute(query):
    try:
        ranked_list = []
        for x in range(len(result)):
            ranked_list.append(result[x][6])
        pos_set = list(set(load_list("%s_good.txt" % ('gt_files_170407/'+query))
                           + load_list("%s_ok.txt" % ('gt_files_170407/'+query))))
        junk_set = load_list("%s_junk.txt" % ('gt_files_170407/'+query))

        print(compute_ap(pos_set, junk_set, ranked_list))
    except IOError as e:
        print("IO error while opening files. %s" % e)
        sys.exit(1)


def Sort(final_list):
    return sorted(final_list,
                  reverse=True,
                  key=lambda x: len(x[4]))


def BruteForceMatcher(img1, kp1, des1, img2, kp2, des2):
    start = time.time()
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    findHomography(img1, kp1, img2, kp2, matches)
    return time.time() - start


def FlannMatcher(img1, kp1, des1, img2, kp2, des2):
    start = time.time()
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
    return time.time() - start


def findHomography(img1, kp1, img2, kp2, matches):
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
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
    start = time.time()
    img2 = cv.imread(img_path2, cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)

    """
    file = img_path1 + '_orb.txt'
    np.savetxt(file, des1, fmt='%3.0d')
    des1 = np.loadtxt(file, dtype=np.float32)
    file = img_path2 + '_orb.txt'
    np.savetxt(file, des1, fmt='%3.0d')
    des2 = np.loadtxt(file, dtype=np.float32) 
    """
    t = time.time() - start
    if matcher == '1':
        t_match = BruteForceMatcher(img1, kp1, des1, img2, kp2, des2)
    else:
        t_match = FlannMatcher(img1, kp1, des1, img2, kp2, des2)
    return t, t_match


def ORBMatch(img1, kp1, des1, img_path2, matcher):
    start = time.time()
    img2 = cv.imread(img_path2, cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp2, des2 = orb.detectAndCompute(img2, None)

    """
    file = img_path1 + '_orb.txt'
    np.savetxt(file, des1, fmt='%3.0d')
    des1 = np.loadtxt(file, dtype=np.uint8)
    file = img_path2 + '_orb.txt'
    np.savetxt(file, des2, fmt='%3.0d')
    des2 = np.loadtxt(file, dtype=np.uint8)
    """

    """
    # print the amount of keypoints and descriptors
    print("descriptors' size is", des1.shape[1], "in the first image")
    print("descriptors' size is", des2.shape[1], "in the second image")
    print("there are", len(kp1), "keypoints in the first image")
    print("there are", len(kp2), "keypoints in the second image")
    """
    t = time.time() - start
    if matcher == '1':
        t_match = BruteForceMatcher(img1, kp1, des1, img2, kp2, des2)
    else:
        t_match = FlannMatcher(img1, kp1, des1, img2, kp2, des2)
    return t, t_match


if __name__ == "__main__":
    average_time_des = 0
    average_time_matcher = 0

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
        for img in fds:
            if img.endswith('.jpg'):
                print(img)
                tmp_1, tmp_2 = match(img1, kp1, des1, 'oxbuild_images/' + img, matcher)
                initial_img_path = img
                average_time_des += tmp_1
                average_time_matcher += tmp_2

        if descriptorsFinder == '1':
            img1 = cv.imread(query_img, cv.IMREAD_GRAYSCALE)  # queryImage
            orb = cv.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1, None)
            for img in fds:
                if img.endswith('.jpg'):
                    print(img)
                    tmp_1, tmp_2 = ORBMatch(img1, kp1, des1, 'oxbuild_images/'+img, matcher)
                    initial_img_path = img
                    average_time_des += tmp_1
                    average_time_matcher += tmp_2
        else:
            img1 = cv.imread(query_img, cv.IMREAD_GRAYSCALE)  # queryImage
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            for img in fds:
                if img.endswith('.jpg'):
                    print(img)
                    try:

                        tmp_1, tmp_2 = SIFTMatch(img1, kp1, des1, 'oxbuild_images/'+img, matcher)
                        initial_img_path = img
                        average_time_des += tmp_1
                        average_time_matcher += tmp_2
                    except cv.error:
                        print('Error {}'.format(img))

    print('Average time descriptors and keypoints finder works: %s' % (average_time_des/len(fds)*2),
          '\nAverage time matcher works: %s' % (average_time_matcher/len(fds)))

    result = Sort(result)
    for x in range(len(result)):
        img3 = cv.drawMatches(result[x][0], result[x][1], result[x][2], result[x][3], result[x][4], None,
                              **result[x][5])
        plt.title('Top %d' % (x+1),
                  fontweight="bold")
        plt.imshow(img3, 'gray'), plt.show()
    compute(query_img[:-4].replace('00000', ''))
    """       
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(i, None)
    for _ in range(100):
        d = des1
        kp1, des1 = orb.detectAndCompute(img1, None)
        if d.shape != des1.shape or not np.all(d == des1):
            print("error")
    """