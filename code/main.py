import cv2 as cv
import numpy as np
import pickle
import os
import yaml
import matplotlib.pyplot as plt
import compute_ap


def Sort(final_list):
    return sorted(final_list,
                  reverse=True,
                  key=lambda x: len(x[4]))


def BruteForceMatcher(img1, kp1, des1, img2, kp2, des2):
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    return findHomography(img1, kp1, img2, kp2, matches)


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
    return findHomography(img1, kp1, img2, kp2, matches)


def findHomography(img1, kp1, img2, kp2, matches):
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        if M is not None:
            dst = cv.perspectiveTransform(pts, M)
            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=None,
                               matchesMask=matchesMask,
                               flags=2)
            # result.append([img1, kp1, img2, kp2, good, draw_params, initial_img_path])
            return [img1, kp1, img2, kp2, good, draw_params]
    # print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    return None


def SIFTMatch(img1, kp1, des1, img_path2, matcher):
    img2 = cv.imread('oxbuild_images/'+img_path2, cv.IMREAD_GRAYSCALE)  # trainImage

    file_name = 'data/sift/' + img_path2[:-4] + '_des.txt'
    des2 = np.loadtxt(file_name).astype('float32')
    file_name = file_name.replace('_des', '_kp')
    f = open(file_name, 'rb')
    index = pickle.loads(f.read())
    f.close()
    kp2 = []
    for point in index:
        temp = cv.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                           _response=point[3], _octave=point[4], _class_id=point[5])
        kp2.append(temp)

    return matcher(img1, kp1, des1, img2, kp2, des2)


def ORBMatch(img1, kp1, des1, img_path2, matcher):
    img2 = cv.imread('oxbuild_images/'+img_path2, cv.IMREAD_GRAYSCALE)  # trainImage

    file_name = 'data/orb/' + img_path2[:-4] + '_des.txt'
    des2 = np.loadtxt(file_name).astype('float32')
    file_name = file_name.replace('_des', '_kp')
    f = open(file_name, 'rb')
    index = pickle.loads(f.read())
    f.close()
    kp2 = []
    for point in index:
        temp = cv.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                           _response=point[3], _octave=point[4], _class_id=point[5])
        kp2.append(temp)

    return matcher(img1, kp1, des1, img2, kp2, des2)


if __name__ == "__main__":
    """
    cfg = {}
    with open("configs/default.yaml", 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    descriptorsFinder = cfg['descriptor']
    matcher = cfg['matcher']
    query_img = cfg['query_path']
    """

    descriptorsFinder = 'orb'
    matcher = 'flann'

    fds = sorted(os.listdir('oxbuild_images/'))

    if descriptorsFinder == 'orb':
        detector = cv.ORB_create()
        match = ORBMatch
    elif descriptorsFinder == 'sift':
        detector = cv.SIFT_create()
        match = SIFTMatch
    else:
        raise NotImplementedError('Wrong value of descriptor')

    if matcher == 'brute-force':
        matcher = BruteForceMatcher
    elif matcher == 'flann':
        matcher = FlannMatcher
    else:
        raise NotImplementedError('Wrong value of descriptor')

    queries_array = sorted(os.listdir('gt_files_170407/'))
    query_img_array = []
    title_array = []

    for file in queries_array:
        if file.endswith('query.txt'):
            f = open('gt_files_170407/'+file)
            str = f.read()
            index = str.find(' ')
            query_img_array.append(str[5:index]+'.jpg')
            title_array.append(file.replace('_query.txt', ''))

    for i in range(len(query_img_array)-6):
        query_img = query_img_array[i+6]
        print(query_img)
        file_name = 'data/' + descriptorsFinder + '/' + query_img[:-4] + '_des.txt'
        des1 = np.loadtxt(file_name).astype('float32')

        file_name = file_name.replace('_des', '_kp')
        f = open(file_name, 'rb')
        index = pickle.loads(f.read())
        f.close()
        kp1 = []
        for point in index:
            temp = cv.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                               _response=point[3], _octave=point[4], _class_id=point[5])
            kp1.append(temp)

        query_img = 'oxbuild_images/' + query_img_array[i]
        img1 = cv.imread(query_img, cv.IMREAD_GRAYSCALE)  # queryImage

        result = []
        for img in fds[1:]:
            if img.endswith('.jpg'):
                # print(img)
                tmp = match(img1, kp1, des1, img, matcher)
                if tmp is not None:
                    tmp.append(img.replace('.jpg', ''))
                    result.append(tmp)

        result = Sort(result)
        output_file = open('ranked_list.txt', 'w')
        for index in range(len(result)):
            output_file.write(result[index][6] + '\n')
        output_file.close()
        compute_ap.compute('gt_files_170407/'+title_array[i])
    """
    for x in range(len(result)):
        img3 = cv.drawMatches(result[x][0], result[x][1], result[x][2], result[x][3],
                              result[x][4], None, **result[x][5])
        plt.title('Top %d' % (x + 1),
                  fontweight="bold")
        plt.imshow(img3, 'gray'), plt.show()
    """
