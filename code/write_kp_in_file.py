import cv2
import numpy as np
import pickle
import os

fds = sorted(os.listdir('oxbuild_images/'))
for i in range(1):
    if i == 0:
        name = 'orb/'
        detector = cv2.ORB_create()
    else:
        name = 'sift/'
        detector = cv2.SIFT_create()
    for img in fds[:75]:
        if img.endswith('.jpg'):
            print(img)
            initial_img = cv2.imread('oxbuild_images/' + img, cv2.IMREAD_GRAYSCALE)  # trainImage
            kp, des = detector.detectAndCompute(initial_img, None)
            """
            file_name = 'data/' + name + img[:-4] + '_kp.yaml'
            cv_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
            cv_file.write('KEYPOINTS', kp)
            cv_file.write('DESCRIPTORS', des)
            cv_file.release()
            """
            index = []
            for point in kp:
                temp = (point.pt, point.size, point.angle, point.response, point.octave,
                        point.class_id)
                index.append(temp)

            # Dump the keypoints
            file_name = 'data/' + name + img[:-4] + '_kp.txt'
            f = open(file_name, "wb")
            f.write(pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL))
            f.close()

            # Dump the descriptors
            file_name = 'data/' + name + img[:-4] + '_des.txt'
            np.savetxt(file_name, des)
