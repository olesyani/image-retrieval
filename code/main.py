import cv2 as cv
import numpy as np
import time
import yaml
import pickle
import os
import compute_ap


class KeypointsAndDescriptors:
    def __init__(self):
        self.keypoints = []
        self.descriptors = []

    def read_keypoints(self, des_finder, path):
        # Считываем ключевые точки из файла
        file = 'data/' + des_finder + '/' + path + '_kp.txt'
        f = open(file, 'rb')
        index = pickle.loads(f.read())
        f.close()
        for point in index:
            tmp = cv.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                              _response=point[3], _octave=point[4], _class_id=point[5])
            self.keypoints.append(tmp)

    def read_descriptors(self, des_finder, path):
        # Считываем дескрипторы из файла
        file = 'data/' + des_finder + '/' + path + '_des.txt'
        self.descriptors = np.loadtxt(file).astype('float32')

    def write_keypoints_and_descriptors(self, des_finder, path):
        if des_finder == 'orb':
            detector = cv.ORB_create()
        elif des_finder == 'sift':
            detector = cv.SIFT_create()
        else:
            raise NotImplementedError('Wrong value of descriptor')

        initial_img = cv.imread('oxbuild_images/' + path, cv.IMREAD_GRAYSCALE)
        kp, des = detector.detectAndCompute(initial_img, None)
        index = []
        for point in kp:
            temp = (point.pt, point.size, point.angle, point.response, point.octave,
                    point.class_id)
            index.append(temp)
        path = path.replace('.jpg', '')

        # Записываем ключевые точки в файл
        file_name = 'data/' + des_finder + '/' + path + '_kp.txt'
        f = open(file_name, "wb")
        f.write(pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL))
        f.close()

        # Записываем дескрипторы точки в файл
        file_name = 'data/' + des_finder + '/' + path + '_des.txt'
        np.savetxt(file_name, des)


def BruteForceMatcher(des1, des2, params, search_params):
    # Создаем BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    return findHomography(matches)


def FlannMatcher(des1, des2, params, search_params):
    # Создаем FLANNMatcher object
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)
    flann = cv.FlannBasedMatcher(params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return findHomography(matches)


def findHomography(matches):
    # Находим хорошие пары
    # Для каждой точки сохраняется два лучших совпадения, если они достаточно разные, это проверяется Lowe's ratio test, приведенным ниже, то считается, что совпадение найдено, иначе считается, что match не найден
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    if len(good) > 0:
        return len(good)

    return None


if __name__ == "__main__":

    cfg = {}
    with open("configs/default.yaml", 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    descriptorsFinder = cfg['descriptor']
    matcher = cfg['matcher']

    fds = sorted(os.listdir('oxbuild_images/'))

    FLANN_ALGORITHM = 0
    index_params = dict()

    if descriptorsFinder == 'orb':
        FLANN_ALGORITHM = 1
        index_params = dict(algorithm=FLANN_ALGORITHM,
                            trees=5)
        search_params = dict(checks=50)
    elif descriptorsFinder == 'sift':
        FLANN_ALGORITHM = 1
        index_params = dict(algorithm=FLANN_ALGORITHM,
                            trees=5)
        search_params = dict(checks=50)
    else:
        raise NotImplementedError('Wrong value of descriptor')

    if matcher == 'brute-force':
        matching = BruteForceMatcher
    elif matcher == 'flann':
        matching = FlannMatcher
    else:
        raise NotImplementedError('Wrong value of descriptor')

    queries_array = sorted(os.listdir('gt_files_170407/'))
    query_img_array = []  # будет содержать массив из названий query-изображений
    title_array = []  # название файла, в котором содержится название query-изображения

    for file in queries_array:
        if file.endswith('query.txt'):
            f = open('gt_files_170407/' + file)
            # Читаем в файле название query-изображения, так как само название файла его не содержит
            str = f.read()
            index = str.find(' ')
            query_img_array.append(str[5:index] + '.jpg')
            title_array.append(file.replace('_query.txt', ''))

    fds = fds[1:]  # первым будет файл .DS_Store, который не нужен

    # Создаем матрицу, в которой будет содержать кол-во совпадений
    result = [[0] * len(fds) for i in range(len(query_img_array))]
    result = np.array(result)
    img_index = 0

    query_info = []

    start = time.time()
    
    # Чтобы каждый раз заново не читать keypoints и descriptors изображений-запросов, они будут храниться в памяти
    for i in range(len(query_img_array)):
        query_path = query_img_array[i]

        query_image = cv.imread('oxbuild_images/' + query_path, cv.IMREAD_GRAYSCALE)

        query_path = query_path.replace('.jpg', '')
        initial_img = KeypointsAndDescriptors()
        initial_img.read_descriptors(descriptorsFinder, query_path)
        initial_img.read_keypoints(descriptorsFinder, query_path)
        query_info.append([query_path, initial_img.descriptors, len(initial_img.keypoints)])

    for img_path in fds:
        if img_path.endswith('.jpg'):
            print(img_path)

            train_image = cv.imread('oxbuild_images/' + img_path, cv.IMREAD_GRAYSCALE)

            img_path = img_path.replace('.jpg', '')
            initial_img = KeypointsAndDescriptors()
            initial_img.read_descriptors(descriptorsFinder, img_path)
            initial_img.read_keypoints(descriptorsFinder, img_path)

            for i in range(len(query_img_array)):
                tmp = matching(query_info[i][1], initial_img.descriptors, index_params, search_params)

                if tmp is not None:
                    number_keypoints = 0
                    if query_info[i][2] >= len(initial_img.keypoints):
                        number_keypoints = query_info[i][2]
                    else:
                        number_keypoints = len(initial_img.keypoints)

                    # Нас интересует, какой процент ключевых точек от их общего числа схож у двух
                    # изображений. По нему и отсортировываются изображения
                    percentage_similarity = float(tmp) / number_keypoints * 100
                    result[i, img_index] = percentage_similarity

            img_index += 1

    # Выводим среднее время работы алгоритма для одного изображения-запроса
    worktime = (time.time() - start) / len(query_img_array)
    print(worktime)

    # Сортируем таблицу, а затем с помощью метрики подсчитываем, как хорошо работает алгоритм
    sorted_result = np.argsort(result, axis=1)

    # Переменная необходима для того, чтобы в итоге показать среднее значение MAP для всех изображений-запросов
    average_map = 0.0

    # Был создан файл для записи результатов
    with open('experiments_results.txt', 'a') as results_file:
        results_file.write('using ' + descriptorsFinder + ' and ' + matcher + '\n')
        results_file.write('ratio test const = ' + distance_str + '\n')

    for i in range(sorted_result.shape[0]):
        with open('ranked_list.txt', 'w') as output_file:
            for index in reversed(range(0, sorted_result.shape[1])):
                output_file.write(fds[sorted_result[i, index]].replace('.jpg', '') + '\n')

        print('gt_files_170407/' + title_array[i])
        initial_map = compute_ap.compute('gt_files_170407/' + title_array[i])
        average_map = average_map + float(initial_map)
        print(initial_map)
        with open('experiments_results.txt', 'a') as results_file:
            results_file.write(title_array[i] + ': ' + initial_map + '\n')

    print(average_map/len(query_img_array))
