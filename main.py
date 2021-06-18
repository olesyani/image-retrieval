import cv2 as cv
import numpy as np
import time
import pickle
import os
from metrics import compute_ap

DISTANCE = 0.7


class KeypointsAndDescriptors:
    def __init__(self):
        self.keypoints = []
        self.descriptors = []

    def read_keypoints(self, des_finder, path):
        # Считываем ключевые точки из файла
        file = 'data/' + des_finder + '_upd/' + path + '_kp.txt'
        f = open(file, 'rb')
        index = pickle.loads(f.read())
        f.close()
        for point in index:
            tmp = cv.KeyPoint(x=point[0][0],
                              y=point[0][1],
                              _size=point[1],
                              _angle=point[2],
                              _response=point[3],
                              _octave=point[4],
                              _class_id=point[5])
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

        initial_img = cv.imread('data/oxbuild_images/' + path,
                                cv.IMREAD_GRAYSCALE)
        kp, des = detector.detectAndCompute(initial_img, None)
        index = []
        for point in kp:
            temp = (point.pt,
                    point.size,
                    point.angle,
                    point.response,
                    point.octave,
                    point.class_id)
            index.append(temp)
        path = path.replace('.jpg', '')

        # Записываем ключевые точки в файл
        file_name = 'data/' + des_finder + '/' + path + '_kp.txt'
        f = open(file_name, "wb")
        f.write(pickle.dumps(index,
                             protocol=pickle.HIGHEST_PROTOCOL))
        f.close()

        # Записываем дескрипторы точки в файл
        file_name = 'data/' + des_finder + '/' + path + '_des.txt'
        np.savetxt(file_name, des)


def BruteForceMatcher(des1, des2):
    # Создаем BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(np.asarray(des1, np.float32),
                          np.asarray(des2, np.float32),
                          k=2)
    return findHomography(matches)


def FlannMatcher(des1, des2):
    # Создаем FLANNMatcher object
    FLANN_ALGORITHM = 1
    index_params = dict(algorithm=FLANN_ALGORITHM,
                        trees=5)
    search_params = dict(checks=50)
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return findHomography(matches)


def findHomography(matches):
    # Находим хорошие пары
    # Для каждой точки сохраняется два лучших совпадения, если
    # они достаточно разные, это проверяется Lowe's ratio test,
    # приведенным ниже, то считается, что совпадение найдено,
    # иначе считается, что match не найден.
    good = []
    for m, n in matches:
        if m.distance < DISTANCE * n.distance:
            good.append(m)

    if len(good) > 0:
        return len(good)

    return None


def readingQueries(query_path):
    if query_path.endswith('.jpg'):
        # Чтобы каждый раз заново не читать descriptors
        # изображений-запросов, они будут храниться в памяти
        query_path = query_path.replace('.jpg', '')
        initial_img = KeypointsAndDescriptors()
        initial_img.read_descriptors(descriptorsFinder,
                                     query_path)
        num_of_kp = (initial_img.descriptors).shape[0]
        return [query_path, initial_img.descriptors, num_of_kp]


def matchingImages(img_path):
    if img_path.endswith('.jpg'):
        print(img_path)

        list_result = []
        img_path = img_path.replace('.jpg', '')
        initial_img = KeypointsAndDescriptors()
        initial_img.read_descriptors(descriptorsFinder,
                                     img_path)
        num_of_kp = (initial_img.descriptors).shape[0]

        for i in range(len(queries_path_array)):
            tmp = matching(query_info[i][1],
                           initial_img.descriptors)

            if tmp is not None:
                number_keypoints = 0
                if query_info[i][2] >= num_of_kp:
                    number_keypoints = query_info[i][2]
                else:
                    number_keypoints = num_of_kp

                # Нас интересует, какой процент ключевых точек от
                # их общего числа схож у двух изображений.
                # По нему и отсортировываются изображения
                percentage_similarity = float(tmp) / number_keypoints * 100
                list_result.append(percentage_similarity)

            else:
                list_result.append(0.0)

        return list_result


if __name__ == "__main__":

    descriptorsFinder = 'orb'
    matcher = 'brute-force'

    img = sorted(os.listdir('data/oxbuild_images/'))

    if descriptorsFinder != 'orb' and descriptorsFinder != 'sift':
        raise NotImplementedError('Wrong value of descriptor')

    if matcher == 'brute-force':
        matching = BruteForceMatcher
    elif matcher == 'flann':
        matching = FlannMatcher
    else:
        raise NotImplementedError('Wrong value of descriptor')

    queries_array = sorted(os.listdir('metrics/gt_files_170407/'))
    # будет содержать массив из названий query-изображений
    queries_path_array = []
    # массив из названий файлов, в которых содержится query-изображения
    title_array = []

    for file in queries_array:
        if file.endswith('query.txt'):
            f = open('metrics/gt_files_170407/' + file)
            # Читаем в файле название query-изображения,
            # так как само название файла его не содержит
            file_str = f.read()
            index = file_str.find(' ')
            queries_path_array.append(file_str[5:index] + '.jpg')
            title_array.append(file.replace('_query.txt', ''))

    # Создаем матрицу, в которой будет содержать кол-во совпадений
    img_index = 0
    query_info = []
    nested_list_result = []

    start = time.time()
    for path in queries_path_array:
        if path.endswith('.jpg'):
            query_info.append(readingQueries(path))

    for path in img:
        if path.endswith('.jpg'):
            nested_list_result.append(matchingImages(path))

    result = np.array(nested_list_result)

    # Выводим среднее время работы алгоритма для одного
    # изображения-запроса
    worktime = (time.time() - start) / len(queries_path_array)
    print(worktime)

    # Сортируем таблицу, а затем с помощью метрики подсчитываем,
    # как хорошо работает алгоритм
    sorted_result = np.argsort(result, axis=0)

    # Переменная необходима для того, чтобы в итоге показать среднее
    # значение MAP для всех изображений-запросов
    average_map = 0.0

    # Был создан файл для записи результатов
    with open('experiments_results.txt', 'a') as results_file:
        results_file.write('using '
                           + descriptorsFinder
                           + ' and '
                           + matcher
                           + '\n')
        results_file.write('time per query: '
                           + str(worktime)
                           + '\n')
    print(sorted_result)
    print(sorted_result.shape)

    for i in range(sorted_result.shape[1]):
        with open('ranked_list.txt', 'w') as output_file:
            for index in reversed(range(0,
                                        sorted_result.shape[0])):
                output_file.write(img[sorted_result[index,
                                                    i]].replace('.jpg',
                                                                '') + '\n')

        print('gt_files_170407/' + title_array[i])
        initial_map = compute_ap.compute('metrics/gt_files_170407/'
                                         + title_array[i])
        average_map = average_map + float(initial_map)
        print(initial_map)
        with open('experiments_results.txt', 'a') as results_file:
            results_file.write(title_array[i] + ': ' + initial_map + '\n')

    with open('experiments_results.txt', 'a') as results_file:
        tmp = average_map/len(queries_path_array)
        results_file.write('average map: ' + str(tmp) + '\n')

        print(average_map/len(queries_path_array))
