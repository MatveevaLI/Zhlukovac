import copy
import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np

colors = ['darkgrey', 'rosybrown', 'darkred', 'orangered', 'peru', 'brown', 'olive',
          'darkorange', 'goldenrod', 'darkseagreen', 'teal', 'steelblue', 'slateblue', 'plum', 'crimson']


def visualisation(k, clusters, medoids):
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")

    x_coordinates = []
    y_coordinates = []

    cl_coor_x = []
    cl_coor_y = []

    for cl in range(k):
        for i in range(len(clusters)):
            if clusters[i][2] == cl:
                cl_coor_x.append(clusters[i][0])
                cl_coor_y.append(clusters[i][1])

        plt.scatter(cl_coor_x, cl_coor_y, s=30, c=colors[cl])
        cl_coor_x.clear()
        cl_coor_y.clear()

    for m in range(k):
        cl_coor_x.append(medoids[m][0])
        cl_coor_y.append(medoids[m][1])

    plt.scatter(cl_coor_x, cl_coor_y, s=45, c='black')
    plt.show(block=False)
    random_name = random.randint(0, 10000)
    plt.savefig(str(random_name))
    plt.pause(55)
    plt.close()
    return


def generate_all_points():
    first_20_points = generate_20_points()
    points = [x for x in first_20_points]

    counter = 20
    while counter < 200:
        # vybrat jeden z tych bodov co uz existuju
        point = random.choice(points)

        x = point[0] + random.randint(-100, 100)
        y = point[1] + random.randint(-100, 100)
        new_point = [x, y, 0]
        if new_point in points:
            continue

        points.append(new_point)
        counter += 1
    return points


def generate_20_points():
    counter = 0
    first_20_points = []

    while (counter < 20):
        x = random.randint(-5000, 5000)
        y = random.randint(-5000, 5000)
        new_point = [x, y, 0]

        if new_point in first_20_points:
            continue

        first_20_points.append([x, y, 0])
        counter += 1

    return first_20_points


def update_centroid(points, centroids):
    for i in range(len(centroids)):
        avr_x = 0
        avr_y = 0
        avr_cnr = 0
        for k in points:
            if k[2] == i:
                avr_x += k[0]
                avr_y += k[1]
                avr_cnr += 1

        if (avr_cnr == 0):
            continue
        centroids[i][0] = int(avr_x / avr_cnr)
        centroids[i][1] = int(avr_y / avr_cnr)

    return centroids


def update_medoid(points, medoids):
    min = np.iinfo(np.int32).max
    avr = 0
    index = 0
    # pre kazdy bod z klasteru prepocitat euklidovsku vzdialenost
    for m in range(len(medoids)):
        for p in range(len(points)):
            if points[p][2] == m:
                for point in range(len(points)):
                    if points[p][2] == points[point][2]:
                        if points[p] == points[point]:
                            continue
                        avr += int(math.dist((points[p][0], points[p][1]), (points[point][0], points[point][1])))
                        if avr > min:
                            break
                if avr < min:
                    index = p
                    min = avr
                avr = 0
        medoids[m] = points[index]
        min = np.iinfo(np.int32).max
    return medoids


def k_means(k, data_set, method):
    points = copy.deepcopy(data_set)
    centroids = []
    min_dist = []

    # vygenerovat centroid/medoid
    for i in range(k):
        random_centroid = random.randint(0, len(points) - 1)
        new_centroid = points[random_centroid][0], points[random_centroid][1]

        while new_centroid in centroids:
            random_centroid = random.randint(0, len(points) - 1)

        centroids.append([points[random_centroid][0], points[random_centroid][1]])

    for centroid in range(len(centroids)):
        centroids[centroid].append(centroid)
    c = 1

    # definovat klaster prvykrat
    for point in points:

        for centroid in range(len(centroids)):
            d = int(math.dist((point[0], point[1]), (centroids[centroid][0], centroids[centroid][1])))
            min_dist.append(d)

        point[2] = min_dist.index(min(min_dist))

        min_dist.clear()

    # prepocitat klaster pre kazdy bod
    if method == "centroid":
        update = True
        while update:
            update = False
            centroids = update_centroid(points, centroids)
            for point in points:
                for centroid in range(len(centroids)):
                    d = int(math.dist((point[0], point[1]), (centroids[centroid][0], centroids[centroid][1])))
                    min_dist.append(d)

                if point[2] != min_dist.index(min(min_dist)):
                    point[2] = min_dist.index(min(min_dist))
                    update = True

                min_dist.clear()

        return points, centroids

    if method == "medoid":

        medoids = [x for x in centroids]
        del centroids
        update = True
        while update:
            update = False

            medoids = update_medoid(points, medoids)
            for point in points:
                for medoid in range(len(medoids)):
                    d = int(math.dist((point[0], point[1]), (medoids[medoid][0], medoids[medoid][1])))
                    min_dist.append(d)

                if point[2] != min_dist.index(min(min_dist)):
                    point[2] = min_dist.index(min(min_dist))
                    update = True

                min_dist.clear()

    return points, medoids


# k-means funkcia pre divizne
# rozdiel je ze sa negeneruje centroid
def k_means_div(k, points, init, centroids):
    min_dist = []
    if init == True:
        for point in points:

            for centroid in range(len(centroids)):
                d = int(math.dist((point[0], point[1]), (centroids[centroid][0], centroids[centroid][1])))
                min_dist.append(d)

            point[2] = min_dist.index(min(min_dist))

            min_dist.clear()

    # prepocitat klaster pre kazdy bod
    update = True
    while update:
        update = False
        centroids = update_centroid(points, centroids)
        for point in points:
            for centroid in range(len(centroids)):
                d = int(math.dist((point[0], point[1]), (centroids[centroid][0], centroids[centroid][1])))
                min_dist.append(d)

            if point[2] != min_dist.index(min(min_dist)):
                point[2] = min_dist.index(min(min_dist))
                update = True

            min_dist.clear()

    return points


# najst body ktore maju najvacsiu vzdialenost
def init_centroids(points):
    matrix = np.full((len(points), len(points)), -1)
    for p in range(len(points)):
        for m in range(len(points)):
            if points[p][0] != points[m][0] and points[p][1] != points[m][1] and (
                    matrix[p][m] == -1 or matrix[m][p] == -1):
                dist = int(math.dist((points[p][0], points[p][1]), (points[m][0], points[m][1])))
                matrix[p][m] = dist
                matrix[m][p] = dist
            else:
                continue

    max = np.argwhere(matrix == np.max(matrix))[0]

    centroid_1 = points[max[0]]
    centroid_2 = points[max[1]]

    del matrix

    return centroid_1, centroid_2


# najst vzdialenost medzi bodmi v klasteroch
def find_dist(clusters):
    max = 0
    num_cl = 0

    for c in range(len(clusters)):
        matrix = np.full((len(clusters[c]), len(clusters[c])), -1)
        for p in range(len(clusters[c])):
            for m in range(len(clusters[c])):
                if clusters[c][p][0] != clusters[c][m][0] and clusters[c][p][1] != clusters[c][m][1] and (
                        matrix[p][m] == -1 or matrix[m][p] == -1):
                    dist = int(
                        math.dist((clusters[c][p][0], clusters[c][p][1]), (clusters[c][m][0], clusters[c][m][1])))
                    matrix[p][m] = dist
                    matrix[m][p] = dist
                else:
                    continue

        max_tmp = np.max(matrix)
        if max_tmp > max:
            max = max_tmp
            num_cl = c
            del matrix
            continue
        else:
            del matrix
            continue

    return num_cl


def divisive(k, data_set):
    points = copy.deepcopy(data_set)
    centroids = []
    # inicializavat centroidy
    centroid_1, centroid_2 = init_centroids(points)
    clusters = []
    centroids.append(centroid_1)
    centroids.append(centroid_2)

    # vypocitat prve dva klastre
    points = k_means_div(2, points, True, centroids)

    for c in range(len(centroids)):
        clusters.append([])
        for p in range(len(points)):
            if points[p][2] == c:
                clusters[c].append(points[p])
    centroids.clear()

    while len(clusters) < k:
        # najst klaster s najvacsej vzdialenostou
        tmp_clr_num = find_dist(clusters)
        tmp_clr = []
        tmp_clr = copy.deepcopy(clusters[tmp_clr_num])
        centroid_1, centroid_2 = init_centroids(tmp_clr)
        centroids.append(centroid_1)
        centroids.append(centroid_2)

        # rozdelit tento klaster
        tmp_clr = k_means_div(2, tmp_clr, True, centroids)
        del clusters[tmp_clr_num]

        for c in range(2):
            clusters.append([])
            for p in range(len(tmp_clr)):
                if tmp_clr[p][2] == c:
                    clusters[-1].append(tmp_clr[p])
        centroids.clear()

    for p in range(len(points)):
        for c in range(len(clusters)):
            for k in range(len(clusters[c])):
                if points[p][0] in clusters[c][k] and points[p][1] in clusters[c][k]:
                    points[p][2] = c
                    break
    centroids.clear()
    for c in range(len(clusters)):
        centroids.append([0, 0])
    centroids = update_centroid(points, centroids)

    distance = []
    for p in range(len(points)):
        for c in range(len(clusters)):
            dist = int(math.dist((centroids[c][0], centroids[c][1]), (points[p][0], points[p][1])))
            distance.append(dist)

        points[p][2] = np.argwhere(distance == np.min(distance))[0][0]

        distance.clear()

    centroids = update_centroid(points, centroids)

    return points, centroids


def agglomerative(k, data_set):
    num_clasters = len(data_set)
    points = copy.deepcopy(data_set)
    clusters = []

    matrix = np.full((len(data_set), len(data_set)), -1)
    clusters = copy.deepcopy(points)

    for c in range(len(clusters)):
        clusters[c][2] = 1

    for i in range(len(points)):
        points[i][2] = i

    # naplnit maticu susednosti
    for p in range(len(points)):
        for m in range(len(points)):
            if points[p][0] != points[m][0] and points[p][1] != points[m][1] and (
                    matrix[p][m] == -1 or matrix[m][p] == -1):
                dist = int(math.dist((points[p][0], points[p][1]), (points[m][0], points[m][1])))
                matrix[p][m] = dist
                matrix[m][p] = dist
            else:
                continue

    np.fill_diagonal(matrix, np.iinfo(np.int32).max)

    while (num_clasters > k):
        min = np.argwhere(matrix == np.min(matrix))[0]
        matrix, points, num_clasters, clusters = merge_clusters(min, points, matrix, num_clasters, clusters)

    for p in range(len(points)):
        for c in range(len(clusters)):
            if points[p][0] in clusters[c] and points[p][1] in clusters[c]:
                points[p][2] = c
                break

    centroids = []

    for n in range(len(clusters)):
        centroids.append([clusters[n][0], clusters[n][1]])

    distance = []
    for p in range(len(points)):
        for c in range(len(clusters)):
            dist = int(math.dist((centroids[c][0], centroids[c][1]), (points[p][0], points[p][1])))
            distance.append(dist)

        points[p][2] = np.argwhere(distance == np.min(distance))[0][0]

        distance.clear()

    centroids = update_centroid(points, centroids)
    return points, centroids


def merge_clusters(min, points, matrix, num_clasters, clusters):
    avr_x = 0
    avr_y = 0
    cl_num = 0

    cl_1 = []
    cl_2 = []

    cl_1 = clusters[min[0]]
    cl_2 = clusters[min[1]]

    while True:
        # ak v 2 clasteroch je len 1 bod
        if cl_1[2] == 1 and cl_2[2] == 1:
            avr_x = int((cl_1[0] + cl_2[0]) / 2)
            avr_y = int((cl_1[1] + cl_2[1]) / 2)
            new_cl = [avr_x, avr_y, 2, cl_1[0], cl_1[1], cl_2[0], cl_2[1]]
            break

        # ak v 1 klastere viac ako 1 bod, v 2 je jeden
        if cl_1[2] > 1 and cl_2[2] == 1:

            avr_x, avr_y = average(cl_1, cl_2, num_clasters)
            new_cl = [avr_x, avr_y, cl_1[2] + 1, cl_2[0], cl_2[1]]

            for k in range(3, len(cl_1)):
                new_cl.append(cl_1[k])
            break

        # ak v 2 klastere viac ako 1 bod, v 1 je jeden
        if cl_2[2] > 1 and cl_1[2] == 1:
            avr_x, avr_y = average(cl_1, cl_2, num_clasters)
            new_cl = [avr_x, avr_y, cl_2[2] + 1, cl_1[0], cl_1[1]]

            for k in range(3, len(cl_2)):
                new_cl.append(cl_2[k])
            break

        # v 2-och klasteroch je viac bodov ako 1
        if cl_1[2] > 1 and cl_2[2] > 1:
            avr_x, avr_y = average(cl_1, cl_2, num_clasters)
            new_cl = [avr_x, avr_y, cl_1[2] + cl_2[2]]

            for k in range(3, len(cl_1)):
                new_cl.append(cl_1[k])

            for m in range(3, len(cl_2)):
                new_cl.append(cl_2[m])
            break

    # vymazat z matice susednosti
    if min[0] > min[1]:
        matrix = np.delete(matrix, min[0], 0)
        matrix = np.delete(matrix, min[1], 0)
        matrix = np.delete(matrix, min[0], 1)
        matrix = np.delete(matrix, min[1], 1)
        del clusters[min[0]]
        del clusters[min[1]]
    else:
        matrix = np.delete(matrix, min[1], 0)
        matrix = np.delete(matrix, min[0], 0)
        matrix = np.delete(matrix, min[1], 1)
        matrix = np.delete(matrix, min[0], 1)
        del clusters[min[1]]
        del clusters[min[0]]

    num_clasters -= 1

    # pridat novy klaster a vypocitat vzdialenost
    matrix = np.append(matrix, np.array([[np.uint8() for _ in range(len(matrix))]]), 0)
    matrix = np.append(matrix, np.array([[np.uint8()] for _ in range(len(matrix))]), 1)

    for p in range(len(clusters)):
        dist = int(math.dist((clusters[p][0], clusters[p][1]), (new_cl[0], new_cl[1])))
        matrix[-1][p] = dist
        matrix[p][-1] = dist

    clusters.append(new_cl)
    np.fill_diagonal(matrix, np.iinfo(np.int32).max)
    return matrix, points, num_clasters, clusters


# najst stred medzi klastrami ktore budu spojene
def average(cl_1, cl_2, cl_a):
    avr_x = 0
    avr_y = 0
    cl_1_size = cl_1[2]
    cl_2_size = cl_2[2]

    if cl_a == 5:
        print()

    if cl_1_size == 1 or cl_2_size == 1:
        if cl_1_size == 1:
            avr_x += cl_1[0]
            avr_y += cl_1[1]

            for i in range(3, len(cl_2)):
                if i % 2 == 0:
                    avr_y += cl_2[i]
                else:
                    avr_x += cl_2[i]

        if cl_2_size == 1:
            avr_x += cl_2[0]
            avr_y += cl_2[1]

            for i in range(3, len(cl_1)):
                if i % 2 == 0:
                    avr_y += cl_1[i]
                else:
                    avr_x += cl_1[i]

        avr_x = int((avr_x) / (cl_2_size + cl_1_size))
        avr_y = int((avr_y) / (cl_2_size + cl_1_size))

        return avr_x, avr_y

    for i in range(3, len(cl_1)):
        if i % 2 == 0:
            avr_y += cl_1[i]
        else:
            avr_x += cl_1[i]

    for m in range(3, len(cl_2)):
        if m % 2 == 0:
            avr_y += cl_2[m]
        else:
            avr_x += cl_2[m]

    avr_x = int((avr_x) / (cl_2_size + cl_1_size))
    avr_y = int((avr_y) / (cl_2_size + cl_1_size))

    return avr_x, avr_y


# pre vsetky klastre vypocitat priemernu vzdialenost od stredu
def test(points, centroids):
    distance = 0
    counter = 0
    avr_all = 0

    for c in range(len(centroids)):
        for p in range(len(points)):
            if points[p][2] == c:
                counter += 1
                distance += int(math.dist((centroids[c][0], centroids[c][1]), (points[p][0], points[p][1])))
        print(f"avr. dist: {distance / counter}")
        avr_all += (distance / counter)
        distance = 0
        counter = 0
    avr_all = avr_all / len(centroids)
    print(f"avr. dist all: {avr_all}")


print("1 K-means with centroid\n"
      "2 K-means with medoid\n"
      "3 Divisive clustering\n"
      "4 Agglomerative clustering\n"
      "5 All\n")
option = input("Algoritm: ")
data_set = generate_all_points()

k = 15
while True:
    if option == "1" or option == "5":
        start_time = time.time()
        points, centroids = k_means(k, data_set, "centroid")
        end_time = time.time()
        print("K-means centroid time finish:", round((end_time - start_time) / 60, 3), "min")
        visualisation(k, points, centroids)
        if option == "1":
            break

    if option == "2" or option == "5":
        start_time = time.time()
        points, medoids = k_means(k, data_set, "medoid")
        end_time = time.time()
        print("K-means medoid time finish:", round((end_time - start_time) / 60, 3), "min")
        visualisation(k, points, medoids)
        if option == "2":
            break

    if option == "3" or option == "5":
        start_time = time.time()
        points_a, centroids = agglomerative(k, data_set)
        # centroids = update_centroid(points_a, centroids)
        end_time = time.time()
        print("Agglomarative time finish:", round((end_time - start_time) / 60, 3), "min")
        visualisation(k, points_a, centroids)
        if option == "3":
            break

    if option == "4" or option == "5":
        start_time = time.time()
        points_d, centroids = divisive(k, data_set)
        # centroids = update_centroid(points_d, centroids)
        end_time = time.time()
        print("Divisive time finish:", round((end_time - start_time) / 60, 3), "min")
        visualisation(k, points_d, centroids)
        break

    # test(points_d, centroids)
