import matplotlib
import matplotlib.pyplot as plt
import random
import time

def squared_distance(a, b):
    s = 0
    for i in range(0, 9):
        s = s + (float(a[i]) - float(b[i])) ** 2
    return s

def potential(data, centroid, clusters, k):
    L = 0
    for i in range(k):
        cen = centroid[i]
        for dp_id in clusters[i]:
            L = L + squared_distance(cen, data[dp_id])
    return L

def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        c = [0] * 9
        for dp_id in clusters[i]:
            for j in range(9):
                c[j] = c[j] + float(data[dp_id][j])
        for j in range(9):
            c[j] = c[j] / len(clusters[i])
        new_centroids.append(c)
    return new_centroids

'''
    centroid contains k vectors
    clusters is a list of list of dp ids: [[], [], ..., []]
'''
def k_means(k, data):
    # initialize
    random.seed(time.time())
    ran_nums = random.sample(range(len(data)), k)
    centroid = []
    for ran_num in ran_nums:
        dp = list(data.values())[ran_num]
        centroid.append(dp)
    update = True
    old_clusters = []
    while update:
        update = False
        clusters = [] * k   # initialize empty clusters
        for i in range(k):
            clusters.append([])
        # assign each point based on distances
        for key in data.keys():
            distance = float('inf')
            assignment = -1
            for i in range(len(centroid)):
                d = squared_distance(data[key], centroid[i])
                if d < distance:
                    distance = d
                    assignment = i
            clusters[assignment].append(key)
        # Compute the value of potential function
        L = potential(data, centroid, clusters, k)
        # if there is assignment changed, update = true
        if clusters != old_clusters:
            update = True
        # update the centroids using the mean value of points
        old_clusters = clusters # save current clusters
        centroid = update_centroids(data, clusters, k)
    print('k = ' + str(k) + ':')
    # print(centroid)
    print(L)
    return L

def preprocess(filepath):
    '''
        Returns a dictionary of data points:
        ['672113': ['7', '5', '6', '10', '4', '10', '5', '3', '1'], ...]
    '''
    data = {}
    f = open(filepath, "r")
    line = f.readline()
    while line:
        data_line = line.split(',')
        data_good = True
        for i in range(len(data_line)):
            if data_line[i] == '?':
                data_good = False
        if data_good:
            data[data_line[0]] = data_line[1:10]
        line = f.readline()
    return data

if __name__ == "__main__":
    filepath = "breast-cancer-wisconsin.data"
    data = preprocess(filepath)
    L = [0] * 7
    for k in range(2, 9):
        L[k - 2] = k_means(k, data)
    print(L)
    # Data for plotting
    k_list = [2, 3, 4, 5, 6, 7, 8]
    fig, ax = plt.subplots()
    ax.plot(k_list, L)

    ax.set(xlabel='K value', ylabel='L(K)',
        title='K-Means Plot')
    ax.grid()
    plt.show()
