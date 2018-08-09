import cv2
import os
import scipy
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import json

from collections import Counter

from sklearn.cluster import KMeans
from sklearn import svm
from vlfeat import vl_dsift
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc, precision_recall_curve


BASE_DIR = '/home/sahel/Desktop/vision/hw3/HW3/Data/'
TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
TEST_DIR = os.path.join(BASE_DIR, 'Test')
TRAIN_IMAGES = []
VALIDATION_IMAGES = []
TEST_IMAGES = []
CLASSES = set()


def get_img_vector(img):
    img_vector = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_vector.append(img[i, j]/255.0)
    return img_vector


class MY_IMG:
    def __init__(self, img, class_name, img_name):
        self.img_name = img_name
        self.img = scipy.misc.imresize(img, (256, 256))
        self.vector = get_img_vector(scipy.misc.imresize(self.img, (10, 10)))
        self.original_class_name = class_name
        self.random_estimated_class_name = None
        self.knn_class_name = None
        self.bow_knn_class_name = None
        self.bow_svm_class_name = None
        self.histogram = []


def load_imgs(directory, lst, lst2=None):
    class_names = os.listdir(directory)
    for class_name in class_names:
        if class_name == '.DS_Store':
            continue
        CLASSES.add(class_name)
        class_dir = os.path.join(directory, class_name)
        image_names = os.listdir(class_dir)
        for index, image_name in enumerate(image_names):
            if image_name == '.DS_Store':
                continue
            image = cv2.imread(os.path.join(class_dir, image_name), 0)
            if lst2 is not None and index > 0.7 * len(image_names):
                lst2.append(MY_IMG(image, class_name, image_name))
            else:
                lst.append(MY_IMG(image, class_name, image_name))
    return lst, lst2


def get_y_test():
    y_test = []
    classes = list(CLASSES)
    for image in TEST_IMAGES:
        item = [0] * len(classes)
        item[classes.index(image.original_class_name)] = 1
        y_test.append(item)
    return y_test


def calculate_random_class():
    pr_or_classes = []
    classes = list(CLASSES)
    total_scores = []
    for image in TEST_IMAGES:
        scores = random.sample(range(30), len(classes))
        sum_s = sum(scores)
        for i in range(len(scores)):
            scores[i] = scores[i] * 1.0 / sum_s
        image.random_estimated_class_name = classes[scores.index(max(scores))]
        pr_or_classes.append((image.random_estimated_class_name, image.original_class_name))
        total_scores.append(scores)
    return pr_or_classes, np.array(total_scores)


def BOW(n_clusters=40):
    all_descriptors = []
    for image in TRAIN_IMAGES:
        f, des = vl_dsift(image.img, size=8, step=8)
        for item in des.tolist():
            all_descriptors.append(item)

    kmeans = KMeans(n_clusters=n_clusters).fit(np.array(all_descriptors))
    clusters_mean = kmeans.cluster_centers_.tolist()
    return clusters_mean


def get_histogram(des, clusters_mean):
    img_dises = []
    histogram = Counter()
    if des is None:
        return histogram
    for item in des.tolist():
        for index, cluster_mean in enumerate(clusters_mean):
            dis = distance.euclidean(item, cluster_mean)
            img_dises.append((dis, index))
        img_dises = sorted(img_dises, key=lambda tup: tup[0])
        histogram[img_dises[0][1]] += 1
    return histogram


def calculate_histograms(images, clusters_mean):
    for image in images:
        f, des = vl_dsift(image.img, size=8, step=8)
        histogram = get_histogram(des, clusters_mean)
        his_sum = sum(histogram.itervalues()) * 1.0
        for item in range(len(clusters_mean)):
            image.histogram.append(histogram[item]/his_sum if his_sum != 0 else histogram[item])
    return images


def _bow_svm(IMAGES, gamma, class_num, X, y):
    pr_or_classes = []
    score = 0.0
    clf = svm.SVC(gamma=gamma, probability=True)
    clf.fit(X, y)
    total_scores = []
    for validation_image in IMAGES:
        validation_image.bow_svm_class_name = class_num[clf.predict([validation_image.histogram])[0]]
        scores = clf.predict_proba([validation_image.histogram])[0]
        total_scores.append(scores)
        pr_or_classes.append((validation_image.bow_svm_class_name, validation_image.original_class_name))
        if validation_image.bow_svm_class_name == validation_image.original_class_name:
            score += 1
    score /= len(VALIDATION_IMAGES)
    return score, pr_or_classes, np.array(total_scores)


def BOW_SVM():
    clusters_mean = BOW()

    class_num = set()
    for train_image in VALIDATION_IMAGES:
        class_num.add(train_image.original_class_name)
    class_num = list(class_num)
    class_num_dict = {}
    for index, class_name in enumerate(class_num):
        class_num_dict[class_name] = index

    calculate_histograms(VALIDATION_IMAGES, clusters_mean)
    X = []
    y = []
    calculate_histograms(TRAIN_IMAGES, clusters_mean)
    for train_image in TRAIN_IMAGES:
        X.append(train_image.histogram)
        y.append(class_num_dict[train_image.original_class_name])

    scores = Counter()
    for i in range(-6, 5):
        gamma = 10**i
        score, pr_or_classes, totla_scores = _bow_svm(VALIDATION_IMAGES, gamma, class_num, X, y)
        scores[gamma] = score
    return scores.most_common()[0][0], clusters_mean, class_num, X, y


def get_KNN(img_dises, k):
    img_dises = sorted(img_dises, key=lambda tup: tup[0])
    class_cnt = Counter()
    class_index = Counter()
    scores = []
    for index, (img_dis, class_name) in enumerate(img_dises[:k]):
        class_cnt[class_name] += 1
        class_index[class_name] += index
    for class_name in CLASSES:
        scores.append(class_cnt[class_name])
    sum_s = sum(scores)
    for i in range(len(scores)):
        scores[i] = scores[i] * 1.0 / sum_s
    class_cnt = class_cnt.most_common()
    nearest_class = class_cnt[0][0]
    return nearest_class, scores


def _bow_knn(IMAGES, k):
    score = 0.0
    pr_or_classes = []
    total_scores = []
    for validation_image in IMAGES:
        img_dises = []
        for train_image in TRAIN_IMAGES:
            dis = distance.euclidean(validation_image.histogram, train_image.histogram)
            img_dises.append((dis, train_image.original_class_name))
        validation_image.bow_knn_class_name, scores = get_KNN(img_dises, k)
        total_scores.append(scores)
        pr_or_classes.append((validation_image.bow_knn_class_name, validation_image.original_class_name))
        if validation_image.bow_knn_class_name == validation_image.original_class_name:
            score += 1
    score /= len(VALIDATION_IMAGES)
    return score, pr_or_classes, np.array(total_scores)


def BOW_KNN():
    clusters_mean = BOW()
    calculate_histograms(VALIDATION_IMAGES, clusters_mean)
    calculate_histograms(TRAIN_IMAGES, clusters_mean)

    scores = Counter()
    for i in range(1, 10):
        k = 5*i
        score, pr_or_classes, total_scores = _bow_knn(VALIDATION_IMAGES, k)
        scores[k] = score
    return scores.most_common()[0][0], clusters_mean


def _vector_knn(IMAGES, k):
    pr_or_classes = []
    score = 0.0
    total_scores = []
    for validation_image in IMAGES:
        img_dises = []
        scores = []
        for train_img in TRAIN_IMAGES:
            dis = distance.euclidean(validation_image.vector, train_img.vector)
            img_dises.append((dis, train_img.original_class_name))
        img_dises = sorted(img_dises, key=lambda tup: tup[0])
        class_cnt = Counter()
        for img_dis, class_name in img_dises[:k]:
            class_cnt[class_name] += 1

        for class_name in CLASSES:
            scores.append(class_cnt[class_name])
        sum_s = sum(scores)
        for i in range(len(scores)):
            scores[i] = scores[i] * 1.0 / sum_s
        total_scores.append(scores)

        validation_image.knn_class_name = class_cnt.most_common(1)[0][0]
        pr_or_classes.append((validation_image.knn_class_name, validation_image.original_class_name))
        if validation_image.knn_class_name == validation_image.original_class_name:
            score += 1
    score /= len(IMAGES)
    return score, pr_or_classes, np.array(total_scores)


def vector_KNN():
    scores = Counter()
    for i in range(0, 6):
        k = 2**i
        score, pr_or_classes, total_scores = _vector_knn(VALIDATION_IMAGES, k)
        scores[k] = score
    return scores.most_common()[0][0]


def calculate_conf_matrix(pr_or_classes):
    conf_matrix_counter = Counter()
    for predict, original in pr_or_classes:
        conf_matrix_counter[(predict, original)] += 1
    for item in conf_matrix_counter:
        value = conf_matrix_counter[item]
        conf_matrix_counter[item] = value * 1.0 * len(CLASSES) / len(pr_or_classes)
    return conf_matrix_counter


def draw_conf_matrix(conf_matrix_counter, matrix_name):
    conf_arr = np.zeros((len(CLASSES), len(CLASSES)))
    for i, class_name in enumerate(CLASSES):
        for j, class_name1 in enumerate(CLASSES):
            conf_arr[i, j] = conf_matrix_counter[(class_name, class_name1)]
    plt.imshow(conf_arr)
    plt.colorbar()
    plt.savefig('confusion_matrix_%s.png' % matrix_name)


def draw_p_r_curve(y_test, y_score, name):
    classes = list(CLASSES)
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_score[:, i])

        plt.figure(clear=True)
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig('precision_recall_curve_%s_%s' % (name, class_name))


def draw_roc_curve(y_test, y_score, name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(clear=True)
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_%s' % name)


def calculate_scores(pr_or_classes):
    accuracy_score = 0.0
    for predict, original in pr_or_classes:
        if predict == original:
            accuracy_score += 1
    accuracy_score /= len(pr_or_classes)

    TP = Counter()
    TN = Counter()
    FP = Counter()
    FN = Counter()
    precision = {}
    recall = {}
    fall_out = {}
    for class_name in CLASSES:
        for predict, original in pr_or_classes:
            if predict == original:
                if original == class_name:
                    TP[class_name] += 1
                else:
                    TN[class_name] += 1
            else:
                if original == class_name:
                    FN[class_name] += 1
                else:
                    FP[class_name] += 1
        recall[class_name] = TP[class_name]*1.0 / (TP[class_name] + FN[class_name])
        precision[class_name] = TP[class_name]*1.0 / (TP[class_name] + FP[class_name])
        fall_out[class_name] = FP[class_name]*1.0 / (FP[class_name] + TN[class_name])

    return accuracy_score, TP.most_common(), TN.most_common(), FP.most_common(), FN.most_common(), precision, recall, fall_out


TRAIN_IMAGES, VALIDATION_IMAGES = load_imgs(TRAIN_DIR, TRAIN_IMAGES, VALIDATION_IMAGES)
TEST_IMAGES, _ = load_imgs(TEST_DIR, TEST_IMAGES)
y_test = np.array(get_y_test())
name = sys.argv[1]
if name == 'knn_vector':
    k = vector_KNN()
    score, pr_or_classes, y_score = _vector_knn(TEST_IMAGES, k)
elif name == 'bow_knn':
    k, clusters_mean = BOW_KNN()
    calculate_histograms(TEST_IMAGES, clusters_mean)
    score, pr_or_classes, y_score = _bow_knn(TEST_IMAGES, k)
elif name == 'bow_svm':
    gamma, clusters_mean, class_num, X, y = BOW_SVM()
    calculate_histograms(TEST_IMAGES, clusters_mean)
    score, pr_or_classes, y_score = _bow_svm(TEST_IMAGES, gamma, class_num, X, y)
else:
    name = 'random'
    pr_or_classes, y_score = calculate_random_class()
accuracy_score, TP, TN, FP, FN, precision, recall, fall_out = calculate_scores(pr_or_classes)
print 'accuracy score is', accuracy_score, '\n\n'
print 'true positive scores are', TP, '\n\n'
print 'true negative scores are', TN, '\n\n'
print 'false positive scores are', FP, '\n\n'
print 'false negative scores are', FN, '\n\n'
print 'precision scores are', precision, '\n\n'
print 'recall scores are', recall, '\n\n'
print 'fall_out scores are', fall_out, '\n\n'
print '\n\n\n\n'
conf_counter = calculate_conf_matrix(pr_or_classes)
draw_conf_matrix(conf_counter, name)
draw_roc_curve(y_test, y_score, name)
draw_p_r_curve(y_test, y_score, name)
