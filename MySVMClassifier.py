from __future__ import division
import random
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt


class MySVMClassifier:
    def __init__(self, dataSetFilePathName):
        self.dataSetFilePathName = dataSetFilePathName

    # Helper Functions
    def get_domain_from_url(self, url):
        return url[:url.find('/',8)]

    def get_path_from_url(self, url):
        return url[url.find('/',8):]

    def get_length(self, str):
        return len(str)

    def get_forward_slash_count(self, str):
        return str.count('/')

    def get_dot_count(selfm, str):
        return str.count('.')

    # LEXICAL FEATURES
    # URL Related
    def get_forward_slash_count_in_url(self, url):
        return self.get_forward_slash_count(url)

    def get_dot_count_in_url(self, url):
        return self.get_dot_count(url)

    def get_url_length(self, url):
        return self.get_length(url)

    # Domain Related
    def get_forward_slash_count_in_domain(self, url):
        return self.get_forward_slash_count(self.get_domain_from_url(url))

    def get_dot_count_in_domain(self, url):
        return self.get_dot_count(self.get_domain_from_url(url))

    def get_domain_length(self, url):
        return self.get_length(self.get_domain_from_url(url))

    # Path Related
    def get_forward_slash_count_in_path(self, url):
        return self.get_forward_slash_count(self.get_path_from_url(url))

    def get_dot_count_in_path(self, url):
        return self.get_dot_count(self.get_path_from_url(url))

    def get_path_length(self, url):
        return self.get_length(self.get_path_from_url(url))

    def make_feature_vector(self, url, label):
        feature_vector = []
        for i in range(len(url)):
            s = url[self.from_index+i]
            if len(s) > 0:
                feature = []
                feature.append(self.get_forward_slash_count_in_url(s))
                feature.append(self.get_dot_count_in_url(s))
                feature.append(self.get_url_length(s))

                feature.append(self.get_forward_slash_count_in_domain(s))
                feature.append(self.get_dot_count_in_domain(s))
                feature.append(self.get_domain_length(s))

                feature.append(self.get_forward_slash_count_in_path(s))
                feature.append(self.get_dot_count_in_path(s))
                feature.append(self.get_path_length(s))

                feature.append(label[self.from_index+i])
            feature_vector.append(feature)

        return feature_vector

    def make_np_array_XY(self, xy):
        a = np.array(xy)
        x = a[:, 0:-1]
        y = a[:, -1]
        return x, y

    def get_f1_score(self, Y_test, Y_predict, title):
        test_size = len(Y_test)
        score = 0
        for i in range(test_size):
            if Y_predict[i] == Y_test[i]:
                score += 1
        print('Got %s out of %s' % (score, test_size))
        print('f1 =%.2f' % (f1_score(Y_test, Y_predict, average='binary')))
        print('precision =%.2f' % (precision_score(Y_test, Y_predict, average='binary')))
        print('recall =%.2f' % (recall_score(Y_test, Y_predict, average='binary')))
        print('accuracy =%.2f' % (accuracy_score(Y_test, Y_predict)))

        bar_width = 0.20
        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        index = np.arange(1)

        fig, ax = plt.subplots()

        ax.bar(index, f1_score(Y_test, Y_predict, average='binary'), bar_width,
               alpha=opacity, color='b',
               error_kw=error_config,
               label='F Score')
        ax.bar(index + bar_width * 1, precision_score(Y_test, Y_predict, average='binary'), bar_width,
               alpha=opacity, color='r',
               error_kw=error_config,
               label='Precision')
        ax.bar(index + bar_width * 2, recall_score(Y_test, Y_predict, average='binary'), bar_width,
               alpha=opacity, color='g',
               error_kw=error_config,
               label='Recall')
        ax.bar(index + bar_width * 3, accuracy_score(Y_test, Y_predict), bar_width,
               alpha=opacity, color='y',
               error_kw=error_config,
               label='Accuracy')
        ax.set_xlabel('Measures')
        ax.set_ylabel('Scores')
        ax.set_title(title)
        ax.tick_params(
            axis='x',           # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom=False,       # ticks along the bottom edge are off
            top=False,          # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax.legend()
        fig.tight_layout()
        plt.show()

    def read_data_from_dataset_file(self, from_index, to_index):
        self.from_index = from_index
        self.to_index = to_index
        data = pd.read_csv(self.dataSetFilePathName)
        url       = data.iloc[self.from_index:self.to_index, 0]
        label     = data.iloc[self.from_index:self.to_index, 13]
        return url, label


if __name__ == '__main__':
    svm1 = MySVMClassifier('DataSet/PhishLegitimateDataSet.csv')
    url, label = svm1.read_data_from_dataset_file(30000,65000)

    # Features and Labels
    features_and_labels = svm1.make_feature_vector(url, label)
    random.shuffle(features_and_labels)

    # Train and Test Set
    cut = int(len(features_and_labels) * 0.9)
    XY_train = features_and_labels[:cut]
    XY_test = features_and_labels[cut:]

    print("SVM SVC Classifier")
    X_train, Y_train = svm1.make_np_array_XY(XY_train)
    X_test, Y_test = svm1.make_np_array_XY(XY_test)

    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
    Y_predict = svc.predict(X_test)
    print('Y_predict:\n', Y_predict)
    print('Y_test:   \n', Y_test)
    svm1.get_f1_score(Y_test, Y_predict, "Various Performance Measures for SVM SVC Classifier with Linear Kernel")

    print("SGD Classifier")
    from sklearn import linear_model
    svc = linear_model.SGDClassifier().fit(X_train, Y_train)

    print("svc.predict()...")
    Y_predict = svc.predict(X_test)

    print('Y_predict:\n', Y_predict)
    print('Y_test:   \n', Y_test)
    svm1.get_f1_score(Y_test, Y_predict, "Various Performance Measures for SGD Classifier")