import random
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt


class NBClassifier:
    def __init__(self, data_set_file_path_name):
        self.data_set_file_path_name = data_set_file_path_name

    # Helper Functions
    def get_domain_from_url(self, url):
        return url[:url.find('/', 8)]

    def get_path_from_url(self, url):
        return url[url.find('/', 8):]

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

    def get_f1_score(self, Y_test, Y_predict):
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
        ax.bar(index + bar_width*1, precision_score(Y_test, Y_predict, average='binary'), bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='Precision')
        ax.bar(index + bar_width*2, recall_score(Y_test, Y_predict, average='binary'), bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='Recall')
        ax.bar(index + bar_width*3, accuracy_score(Y_test, Y_predict), bar_width,
                alpha=opacity, color='y',
                error_kw=error_config,
                label='Accuracy')
        ax.set_xlabel('Measures')
        ax.set_ylabel('Scores')
        ax.set_title('Various Performanace Measures for Naive Bayes')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax.legend()
        fig.tight_layout()
        plt.show()

    def get_instances_stats_in_train_set(self, Y_train):
        test_size = len(Y_train)
        phish = 0
        legitimate = 0
        for i in range(test_size):
            if Y_train[i] == 1:
                phish += 1
            if Y_train[i] == 0:
                legitimate += 1
        print("Phish Count="+str(phish)+" and legitimate="+str(legitimate))
        return phish, legitimate

    def read_data_from_dataset_file(self, from_index, to_index):
        self.from_index = from_index
        self.to_index = to_index
        data = pd.read_csv(self.data_set_file_path_name)
        url       = data.iloc[self.from_index:self.to_index, 0]
        label     = data.iloc[self.from_index:self.to_index, 13]
        return url, label

    def make_feature_vector(self, url, label):
        feature_vector = []
        for i in range(len(url)):
            s = url[self.from_index + i]
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

                feature.append(label[self.from_index + i])
            feature_vector.append(feature)

        return feature_vector

    def ExecuteClassifier(self, train_ratio):
        url, label = self.read_data_from_dataset_file(30000, 65000)
        print("Total Number of Instances: " + str(len(url)))
        print("Running . . . ")

        # Features and Labels
        features_and_labels = self.make_feature_vector(url, label)
        random.shuffle(features_and_labels)
        features_and_labels = np.array(features_and_labels)

        # Train and Test Set
        cut = int(len(features_and_labels) * train_ratio)
        XY_train = features_and_labels[:cut]
        XY_test = features_and_labels[cut:]

        XY_train_features = XY_train[:, :-1]
        XY_train_labels = XY_train[:, -1]

        XY_test_features = XY_test[:, :-1]
        XY_test_labels = XY_test[:, -1]

        self.get_instances_stats_in_train_set(XY_train_labels)

        gnb = GaussianNB()
        gnb.fit(XY_train_features, XY_train_labels)
        predicted = gnb.predict(XY_test_features)
        self.get_f1_score(XY_test_labels, predicted)

nbRun2 = NBClassifier('DataSet/PhishLegitimateDataSet.csv')
nbRun2.ExecuteClassifier(0.9)