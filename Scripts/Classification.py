import collections
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.tree import DecisionTreeClassifier


def apply_classifiers(dataset_path):
    folders = list(filter(lambda x: os.path.isdir(os.path.join(dataset_path, x)), os.listdir(dataset_path)))

    for folder in folders:

        results = dict()
        p = os.path.join(dataset_path, folder)

        files = glob.glob('{0}/*.csv'.format(p))
        # print(files)

        if not files:
            continue

        for file in files:

            file_name = os.path.basename(os.path.splitext(file)[0])

            win_size = float(file_name[7:])
            print(win_size)

            dataset = pd.read_csv(file, sep='\t')
            groups = dataset['group']
            X = dataset.iloc[:, 1:-1].values

            Y = dataset.iloc[:, dataset.shape[1] - 1].values

            logo = LeaveOneGroupOut()
            for model_name, model in models.items():
                f1 = []

                for train_index, test_index in logo.split(X, Y, groups=groups):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = Y[train_index], Y[test_index]

                    classifier = model
                    classifier.fit(X_train, y_train)

                    y_pred = classifier.predict(X_test)
                    f1.append(f1_score(y_true=y_test, y_pred=y_pred, average='micro'))

                if win_size in results:
                    results[win_size].append(np.mean(f1))
                else:
                    results[win_size] = [np.mean(f1)]

        # export as csv file

        results = collections.OrderedDict(sorted(results.items()))

        final = []
        col = list(models.keys())
        col.insert(0, "window-size")
        final.append(col)
        for k, v in results.items():
            tmp = []
            tmp.append([k])
            tmp.append(v)
            flattened = [val for sublist in tmp for val in sublist]
            final.append(flattened)

        np.savetxt('./Results/{}-subjectCV.csv'.format(folder), final, delimiter=',', fmt='%s')


models = {'DT': DecisionTreeClassifier(criterion='entropy'), 'NB': GaussianNB(),
          'NCC': NearestCentroid(), "KNN": KNeighborsClassifier(n_neighbors=3)}

if (len(sys.argv) < 2):
    print('Please enter the path of datasets')
    exit()
elif (not os.path.exists(sys.argv[1])):
    print("This path does not exist!!")
    exit()

else:
    dataset_path = sys.argv[1]

    apply_classifiers(dataset_path=dataset_path)
