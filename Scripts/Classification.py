import collections
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.tree import DecisionTreeClassifier


def subject_cross_validation(X, Y, groups, classifier):
    f1 = []
    logo = LeaveOneGroupOut()

    for train_index, test_index in logo.split(X, Y, groups=groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        f1.append(f1_score(y_true=y_test, y_pred=y_pred, average='micro'))
    return np.mean(f1)


# cv_strategy can be iid or sbj for k-fold cv and subject cv respectively

def apply_classifiers(dataset_path, models, cv_strategy):

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

            Y = dataset.iloc[:, -1].values


            for model_name, model in models.items():
                f1 = 0

                if cv_strategy == 'sbj':

                    f1 = subject_cross_validation(X, Y, groups, model)

                else:

                    f1 = cross_val_score(estimator=model, X=X, y=Y,
                                         cv=KFold(n_splits=10, shuffle=True, random_state=1), scoring='f1_micro',
                                         n_jobs=-1).mean()

                if win_size in results:
                    results[win_size].append(f1)
                else:
                    results[win_size] = [f1]

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

        output_name = ''
        if cv_strategy == 'sbj':

            output_name = folder + '-' + 'subjective'

        else:

            output_name = folder + '-' + 'iid'

        np.savetxt(os.path.join(os.path.dirname(os.getcwd()), 'Results', output_name + '.csv'), final, delimiter=',',
                   fmt='%s')



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

    apply_classifiers(dataset_path=dataset_path, models=models, cv_strategy='sbj')
