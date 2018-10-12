from sklearn.model_selection import (TimeSeriesSplit, KFold,
                                     LeaveOneGroupOut,ShuffleSplit)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
np.random.seed(1)

def plot_group_class(classes, groups):

    fig, ax = plt.subplots()
    ax.scatter(range(len(groups)),  [.5] * len(groups), c=groups, marker='_',
               lw=50,cmap=plt.cm.tab20)
    ax.scatter(range(len(groups)),  [3.5] * len(groups), c=classes, marker='_',
               lw=50,cmap=plt.cm.tab20b)
    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['Subject', 'Class'], xlabel="Sample index")

    ax.legend([Patch(color='navy')],
              ['Dominant class'], loc=(1.003,.94))
    plt.tight_layout()
    fig.subplots_adjust(right=.75)
    plt.savefig('Class_subject.png')


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):

    splits = cv.split(X=X, y=y, groups=group)

    for index, (training, test) in enumerate(splits):

        indices = np.array([np.nan] * len(X))
        indices[test] = 1
        indices[training] = 0



        ax.scatter(range(len(indices)), [index + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=plt.cm.coolwarm
                   )


    ax.scatter(range(len(X)), [index + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.tab20)

    ax.scatter(range(len(X)), [index + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=plt.cm.tab20)


    if(isinstance(cv,LeaveOneGroupOut)):
      n_splits=cv.get_n_splits(groups=group)

    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax



def plot_cv(dataset,CVs,n_splits):

    if(len(CVs)==0):
        raise  ValueError('There is any CV to plot.')


    dataset = pd.read_csv(dataset, sep='\t')

    groups = dataset['group'].values.ravel()

    X = dataset.iloc[:, 1:-1].values

    Y = dataset.iloc[:, dataset.shape[1] - 1].values.revel()

    plot_group_class(classes=Y,groups=groups)

    for cv in CVs:


        if (cv==LeaveOneGroupOut):
            cur_cv=cv()

        else:
            cur_cv=cv(n_splits=n_splits)

        fig_name = type(cur_cv).__name__
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_cv_indices(cv=cur_cv,X=X,y=Y,group=groups,ax=ax,n_splits=n_splits)

        ax.legend([Patch(color='r'), Patch(color='b')],
                  ['Testing set', 'Training set'], loc=(1.02, .8))

        plt.tight_layout()
        fig.subplots_adjust(right=.7)
        plt.savefig('{}.png'.format(fig_name))





cvs = [TimeSeriesSplit,KFold,LeaveOneGroupOut,ShuffleSplit]

plot_cv(dataset='dataset0.5.csv',CVs=cvs,n_splits=10)

