import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

project_root = os.path.dirname(os.path.dirname(__file__))
output_folder = os.path.join(project_root, 'Figures')
input_path = os.path.join(project_root, 'Results')


def plot_csv(csv_file, ax):
    results = pd.read_csv(csv_file)

    windows = results.pop('window-size')

    for col in results:
        max = results[col].idxmax()
        ax.plot(windows, results[col], label=col)
        ax.plot(windows[max], results[col][max], 'r*', label='peak')

    return ax


def plot_results(path):
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))

    for folder in folders:
        p = os.path.join(path, folder)

        files = glob.glob('{0}/*.csv'.format(p))

        if not files:
            continue

        for file in files:
            files_name = os.path.splitext(os.path.basename(file))[0]

            if ('non' in folder):

                figure_title = '{}-non-overlapping'.format(files_name)
            else:

                figure_title = '{}-overlapping'.format(files_name)

            fig, ax = plt.subplots()

            plot_csv(file, ax)

            fig.subplots_adjust(right=.84)

            # remove duplicate labels from legend

            handles, labels = ax.get_legend_handles_labels()
            handle_list, label_list = [], []
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)

            max_label_index = label_list.index('peak')
            max_label = label_list.pop(max_label_index)
            handle = handle_list.pop(max_label_index)

            handle_list.append(handle)
            label_list.append(max_label)

            ax.legend(handle_list, label_list, loc=(1.03, .72))

            plt.xlabel('Windows Size')
            plt.ylabel('f1_score')
            plt.ylim([0, 1])
            plt.title(figure_title)

            plt.savefig('{}/{}.png'.format(output_folder, figure_title))

    return True


plot_results(path=input_path)


def test_plot_results():
    assert plot_results(input_path)
