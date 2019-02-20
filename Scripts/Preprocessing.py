import glob

import pandas as pd
import numpy as np
import os, sys

sample_rate = 50  # number of observation per second based on dataset documentation

sliding_size = int(.2 * sample_rate)  # number of skipped datapoints to start next window


############################################################################################ Feature sets


def mean_crossing_rate(col):
    # col = np.array(values)
    normalized = col - col.mean()  # to make elements of array possitive or negetive
    return ((normalized[:-1] * col[1:]) < 0).sum()  # Zero-Crossing_rate


def FS1(window):  # only mean

    avgs = list(window.mean()[:-1])

    label = window.iloc[:, -1].mode()[0]  ## select the most frequent label as the label of the window
    avgs.append(label)
    return avgs


def FS2(window):  # Mean and std

    features = []
    features.append(np.array(window.mean()[:-1]))
    features.append(np.array(window.std()[:-1]))
    features = np.hstack(features).tolist()

    label = window.iloc[:, -1].mode()  ## select the most frequent label as the label of the window

    features.append(label)

    return features


def FS3(window):  # mean, std,max,min and zero-crossing-rate

    features = []
    features.append(np.array(window.mean()[:-1]))
    features.append(np.array(window.std()[:-1]))
    features.append(np.array(window.min()[:-1]))
    features.append(np.array(window.max()[:-1]))
    mean_crossing = [mean_crossing_rate(window.iloc[:, i].values) for i in range(window.shape[1] - 1)]
    features.append(np.array(mean_crossing))

    features = np.hstack(features).tolist()

    label = window.iloc[:, -1].mode()  ## select the most frequent label as the label of the window
    features.append(label)
    return features


####################################################################################################################

def windowing_dataset(dataset, win_size, feature_extraction_function, subject_id, overlap=False):
    windowed_dataset = []
    win_count = 0
    if overlap:
        step_size = sliding_size  # for Overlapping technique
    else:
        step_size = win_size  # for Non-overlapping technique

    for index in range(0, dataset.shape[0], step_size):

        start = index
        end = start + win_size

        if (end <= dataset.shape[0]):  # to assure all of windows are equal in size
            window = dataset.iloc[start:end, :].reset_index(drop=True)
            win_count = win_count + 1
            features = feature_extraction_function(window)

            windowed_dataset.append(features)

    final = pd.DataFrame(windowed_dataset)
    final.insert(0, 'group', subject_id)  # to use in Subject CV
    return final


def Preprocessing(dataset_path, output_path, overlapping):
    # if (not os.path.exists(dataset_path)):
    #     print("Dataset path does not exist!!")
    #     return
    #
    # if (not os.path.exists(output_path)):
    #     print("Output path does not exist!!")
    #     return

    features_functions = [FS1, FS2, FS3]
    win_sizes = np.linspace(.25, 7, 28, endpoint=True)
    for win_size in win_sizes:

        print("Start for win size {}".format(win_size))
        datapoints_per_window = int(win_size * sample_rate)

        for feature_function in features_functions:

            print(feature_function.__name__)

            windowed_dataset = []

            for subject in range(1, 18):
                file_path = dataset_path + '/subject{0}_ideal.log'.format(subject)
                acc_cols = []
                for i in range(2, 117, 13):  # indices of accelarations
                    indices = list(range(i, i + 3))
                    acc_cols.extend(indices)

                acc_cols.append(119)  # label index

                tmp_db = pd.read_csv(file_path, header=None, usecols=acc_cols, sep='\t')
                tmp_db.columns = list(range(tmp_db.shape[1]))  # re-index the columns

                transformed_db = windowing_dataset(tmp_db, datapoints_per_window, feature_function, subject,
                                                   overlap=overlapping)

                windowed_dataset.append(transformed_db)

            final_dataset = pd.DataFrame()
            print("Merging!")
            final_dataset = final_dataset.append(windowed_dataset, ignore_index=True)

            if overlapping:
                out_folder_name = 'Overlapping_windowed'
            else:
                out_folder_name = 'Non-overlapping_windowed'

            os.makedirs('{}/{}'.format(output_path, out_folder_name), exist_ok=True)

            os.makedirs('{}/{}/FS1'.format(output_path, out_folder_name), exist_ok=True)
            os.makedirs('{}/{}/FS2'.format(output_path, out_folder_name), exist_ok=True)
            os.makedirs('{}/{}/FS3'.format(output_path, out_folder_name), exist_ok=True)

            if (feature_function == FS1):
                final_dataset.to_csv('{}/{}/FS1/dataset{}.csv'.format(output_path, out_folder_name, win_size), sep='\t',
                                     index=False)
            elif (feature_function == FS2):
                final_dataset.to_csv('{}/{}/FS2/dataset{}.csv'.format(output_path, out_folder_name, win_size), sep='\t',
                                     index=False)
            else:
                final_dataset.to_csv('{}/{}/FS3/dataset{}.csv'.format(output_path, out_folder_name, win_size), sep='\t',
                                     index=False)


############################################################################################################################

'''

 - Reads the raw data from input_path
 - Segments the raw datasets into windowed ones by different window sizes  
 - From each window it extracts FS1,FS2 and FS3.
 - Saves results in output_path. 
 
  Parameters:
    -----------
    dataset_path : Path of raw dataset
    
    output_path : Path to save the processed dataset
    
    overlapping : Controls the sliding windows technique;
    1: Overlapping sliding windows
    0: Non-overlapping sliding windows

    

'''

input_path = input('Please enter raw dataset path:')
output_path = input('Please enter the output path:')
overlapping = input(
    'please select sliding windows techniques (1 : Overlapping sliding windows / 0 : Non-overlapping sliding windows )')

if not os.path.exists(input_path):
    raise ValueError("Dataset path does not exist!!")

if not os.path.exists(output_path):
    raise ValueError("Output path does not exist!!")

if not overlapping.isnumeric():
    raise ValueError('Please select 0 or 1 for sliding windows technique')

elif int(overlapping) not in [0, 1]:
    raise ValueError('Please select correct digit for sliding windows technique')

Preprocessing(dataset_path=input_path, output_path=output_path, overlapping=bool(int(overlapping)))
