import pandas as pd
import numpy as np
import scipy.io as spio
import sys, os

'''
-This code export the singleonly.mat to csv 
- The time stamp of some sensors are not the same, so I use the time time stamp of the accelerator sensor.
-The order of time stamp is the same as mat file
- In several cases, the Saccelerator and Sgyrescope do not have any record, so here I fill with nan
- The final csv file has 15 columns (time stamp+subject id+ 4*3 sensors+activity id) and needs about 3.6 g disk
- During the writing code, the priority was mostly being readible than efficency, so feel free to improve it  
'''

activities = None


def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def _getMinRowLenght(struct):
    values = []

    for name in struct.data._fieldnames:

        if (getattr(struct.data, name).shape[0] > 0):
            values.append(getattr(struct.data, name).shape[0])
    return min(values)


def _extract_info(struct):
    rows_lenght = _getMinRowLenght(struct)  # some sensors produced more observations, so we take the min
    records = []
    for sensor_name in struct.data._fieldnames:

        if (
                sensor_name == 'accelDataMatrix'):  # Since sometimes time stams are not equal, we get time stamp from accelerator sensor only
            records.append(getattr(struct.data, sensor_name)[:rows_lenght, :])
        else:
            attr = getattr(struct.data, sensor_name)
            if (len(attr) == 0):
                attr = np.full([rows_lenght, 3],
                               np.nan)  # the data of some of S sensors are not available, so we fill with nan
            else:
                attr = attr[:rows_lenght, 1:]
            records.append(attr)
    records = np.hstack(records)

    records = np.insert(records, 1, struct.subjectID, axis=1)
    activity_id = np.where(activities == struct.activityName)[0][0]
    records = np.insert(records, records.shape[1], activity_id, axis=1)
    return records


#####################################################################################


def export_to_csv(matfile_path):
    if (not (os.path.exists(matfile_path))):
        print("inputed path is not correct")
        return

    print("Reading the file...")
    mat = loadmat(matfile_path)  # path of mat file

    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    global activities
    activities = mat['exerciseConstants']['activities']
    pd.DataFrame(activities).to_csv('activities.csv', index=False)  # export avtivity names

    data = mat['subject_data']
    final_data = []
    print("Exporting...")
    for sbj in range(data.shape[0]):

        for exercise in range(data.shape[1]):

            this_sbj = data[sbj][exercise]
            if isinstance(this_sbj, spio.matlab.mio5_params.mat_struct):
                # print(" {} , {} ".format(sbj, exercise))
                tmp = _extract_info(this_sbj)
                final_data.append(tmp)

            elif (isinstance(this_sbj, np.ndarray) and len(this_sbj) > 0):

                # print(" {} , {} ".format(sbj, exercise))

                for stc in this_sbj:
                    tmp = _extract_info(stc)
                    final_data.append(tmp)

    final_data = np.vstack(final_data)

    cols_name = ['Time Stamp', 'subject_ID', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'sacc_x',
                 'sacc_y',
                 'sacc_z', 'sgyro_x', 'sgyro_y', 'sgyro_z', 'activity_id']
    df = pd.DataFrame(final_data, columns=cols_name)

    

    print("Writing on the disk...")

    df.to_csv('exercise_data.50.0000_singleonly.csv', index=False)
    print("done")


##########################################################################

# start here

export_to_csv(matfile_path='exercise_data.50.0000_singleonly.mat')
