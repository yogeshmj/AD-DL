import pandas as pd
import os
from os import path


def save_data(df, output_dir, folder_name):
    """
    Save data so it can be used by the workflow

    :param df:
    :param output_dir:
    :param folder_name:
    :return: path to the tsv files
    """

    results_dir = path.join(output_dir, 'data', folder_name)
    if not path.exists(results_dir):
        os.makedirs(results_dir)

    df[['diagnosis']].to_csv(path.join(results_dir, 'diagnoses.tsv'), sep="\t", index=False)
    df[['participant_id', 'session_id']].to_csv(path.join(results_dir, 'sessions.tsv'), sep="\t", index=False)

    return results_dir


def save_additional_parameters(workflow, output_dir):
    """
    Saves additional parameters necessary for the testing phase (mask and original shape of the images).

    :param workflow: (MLWorkflow) workflow from which mask and original shape must be saved
    :return: None
    """
    import numpy as np

    mask = workflow._input._data_mask
    orig_shape = workflow._input._orig_shape
    np.savetxt(path.join(output_dir, 'mask.txt'), mask)
    np.savetxt(path.join(output_dir, 'shape.txt'), orig_shape)


def load_data(train_val_path, diagnoses_list, split, n_splits=None, baseline=True):

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()

    if n_splits is None:
        train_path = path.join(train_val_path, 'train')
        valid_path = path.join(train_val_path, 'validation')

    else:
        train_path = path.join(train_val_path, 'train_splits-' + str(n_splits),
                               'split-' + str(split))
        valid_path = path.join(train_val_path, 'validation_splits-' + str(n_splits),
                               'split-' + str(split))

    print("Train", train_path)
    print("Valid", valid_path)

    for diagnosis in diagnoses_list:

        if baseline:
            train_diagnosis_path = path.join(train_path, diagnosis + '_baseline.tsv')
        else:
            train_diagnosis_path = path.join(train_path, diagnosis + '.tsv')

        valid_diagnosis_path = path.join(valid_path, diagnosis + '_baseline.tsv')

        train_diagnosis_df = pd.read_csv(train_diagnosis_path, sep='\t')
        valid_diagnosis_df = pd.read_csv(valid_diagnosis_path, sep='\t')

        train_df = pd.concat([train_df, train_diagnosis_df])
        valid_df = pd.concat([valid_df, valid_diagnosis_df])

    train_df.reset_index(inplace=True, drop=True)
    valid_df.reset_index(inplace=True, drop=True)

    return train_df, valid_df


def load_data_test(test_path, diagnoses_list):

    test_df = pd.DataFrame()

    for diagnosis in diagnoses_list:

        test_diagnosis_path = path.join(test_path, diagnosis + '_baseline.tsv')
        test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
        test_df = pd.concat([test_df, test_diagnosis_df])

    test_df.reset_index(inplace=True, drop=True)

    return test_df
