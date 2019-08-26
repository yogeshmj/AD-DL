from __future__ import print_function

diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1, 'unlabeled': -1}


class SVMTester:
    def __init__(self, fold_dir):
        """
        :param fold_dir: Path to one fold of the trained SVM model.
        """
        import numpy as np
        from os import path
        from clinica.pipelines.machine_learning.voxel_based_io import revert_mask

        mask = np.loadtxt(path.join(fold_dir, 'data', 'train', 'mask.txt')).astype(bool)
        shape = np.loadtxt(path.join(fold_dir, 'data', 'train', 'shape.txt')).astype(int)

        weights = np.loadtxt(path.join(fold_dir, 'classifier', 'weights.txt'))
        self.weights = revert_mask(weights, mask, shape).flatten()
        self.intersect = np.loadtxt(path.join(fold_dir, 'classifier', 'intersect.txt'))

    def test(self, dataset):
        """
        :param dataset: (CAPSVoxelBasedInput) specific dataset of clinica initialized with test data.
        :return:
            (dict) metrics of evaluation
            (DataFrame) individual results of all sesssions
        """
        import numpy as np
        import pandas as pd

        images = dataset.get_x()
        labels = dataset.get_y()

        soft_prediction = np.dot(self.weights, images.transpose()) + self.intersect
        hard_prediction = (soft_prediction > 0).astype(int)
        subjects = dataset._subjects
        sessions = dataset._sessions
        data = np.array([subjects, sessions, labels, hard_prediction]).transpose()
        results_df = pd.DataFrame(data, columns=['participant_id', 'session_id', 'true_diagnosis', 'predicted_diagnosis'])

        return evaluate_prediction(labels, hard_prediction), results_df

    def test_and_save(self, dataset, evaluation_path):
        """
        :param dataset: (CAPSVoxelBasedInput) specific dataset of clinica initialized with test data.
        :param evaluation_path: (str) path to save the outputs
        :return: None
        """
        import pandas as pd
        import os

        metrics, results_df = self.test(dataset)
        metrics_df = pd.DataFrame(metrics, index=[0])

        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        metrics_df.to_csv(os.path.join(evaluation_path, 'metrics.tsv'), sep='\t', index=False)
        results_df.to_csv(os.path.join(evaluation_path, 'results.tsv'), sep='\t', index=False)


def evaluate_prediction(concat_true, concat_prediction, horizon=None):
    """
    This is a function to calculate the different metrics based on the list of true label and predicted label.

    :param concat_true: list of concatenated last labels
    :param concat_prediction: list of concatenated last prediction
    :param horizon: (int) number of batches to consider to evaluate performance
    :return: (dict) metrics
    """
    import numpy as np

    if horizon is not None:
        y = list(concat_true)[-horizon:]
        y_hat = list(concat_prediction)[-horizon:]
    else:
        y = list(concat_true)
        y_hat = list(concat_prediction)

    true_positive = np.sum((y_hat == 1) & (y == 1))
    true_negative = np.sum((y_hat == 0) & (y == 0))
    false_positive = np.sum((y_hat == 1) & (y == 0))
    false_negative = np.sum((y_hat == 0) & (y == 1))

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    if (true_positive + false_negative) != 0:
        sensitivity = true_positive / (true_positive + false_negative)
    else:
        sensitivity = 0.0

    if (false_positive + true_negative) != 0:
        specificity = true_negative / (false_positive + true_negative)
    else:
        specificity = 0.0

    if (true_positive + false_positive) != 0:
        ppv = true_positive / (true_positive + false_positive)
    else:
        ppv = 0.0

    if (true_negative + false_negative) != 0:
        npv = true_negative / (true_negative + false_negative)
    else:
        npv = 0.0

    balanced_accuracy = (sensitivity + specificity) / 2

    results = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy,
               'sensitivity': sensitivity,
               'specificity': specificity,
               'ppv': ppv,
               'npv': npv
               }

    return results


def check_and_clean(d):
    import os
    import shutil

    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def commandline_to_json(commandline, model_type="CNN", log_dir=None):
    """
    This is a function to write the python argparse object into a jason file. This helps for DL when searching for hyperparameters
    :param commandline: a tuple contain the output of `parser.parse_known_args()`
    :return:
    """
    import json
    import os

    commandline_arg_dic = vars(commandline[0])
    commandline_arg_dic['unknown_arg'] = commandline[1]

    # if train_from_stop_point, do not delete the folders
    output_dir = commandline_arg_dic['output_dir']
    if log_dir is None:
        log_dir = os.path.join(output_dir, 'log_dir', model_type, 'fold_' + str(commandline_arg_dic['split']))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save to json file
    json = json.dumps(commandline_arg_dic)
    print("Path of json file:", os.path.join(log_dir, "commandline.json"))
    f = open(os.path.join(log_dir, "commandline.json"), "w")
    f.write(json)
    f.close()
