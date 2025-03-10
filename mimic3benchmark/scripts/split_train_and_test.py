from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse
import csv
import random
import math

def move_to_partition(args, patients, partition):
    if not os.path.exists(os.path.join(args.subjects_root_path, partition)):
        os.mkdir(os.path.join(args.subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.subjects_root_path, partition, patient)
        shutil.move(src, dest)

def split_training(patients, train_percent, validation_percent):
    random.shuffle(patients)
    n = len(patients)
    n_train = math.floor(n*train_percent)
    n_validation = math.floor(n*validation_percent)
    n_test = n - n_train - n_validation
    train_patients = patients[0:n_train]
    validation_patients = patients[n_train:n_train+n_validation]
    test_patients = patients[n_train+n_validation:]
    
    return train_patients, validation_patients, test_patients

def write_data_set(patients, split_name):
    fn = os.path.join(os.path.dirname(__file__), '../resources/{}set.csv'.format(split_name))
    with open(fn, "w") as dataset_file:
        wr = csv.writer(dataset_file)
        wr.writerow(patients)


def main():
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()

    test_set = set()
    with open(os.path.join(os.path.dirname(__file__), '../resources/testset.csv'), "r") as test_set_file:
        for line in test_set_file:
            x, y = line.split(',')
            if int(y) == 1:
                test_set.add(x)

    folders = os.listdir(args.subjects_root_path)
    folders = list((filter(str.isdigit, folders)))
    train_patients = [x for x in folders if x not in test_set]
    test_patients = [x for x in folders if x in test_set]

    assert len(set(train_patients) & set(test_patients)) == 0
    
    train_patients, validation_patients, val_test_patients = split_training(train_patients,
                                                                            train_percent = .44, 
                                                                            validation_percent = .44)
    
    write_data_set(train_patients, "train")
    write_data_set(validation_patients, "val")
    write_data_set(val_test_patients, "val_test")

    move_to_partition(args, train_patients, "train")
    move_to_partition(args, validation_patients, "val")
    move_to_partition(args, val_test_patients, "val_test")
    move_to_partition(args, test_patients, "test")


if __name__ == '__main__':
    main()
