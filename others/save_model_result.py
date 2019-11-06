# main imports
import numpy as np
import pandas as pd

import sys, os, argparse
import subprocess
import time
import json

# models imports
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# image processing imports
from ipfml import processing
from PIL import Image

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

# variables and parameters
threshold_map_folder        = cfg.threshold_map_folder
threshold_map_file_prefix   = cfg.threshold_map_folder + "_"

markdowns_folder            = cfg.models_information_folder
final_csv_model_comparisons = cfg.csv_model_comparisons_filename
models_name                 = cfg.models_names_list

zones                       = cfg.zones_indices

current_dirpath = os.getcwd()


def main():

    
    parser = argparse.ArgumentParser(description="Save data results of learned model")

    parser.add_argument('--data', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--model', type=str, help='.joblib or .json file (sklearn or keras model)')
    parser.add_argument('--choice', type=str, help='Name of the model used', choices=cfg.models_names_list)
    parser.add_argument('--zones', type=int, help='Number of zones used when learning')
    parser.add_argument('--feature', type=str, help='feature data choice', choices=cfg.features_choices_labels)
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=cfg.normalization_choices)


    args = parser.parse_args()

    p_data_file = args.data
    p_model_file = args.model
    p_model_name = args.choice
    p_zones      = args.zones
    p_feature    = args.feature
    p_mode       = args.mode

    model_scores = []

    ########################
    # 1. Get and prepare data
    ########################
    dataset_train = pd.read_csv(p_data_file + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(p_data_file + '.test', header=None, sep=";")

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    # get dataset with equal number of classes occurences
    noisy_df_train = dataset_train[dataset_train.ix[:, 0] == 1]
    not_noisy_df_train = dataset_train[dataset_train.ix[:, 0] == 0]
    nb_noisy_train = len(noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.ix[:, 0] == 1]
    not_noisy_df_test = dataset_test[dataset_test.ix[:, 0] == 0]
    nb_noisy_test = len(noisy_df_test.index)

    final_df_train = pd.concat([not_noisy_df_train[0:nb_noisy_train], noisy_df_train])
    final_df_test = pd.concat([not_noisy_df_test[0:nb_noisy_test], noisy_df_test])

    # shuffle data another time
    final_df_train = shuffle(final_df_train)
    final_df_test = shuffle(final_df_test)

    final_df_train_size = len(final_df_train.index)
    final_df_test_size = len(final_df_test.index)

    # use of the whole data set for training
    x_dataset_train = final_df_train.ix[:,1:]
    x_dataset_test = final_df_test.ix[:,1:]

    y_dataset_train = final_df_train.ix[:,0]
    y_dataset_test = final_df_test.ix[:,0]

    #######################
    # 2. Getting model
    #######################

    model = joblib.load(p_model_file)
    selected_indices = [(i + 1) for i in np.arange(len(model.support_)) if model.support_[i] == True]
    selected_indices_displayed = [i for i in np.arange(len(model.support_)) if model.support_[i] == True]
    print(selected_indices)
    
    # update dataset values using specific indices
    x_dataset_train = x_dataset_train.loc[:, selected_indices]
    x_dataset_test = x_dataset_test.loc[:, selected_indices]

    #######################
    # 3. Fit model : use of cross validation to fit model
    #######################

    model.estimator_.fit(x_dataset_train, y_dataset_train)
    train_accuracy = cross_val_score(model.estimator_, x_dataset_train, y_dataset_train, cv=5)

    ######################
    # 4. Test : Validation and test dataset from .test dataset
    ######################

    # we need to specify validation size to 20% of whole dataset
    val_set_size = int(final_df_train_size/3)
    test_set_size = val_set_size

    total_validation_size = val_set_size + test_set_size

    if final_df_test_size > total_validation_size:
        x_dataset_test = x_dataset_test[0:total_validation_size]
        y_dataset_test = y_dataset_test[0:total_validation_size]

    X_test, X_val, y_test, y_val = train_test_split(x_dataset_test, y_dataset_test, test_size=0.5, random_state=1)

    # update dataset values using specific indices
    y_test_model = model.estimator_.predict(X_test)
    y_val_model = model.estimator_.predict(X_val)
    y_train_model = model.estimator_.predict(x_dataset_train)

    # getting all scores
    val_accuracy = accuracy_score(y_val, y_val_model)
    test_accuracy = accuracy_score(y_test, y_test_model)

    train_f1 = f1_score(y_dataset_train, y_train_model)
    train_recall = recall_score(y_dataset_train, y_train_model)
    train_roc_auc = roc_auc_score(y_dataset_train, y_train_model)

    val_f1 = f1_score(y_val, y_val_model)
    val_recall = recall_score(y_val, y_val_model)
    val_roc_auc = roc_auc_score(y_val, y_val_model)

    test_f1 = f1_score(y_test, y_test_model)
    test_recall = recall_score(y_test, y_test_model)
    test_roc_auc = roc_auc_score(y_test, y_test_model)

    all_x_data = pd.concat([x_dataset_train, X_test, X_val])
    all_y_data = pd.concat([y_dataset_train, y_test, y_val])
    all_y_model = model.estimator_.predict(all_x_data)

    all_accuracy = accuracy_score(all_y_data, all_y_model)
    all_f1_score = f1_score(all_y_data, all_y_model)
    all_recall_score = recall_score(all_y_data, all_y_model)
    all_roc_auc_score = roc_auc_score(all_y_data, all_y_model)

    # stats of dataset sizes
    total_samples = final_df_train_size + val_set_size + test_set_size

    model_scores.append(final_df_train_size)
    model_scores.append(val_set_size)
    model_scores.append(test_set_size)

    model_scores.append(final_df_train_size / total_samples)
    model_scores.append(val_set_size / total_samples)
    model_scores.append(test_set_size / total_samples)

    # add of scores
    model_scores.append(train_accuracy)
    model_scores.append(val_accuracy)
    model_scores.append(test_accuracy)
    model_scores.append(all_accuracy)

    model_scores.append(train_f1)
    model_scores.append(train_recall)
    model_scores.append(train_roc_auc)

    model_scores.append(val_f1)
    model_scores.append(val_recall)
    model_scores.append(val_roc_auc)

    model_scores.append(test_f1)
    model_scores.append(test_recall)
    model_scores.append(test_roc_auc)

    model_scores.append(all_f1_score)
    model_scores.append(all_recall_score)
    model_scores.append(all_roc_auc_score)

    # add final line into data
    final_file_line = p_model_name + ';' + str(selected_indices_displayed) + '; ' + str(p_zones) + '; ' + p_feature + '; ' + p_mode

    for s in model_scores:
        final_file_line += '; ' + str(s)
    
    # Prepare writing in .csv file into results folder
    output_final_file_path = os.path.join(cfg.results_information_folder, final_csv_model_comparisons)
    
    if not os.path.exists(cfg.results_information_folder):
        os.makedirs(cfg.results_information_folder)

    output_final_file = open(output_final_file_path, "a")

    output_final_file.write(final_file_line + '\n')


if __name__== "__main__":
    main()
