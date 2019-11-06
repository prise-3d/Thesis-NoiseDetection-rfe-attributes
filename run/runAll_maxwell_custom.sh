#! bin/bash

# erase "results/models_comparisons.csv" file and write new header
result_file_path="results/models_comparisons.csv"

features_list="(filters_statistics|filters_statistics_sobel|svd|svd_sobel)"
if [[ "$1" =~ ^(filters_statistics|filters_statistics_sobel|svd|svd_sobel)$ ]]; then
    echo "$1 is in the list"
else
    echo "First arguement '$1' is not in the list, need argument from [${features_list}]"
    exit 1
fi

# accept feature param
feature=$1

# if [ -z "$2" ]
#   then
#     echo "No argument supplied"
#     echo "Need argument from [${list}]"
#     exit 1
# fi

data_list="(all|center|split)"
if [[ "$2" =~ ^(all|center|split)$ ]]; then
    echo "$2 is in the list"
else
    echo "Second argument '$2' is not in the list, need argument from [${data_list}]"
fi

# accept data param
data=$2

# check erased data param
erased=$3

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${result_file_path}
    # if necessary
    mkdir -p results 
    touch ${result_file_path}

    # add of header
    echo 'model_name; selected_indices; nb_zones; feature; mode; tran_size; val_size; test_size; train_pct_size; val_pct_size; test_pct_size; train_acc; val_acc; test_acc; all_acc; F1_train; recall_train; roc_auc_train; F1_val; recall_val; roc_auc_val; F1_test; recall_test; roc_auc_test; F1_all; recall_all; roc_auc_all;' >> ${file_path}
fi

bash data_processing/generateAndTrain_maxwell_custom.sh ${feature} ${data}
