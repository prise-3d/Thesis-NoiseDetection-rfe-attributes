#! bin/bash

# result file name
result_file_path="results/models_comparisons.csv"

# selection of four scenes (only maxwell)
scenes="A,D,G,H"

# only one model for the moment
model="rfe_svm_model"

# check feature param
if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need of feature information"
    exit 1
fi

# accept feature param
feature=$1

if [ -z "$2" ]
  then
    echo "No argument supplied"
    echo "Need of kind of data to use"
    exit 1
fi

feature=$1
data=$2

# get size depends on feature
declare -A featuresSize
featuresSize=( ["filters_statistics"]="26" ["svd"]="200" ["filters_statistics_sobel"]="27" ["svd_sobel"]="201")
size=${featuresSize[${feature}]}

# interval of data
start=0
end=$size

for nb_zones in {10,11,12}; do

    for mode in {"svd","svdn","svdne"}; do

        FILENAME="data/${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}_${data}"
        MODEL_NAME="${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}_${data}"
        CUSTOM_MIN_MAX_FILENAME="N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}_${data}_min_max"

        echo $FILENAME

        # only compute if necessary (perhaps server will fall.. Just in case)
        if grep -q "${MODEL_NAME}" "${result_file_path}"; then

            echo "${MODEL_NAME} results already generated..."
        else
            python generate/generate_data_model_random_${data}.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 10 --random 1 --custom ${CUSTOM_MIN_MAX_FILENAME}
            python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

            python others/save_model_result.py --data ${FILENAME} --model "saved_models/${MODEL_NAME}.joblib" --choice ${model} --feature ${feature} --mode ${mode} --zones ${nb_zones}
        fi
    done
done
