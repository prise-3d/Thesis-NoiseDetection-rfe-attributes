#! bin/bash

# file which contains model names we want to use for simulation
simulate_models="simulate_models.csv"

# selection of four scenes (only maxwell)
scenes="A,D,G,H"

# model choice
model="rfe_svm_model"

# check feature param
if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need of feature information"
    exit 1
fi

if [[ "$1" =~ ^(filters_statistics|filters_statistics_sobel|svd|svd_sobel)$ ]]; then
    echo "$1 is in the list"
else
    echo "$1 is not in the list"
    exit 1
fi

# accept feature param
feature=$1

declare -A featuresSize
featuresSize=( ["filters_statistics"]="26" ["svd"]="200" ["filters_statistics_sobel"]="27" ["svd_sobel"]="201")

size=${featuresSize[${feature}]}

for nb_zones in {10,11,12}; do
    for mode in {"svd","svdn","svdne"}; do

        FILENAME="data/${model}_N${size}_B0_E${size}_nb_zones_${nb_zones}_${feature}_${mode}_all"
        MODEL_NAME="${model}_N${size}_B0_E${size}_nb_zones_${nb_zones}_${feature}_${mode}_all"
        CUSTOM_MIN_MAX_FILENAME="N${size}_B0_E${size}_nb_zones_${nb_zones}_${feature}_${mode}_all_min_max"

        # only compute if necessary (perhaps server will fall.. Just in case)
        if grep -xq "${MODEL_NAME}" "${simulate_models}"; then

            # Use of already generated model
            python generate/generate_data_model_random.py --output ${FILENAME} --interval "0,${size}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 40 --random 1 --custom ${CUSTOM_MIN_MAX_FILENAME}
            python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

            python prediction/predict_seuil_expe_maxwell_curve.py --interval "0,${size}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature} --custom ${CUSTOM_MIN_MAX_FILENAME}
        fi
    done
done
