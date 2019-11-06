#! bin/bash

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need of vector size"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No argument supplied"
    echo "Need of feature information"
    exit 1
fi

if [ -z "$3" ]
  then
    echo "No argument supplied"
    echo "Need of kind of data to use"
    exit 1
fi

size=$1
feature=$2
data=$3

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

start=0
end=$size
model="rfe_svm_model"

for nb_zones in {10,11,12}; do

    for mode in {"svd","svdn","svdne"}; do

        FILENAME="data/${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}_${data}"
        MODEL_NAME="${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}_${data}"
        CUSTOM_MIN_MAX_FILENAME="N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}_${data}_min_max"

        echo $FILENAME

        # only compute if necessary (perhaps server will fall.. Just in case)
        if grep -q "${MODEL_NAME}" "${result_filename}"; then

            echo "${MODEL_NAME} results already generated..."
        else
            python generate/generate_data_model_random_${data}.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 10 --random 1 --custom ${CUSTOM_MIN_MAX_FILENAME}
            #python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

            #python prediction/predict_seuil_expe_maxwell.py --interval "${start},${end}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature} --limit_detection '2' --custom ${CUSTOM_MIN_MAX_FILENAME}
            #python others/save_model_result_in_md_maxwell.py --interval "${start},${end}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature}
        fi
    done
done
