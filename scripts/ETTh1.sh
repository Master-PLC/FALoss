#!/bin/bash
MAX_JOBS=32
GPUS=(0 1 2 3 4 5 6 7)
TOTAL_GPUS=${#GPUS[@]}

get_gpu_allocation(){
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=${GPUS[$((job_number % TOTAL_GPUS))]}
    echo $gpu_id
}

check_jobs(){
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0

DATA_ROOT=./dataset
OUT_ROOT=./
EXP_NAME=long_term
seed=2023
des='iTransformer'

model_name=iTransformer

auxi_mode=rfft
auxi_type=complex
auxi_loss=MAE
module_first=1
first_order=1
overlap_ratio=0.0
test_batch_size=1


datasets=(ETTh1)



# hyper-parameters
dst=ETTh1

pl_list=(96 192 336 720)
train_epochs=30
patience=10


rerun=0

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.002 lambda=0.0 batch_size=32 lr_inner=0.002 lr_meta=0.1 meta_inner_steps=5 num_tasks=3 max_norm=5.0 meta_steps=300 auxi_batch_size=64 lradj=type1;;
        192) lr=0.002 lambda=0.2 batch_size=32 lr_inner=0.002 lr_meta=0.05 meta_inner_steps=5 num_tasks=4 max_norm=5.0 meta_steps=300 auxi_batch_size=256 lradj=type1;;
        336) lr=0.002 lambda=0.0 batch_size=32 lr_inner=0.002 lr_meta=0.02 meta_inner_steps=5 num_tasks=5 max_norm=5.0 meta_steps=300 auxi_batch_size=64 lradj=type1;;
        720) lr=0.001 lambda=0.0 batch_size=128 lr_inner=0.001 lr_meta=0.01 meta_inner_steps=5 num_tasks=4 max_norm=5.0 meta_steps=300 auxi_batch_size=64 lradj=type3;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_meta_ml3 \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    } &

done





wait