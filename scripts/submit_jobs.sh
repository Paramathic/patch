#!/bin/bash

# --- Configuration ---
# Define the ranges for your hyperparameters
LEARNING_RATES=("1e-3") # 0.1 0.5 0.25)
REG_FACTORS=("2") # 1 3 5)
TEMP_PAIRS=("4 0.05")
SCALER_PAIRS=("100 500")
TILE_TEMP_PAIRS=("4 0.05")
TILE_SCALER_PAIRS=("25 350")
LOCAL_BATCH_SIZES_ADAM=1 # 4)
OPTIMIZERS=("adamw_torch")
INITIAL_SPARSITY=(0.5)
TARGET_SPARSITY=(0.45) # 0.35)
COPY_DATA=false
MASK_TILE_SIZE="128,128"
SPARSITY_TYPE="2:4"
PRUNING_METHOD="maskllm"
MODEL_NAME=llama2
FINE_TUNING_SEQUENCE_LENGTH=2048
WEIGHT_REG=("0.05") # "0") # "4.25")
TILE_STRENGTH=("3")
WEIGHT_STRENGTH=("3") 
CLUSTER="trillium"
PARALLELISM="data_parallel" # Options: data_parallel, model_parallel
LEARNABLE_2_4_TILE=false
MASKLLM_CHECKPOINT="Vinnnf/LLaMA-2-7B-MaskLLM-C4"

NGPUS_PER_NODE=4
NTASKS_PER_NODE=$((12 * NGPUS_PER_NODE))
MEM=$((64 * NGPUS_PER_NODE))

if [ $MODEL_NAME == 'llama2' ]
then
    MODEL_PREFIX=meta-llama/Llama-2-
    MODEL_POSTFIX=-hf
    MODEL_SIZE_LIST="7b"
elif [ $MODEL_NAME == 'opt' ]
then   
    MODEL_PREFIX=facebook/opt-
    MODEL_POSTFIX=""
    MODEL_SIZE_LIST="30b"
elif [ $MODEL_NAME == 'llama3.2' ]
then
    MODEL_PREFIX=meta-llama/Llama-3.2-
    MODEL_SIZE_LIST="1B"
    MODEL_POSTFIX=""
elif [ $MODEL_NAME == 'llama3.1' ]
then
    MODEL_PREFIX=meta-llama/Llama-3.1-
    MODEL_SIZE_LIST="8B"
    MODEL_POSTFIX=""
elif [ $MODEL_NAME == 'gemma3' ]
then
    MODEL_PREFIX=google/gemma-3-
    MODEL_SIZE_LIST="1b"
    MODEL_POSTFIX="-pt"
elif [ $MODEL_NAME == 'qwen2.5' ]
then
    MODEL_PREFIX=Qwen/Qwen2.5-
    MODEL_SIZE_LIST="0.5B"
    MODEL_POSTFIX=""
fi

SLURM_SCRIPT="scripts/job_template.sh"

echo "Starting job submission loop..."

# Loop through all combinations
job_count=0
for lr in "${LEARNING_RATES[@]}"; do
    for reg in "${REG_FACTORS[@]}"; do
        for opt in "${OPTIMIZERS[@]}"; do
            for temp_pair in "${TEMP_PAIRS[@]}"; do
                read s_temp e_temp <<< "$temp_pair"
                for scaler_pair in "${SCALER_PAIRS[@]}"; do
                    read s_scaler e_scaler <<< "$scaler_pair"
                    for tile_temp_pair in "${TILE_TEMP_PAIRS[@]}"; do
                        read s_tile_temp e_tile_temp <<< "$tile_temp_pair"
                        for tile_scaler_pair in "${TILE_SCALER_PAIRS[@]}"; do
                            read s_tile_scaler e_tile_scaler <<< "$tile_scaler_pair"
                            for weight_reg in "${WEIGHT_REG[@]}"; do
                                for tile_str in "${TILE_STRENGTH[@]}"; do
                                    for weight_strength in "${WEIGHT_STRENGTH[@]}"; do
                                        for init_sparsity in "${INITIAL_SPARSITY[@]}"; do
                                            for target_sparsity in "${TARGET_SPARSITY[@]}"; do
                                                for MODEL_SIZE in $MODEL_SIZE_LIST; do

                                                    local_batch_size=""
                                                    if [ "$opt" == "adafactor" ]; then
                                                        local_batch_size=4
                                                    elif [ "$opt" == "adamw_torch" ]; then
                                                        local_batch_size=$LOCAL_BATCH_SIZES_ADAM
                                                    fi

                                                    JOB_NAME="mask_${lr}_${reg}_${opt}_${s_temp}_${e_temp}_${s_scaler}_${e_scaler}_${s_tile_temp}_${e_tile_temp}_${s_tile_scaler}_${e_tile_scaler}_${init_sparsity}_${target_sparsity}"
                                                    JOB_NAME=$(echo "$JOB_NAME" | sed 's/e-/em/' | sed 's/[^A-Za-z0-9._-]/_/g')

                                                    # echo "--------------------------------------------------"
                                                    # echo "Submitting job #$((job_count + 1)): ${JOB_NAME}"
                                                    # echo "  LR               : ${lr}"
                                                    # echo "  Reg Factor       : ${reg}"
                                                    # echo "  Batch Size       : ${local_batch_size}"
                                                    # echo "  Optimizer        : ${opt}"
                                                    # echo "  Temp Scheduler   : ${s_temp} - ${e_temp}"
                                                    # echo "  Scaler Scheduler : ${s_scaler} - ${e_scaler}"
                                                    # echo "  Initial Sparsity : ${init_sparsity}"
                                                    # echo "  Target Sparsity  : ${target_sparsity}"

                                                    sbatch --account=def-mmehride \
                                                        --job-name="${JOB_NAME}" \
                                                        --gpus-per-node=${NGPUS_PER_NODE} \
                                                        --ntasks-per-node=${NTASKS_PER_NODE} \
                                                        --mem=${MEM}G \
                                                        "${SLURM_SCRIPT}" \
                                                        "${lr}" \
                                                        "${reg}" \
                                                        "${local_batch_size}" \
                                                        "${opt}" \
                                                        "${s_temp}" \
                                                        "${e_temp}" \
                                                        "${s_scaler}" \
                                                        "${e_scaler}" \
                                                        "${init_sparsity}" \
                                                        "${target_sparsity}" \
                                                        "${COPY_DATA}" \
                                                        "${MASK_TILE_SIZE}" \
                                                        "${SPARSITY_TYPE}" \
                                                        "${PRUNING_METHOD}" \
                                                        "${MODEL_PREFIX}${MODEL_SIZE}${MODEL_POSTFIX}" \
                                                        "${FINE_TUNING_SEQUENCE_LENGTH}" \
                                                        "${weight_reg}" \
                                                        "${tile_str}" \
                                                        "${weight_strength}" \
                                                        "${s_tile_temp}" \
                                                        "${e_tile_temp}" \
                                                        "${s_tile_scaler}" \
                                                        "${e_tile_scaler}" \
                                                        "${CLUSTER}" \
                                                        "${PARALLELISM}" \
                                                        "${LEARNABLE_2_4_TILE}" \
                                                        "${MASKLLM_CHECKPOINT}"


                                                    ((job_count++))
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "--------------------------------------------------"
echo "Finished submitting ${job_count} jobs."