export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export MASTER_PORT=29501
HF_TOKEN="--hf_token HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

MASK_FINE_TUNE_LR="$1"
MASK_REG_FACTOR="$2"
LOCAL_BATCH_SIZE="$3"
OPTIMIZER="$4"
START_TEMP="$5"
END_TEMP="$6"
START_SCALER="$7"
END_SCALER="$8"
INITIAL_SPARSITY="$9"
TARGET_SPARSITY="${10}"
MASK_TILE_SIZE="${11:-1,1}" # Default to "1,1" if not provided
SPARSITY_TYPE="${12:-unstructured}" # Default to "unstructured" if not provided
PRUNING_METHOD="${13:-wanda}" # Default to "wanda" if not provided
MODEL_NAME="${14:-meta-llama/Llama-3.2-1B}" # Default to "meta-llama/Llama-3.2-1B" if not provided
FINE_TUNING_SEQUENCE_LENGTH="${15:-1024}" # Default to 1024 if not provided
WEIGHT_REG="${16:-1e-3}"
TILE_STRENGTH="${17:-0}"
WEIGHT_STRENGTH="${18:-0}"
TILE_START_TEMP="${19:-0}"
TILE_END_TEMP="${20:-0}"
TILE_START_SCALER="${21:-0}"
TILE_END_SCALER="${22:-0}"
PARALLELISM="${23:-data_parallel}"
LEARNABLE_2_4_TILE="${24:-false}" # Default to false if not provided
MASKLLM_CHECKPOINT="${25:-'Vinnnf/LLaMA-2-7B-MaskLLM-C4'}" # Default to 'Vinnnf/LLaMA-2-7B-MaskLLM-C4' if not provided

if [ "$LEARNABLE_2_4_TILE" = true ]; then
    LEARNABLE_2_4_TILE="--joint_optim"
else
    LEARNABLE_2_4_TILE=""
fi

echo "--- Starting run_experiment.sh ---"
echo "  LR         : ${MASK_FINE_TUNE_LR}"
echo "  Reg Factor : ${MASK_REG_FACTOR}"
echo "  Batch Size : ${LOCAL_BATCH_SIZE}"
echo "  Optimizer  : ${OPTIMIZER}"
echo "  Initial Sparsity  : ${INITIAL_SPARSITY}"
echo "  Target Sparsity  : ${TARGET_SPARSITY}"
echo "  Temperature : ${START_TEMP} - ${END_TEMP}"
echo "  Gumbel Scaler : ${START_SCALER} - ${END_SCALER}"
echo "  Weight Reg : ${WEIGHT_REG}"
echo "  Tile Strength : ${TILE_STRENGTH}"
echo "  Weight Strength : ${WEIGHT_STRENGTH}"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "$PARALLELISM" = "data_parallel" ]; then
    echo "Using data parallelism with $NUM_GPUS GPUs"
    STARTER_CMD="torchrun --nproc_per_node=$NUM_GPUS --rdzv_endpoint=localhost:29500"
elif [ "$PARALLELISM" = "model_parallel" ]; then
    echo "Using model parallelism"
    STARTER_CMD="accelerate launch --num_processes=1 --mixed_precision=bf16"
else
    echo "Unknown parallelism type: $PARALLELISM. Defaulting to data parallelism."
    STARTER_CMD="torchrun --nproc_per_node=$NUM_GPUS --rdzv_endpoint=localhost:29500"
fi


NUM_CALIBRATION_SAMPLES=128
LOCAL_FILES_ONLY='--local_files_only'
SHIFT_ZERO_METRICS='--shift_zero_metrics'
TEST_LMHARNESS='--test_lmharness'
EVALUATE_PERPLEXITY='--evaluate_perplexity'
GRAD_CHECKPOINT='--grad_checkpoint'
LEARNABLE_MASK='--learnable_mask'

# Set the number of OpenMP threads to number of CPU cores per GPU
export OMP_NUM_THREADS=$(nproc --all)

$STARTER_CMD main_learnable_mask.py \
    --model $MODEL_NAME \
    --prune_method $PRUNING_METHOD \
    --sparsity_ratio $INITIAL_SPARSITY \
    --target_sparsity_ratio $TARGET_SPARSITY \
    --sparsity_type $SPARSITY_TYPE \
    $SHIFT_ZERO_METRICS \
    $TEST_LMHARNESS \
    $EVALUATE_PERPLEXITY \
    $LOCAL_FILES_ONLY \
    --nsamples $NUM_CALIBRATION_SAMPLES \
    $HF_TOKEN \
    $GRAD_CHECKPOINT \
    --lr $MASK_FINE_TUNE_LR \
    --sparse_reg $MASK_REG_FACTOR \
    --wandb \
    --local_bs $LOCAL_BATCH_SIZE \
    --temp_schedule_2_4 $START_TEMP $END_TEMP  \
    --scaler_schedule_2_4 $START_SCALER $END_SCALER \
    --temp_schedule_tile $TILE_START_TEMP $TILE_END_TEMP \
    --scaler_schedule_tile $TILE_START_SCALER $TILE_END_SCALER \
    $LEARNABLE_MASK \
    --mask_tile_size $MASK_TILE_SIZE \
    --fine_tuning_sequence_length $FINE_TUNING_SEQUENCE_LENGTH \
    --prior_strength_2_4 $WEIGHT_STRENGTH \
    --weight_reg $WEIGHT_REG \
    --prior_strength_tile $TILE_STRENGTH \
    $LEARNABLE_2_4_TILE \
    --maskllm_checkpoint $MASKLLM_CHECKPOINT 