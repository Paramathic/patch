#!/bin/bash
#SBATCH --ntasks-per-node=12 # Request enough CPUs for data loading, etc.
#SBATCH --nodes=1
#SBATCH -t 6:30:00
#SBATCH --account=def-mmehride
#SBATCH --job-name=learn_mask # Base name, will be overridden by submit_jobs.

ARG_LR="$1"
ARG_REG_FACTOR="$2"
ARG_LOCAL_BATCH_SIZE="$3"
ARG_OPTIMIZER="$4"
ARG_START_TEMP="$5"
ARG_END_TEMP="$6"
ARG_START_SCALER="$7"
ARG_END_SCALER="$8"
ARG_INIT_SPARSITY="$9"
ARG_TARGET_SPARSITY="${10}"
ARG_COPY_DATA="${11:-false}" # Default to false if not provided
ARG_MASK_TILE_SIZE="${12:-1,1}" # Default to "1,1" if not provided
ARG_SPARSITY_TYPE="${13:-unstructured}" # Default to "unstructured" if not provided
ARG_PRUNING_METHOD="${14:-wanda}" # Default to "wanda" if not provided
ARG_MODEL_NAME="${15:-'meta-llama/Llama-3.2-1B'}" # Default to "meta-llama/Llama-3.2-1B" if not provided
ARG_FINE_TUNING_SEQUENCE_LENGTH="${16:-1024}" # Default to 1024 if not provided
ARG_WEIGHT_REG="${17:-1e-3}"
ARG_TILE_STR="${18:-0}"
ARG_WEIGHT_STR="${19:-0}"
ARG_TILE_START_TEMP="${20:-0}"
ARG_TILE_END_TEMP="${21:-0}"
ARG_TILE_START_SCALER="${22:-0}"
ARG_TILE_END_SCALER="${23:-0}"
ARG_CLUSTER="${24:-'trillium'}"
ARG_PARALLELISM="${25:-data_parallel}"
ARG_LEARNABLE_2_4_TILE="${26:-false}" # Default to false if not provided
ARG_MASKLLM_CHECKPOINT="${27:-'Vinnnf/LLaMA-2-7B-MaskLLM-C4'}" # Default to 'Vinnnf/LLaMA-2-7B-MaskLLM-C4' if not provided
SCRIPT_TO_RUN=scripts/run_patch_args.sh


echo "Starting SLURM job $SLURM_JOB_ID for job name $SLURM_JOB_NAME"
echo "Using SLURM temporary directory: $SLURM_TMPDIR"
echo "Received Parameters: LR=${ARG_LR}, RegFactor=${ARG_REG_FACTOR}, BatchSize=${ARG_LOCAL_BATCH_SIZE}, Optimizer=${ARG_OPTIMIZER}"

# --- Setup Environment ---
module load apptainer 
export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data" 
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export MASTER_PORT=29501
export OMP_NUM_THREADS=12


mkdir -p "$HF_HOME"

USERNAME=$(whoami)

if [ "$ARG_CLUSTER" = "trillium" ]; then
    echo "Running on Trillium cluster"
    CONTAINER_NAME=torch-jax
    DATA_DIR_SRC="${SCRATCH}/data"
    # Additional setup for Trillium can go here
    SINGULARITY_CMD="singularity exec \
        --fakeroot \
        --bind $DATA_DIR_SRC:$PWD/data \
        --nv ${SCRATCH}/$CONTAINER_NAME.sif "
elif [ "$ARG_CLUSTER" = "narval" ]; then
    echo "Running on Narval cluster"
    # --- Data and Container Preparation ---
    if [ "$ARG_COPY_DATA" = true ]; then
        echo "Copying data to SLURM_TMPDIR..."
        DATA_DIR_SRC="/home/${USERNAME}/projects/def-mmehride/${USERNAME}/data"
        DATA_DIR_TMP="$SLURM_TMPDIR/data"
        cp -r "$DATA_DIR_SRC" "$SLURM_TMPDIR/"
        echo "Data copied to $DATA_DIR_TMP"
        CONTAINER_NAME=torch-one-shot

        SINGULARITY_CMD="singularity exec \
            --bind $PWD:/home/${USERNAME} \
            --bind $SLURM_TMPDIR:/tmp \
            --bind $DATA_DIR_TMP:/home/${USERNAME}/data \
            --nv ${SLURM_TMPDIR}/$CONTAINER_NAME.sif "
    else
        echo "Skipping data copy as per user request."
        DATA_DIR_TMP="/home/${USERNAME}/projects/def-mmehride/${USERNAME}/data" # Use the original data directory
    fi

    echo "Preparing container..."
    rm -rf $SLURM_TMPDIR/torch-one-shot.sif;
    mkdir ${SLURM_TMPDIR}/torch-one-shot.sif;
    tar -xf /home/${USERNAME}/projects/def-mmehride/${USERNAME}/torch-one-shot.tar -C $SLURM_TMPDIR;
    mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki;
    mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls;
    mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs;
    cp /etc/ssl/certs/ca-bundle.crt ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs/ca-bundle.crt;
else
    echo "Unknown cluster specified: $ARG_CLUSTER. Exiting."
    exit 1
fi



# --- Execute inside Singularity ---
echo "Executing the script inside Singularity..." 

    
bash ${SINGULARITY_CMD} \
     bash "${SCRIPT_TO_RUN}" "${ARG_LR}" "${ARG_REG_FACTOR}" "${ARG_LOCAL_BATCH_SIZE}" "${ARG_OPTIMIZER}" \
     "${ARG_START_TEMP}" "${ARG_END_TEMP}" "${ARG_START_SCALER}" "${ARG_END_SCALER}" "${ARG_INIT_SPARSITY}" \
     "${ARG_TARGET_SPARSITY}" "${ARG_MASK_TILE_SIZE}" "${ARG_SPARSITY_TYPE}" "${ARG_PRUNING_METHOD}" \
     "${ARG_MODEL_NAME}" "${ARG_FINE_TUNING_SEQUENCE_LENGTH}" "${ARG_WEIGHT_REG}" \
     "${ARG_TILE_STR}" "${ARG_WEIGHT_STR}" "${ARG_TILE_START_TEMP}" "${ARG_TILE_END_TEMP}" \
     "${ARG_TILE_START_SCALER}" "${ARG_TILE_END_SCALER}" "${ARG_PARALLELISM}" "${ARG_LEARNABLE_2_4_TILE}" \
     "${ARG_MASKLLM_CHECKPOINT}"

echo "Singularity execution finished successfully."
echo "SLURM Job $SLURM_JOB_ID finished."