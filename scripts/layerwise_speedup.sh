export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"


MODEL="meta-llama/Llama-2-7b-hf"
LOCAL_CHECKPOINT="tiled_models/Llama-2-7b-hf_LR0.0005_REG7.0_OPTadamw_torch_Sparsity0.5-0.45_T4.0-0.05_S100.0-500.0_STR3.0_WREG20.0.pt"
SEQLEN=1


SPARSE_B="--sparse_b"
PROFILE=true
NSIGHT_SYSTEMS=false


if [ $PROFILE = false ]; then
    PYTHON="python"
fi

if [ $NSIGHT_SYSTEMS = true ]; then
    PROFILER_CMD="nsys profile --cuda-graph=node -w true -t cuda,nvtx,osrt,cudnn,cublas -s none"
    PROFILING_PATH="profile/systems"
else
    PROFILER_CMD="ncu --target-processes all --graph-profiling node "
    PROFILING_PATH="profile/compute"
fi

for DIMS in "(4096, 4096)" "(8192, 8192)" #"(4096, 11008)" "(1108, 4096)"  "(8192, 28672)" "(28672, 8192)"
do
    D_IN=$(echo $DIMS | cut -d',' -f1 | tr -d '() ')
    D_OUT=$(echo $DIMS | cut -d',' -f2 | tr -d '() ')

    for BATCH_SIZE in 16 32 64 4096 8096
    do
        for SPARSITY_RATIO in 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05 0.00
            do
            # $PYTHON -m speedup_tile.layerwise_speedup \
            #     --model $MODEL \
            #     --seqlen $SEQLEN \
            #     --batch_size $BATCH_SIZE \
            #     --local_files_only \
            #     --saved_model_path $LOCAL_CHECKPOINT


                if [ $PROFILE = true ]; then
                    PYTHON="$PROFILER_CMD -o $PROFILING_PATH/stoicc_${BATCH_SIZE}_${D_IN}_${D_OUT}_${SPARSITY_RATIO}${SPARSE_B} python"
                fi
                

                $PYTHON -m speedup_tile.synthetic_layerwise_speedup \
                    --d_in $D_IN \
                    --d_out $D_OUT \
                    --batch_size $BATCH_SIZE \
                    --num_experiments 3 \
                    --num_layers 1 \
                    --sparsity_ratio $SPARSITY_RATIO \
                    --num_load_balances 1 \
                    $SPARSE_B
            done

            if [ $PROFILE = true ]; then
                PYTHON="$PROFILER_CMD -o $PROFILING_PATH/cutlass_${BATCH_SIZE}_${D_IN}_${D_OUT}_0.50 python"
            fi
            $PYTHON -m speedup_tile.synthetic_layerwise_speedup \
                    --d_in $D_IN \
                    --d_out $D_OUT \
                    --batch_size $BATCH_SIZE \
                    --num_experiments 3 \
                    --num_layers 1 \
                    --sparsity_ratio 0.50 \
                    --backend "cutlass"

            if [ $PROFILE = true ]; then
                PYTHON="$PROFILER_CMD -o $PROFILING_PATH/cusparselt_${BATCH_SIZE}_${D_IN}_${D_OUT}_0.50 python"
            fi

            $PYTHON -m speedup_tile.synthetic_layerwise_speedup \
                    --d_in $D_IN \
                    --d_out $D_OUT \
                    --batch_size $BATCH_SIZE \
                    --num_experiments 3 \
                    --num_layers 1 \
                    --sparsity_ratio 0.50 \
                    --backend "cusparselt"

            if [ $PROFILE = true ]; then
                PYTHON="$PROFILER_CMD -o $PROFILING_PATH/cublas_${BATCH_SIZE}_${D_IN}_${D_OUT}_0.00 python"
            fi

            $PYTHON -m speedup_tile.synthetic_layerwise_speedup \
                    --d_in $D_IN \
                    --d_out $D_OUT \
                    --batch_size $BATCH_SIZE \
                    --num_experiments 3 \
                    --num_layers 1 \
                    --sparsity_ratio 0.00 \
                    --backend "cublas"
    done
done