module load apptainer

DATA_DIR=/scratch/mozaffar/data

CONTAINER_NAME="torch-jax"

# mkdir ${SCRATCH}/$CONTAINER_NAME.sif;
# tar -xf ${SCRATCH}/$CONTAINER_NAME.tar -C ${SCRATCH};
# mkdir ${SCRATCH}/$CONTAINER_NAME.sif/etc/pki;
# mkdir ${SCRATCH}/$CONTAINER_NAME.sif/etc/pki/tls;
# mkdir ${SCRATCH}/$CONTAINER_NAME.sif/etc/pki/tls/certs;
# cp /etc/ssl/certs/ca-bundle.crt ${SCRATCH}/$CONTAINER_NAME.sif/etc/pki/tls/certs/ca-bundle.crt;

# singularity exec \
#     --bind /tmp:/tmp \
#     --nv \
#     ${SCRATCH}/$CONTAINER_NAME.sif \
#     mkdir -p /home/mozaffar/data

singularity shell \
    --fakeroot \
    --bind $DATA_DIR:/scratch/mozaffar/patch-dev/data \
    --nv ${SCRATCH}/$CONTAINER_NAME.sif 