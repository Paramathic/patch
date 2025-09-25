module load apptainer

DATA_DIR=/home/mozaffar/projects/def-mmehride/mozaffar/data
CONTAINER_NAME=torch-one-shot

rm -rf $SLURM_TMPDIR/$CONTAINER_NAME.sif;
mkdir ${SLURM_TMPDIR}/$CONTAINER_NAME.sif;
tar -xf /home/mozaffar/projects/def-mmehride/mozaffar/$CONTAINER_NAME.tar -C $SLURM_TMPDIR;
mkdir ${SLURM_TMPDIR}/$CONTAINER_NAME.sif/etc/pki;
mkdir ${SLURM_TMPDIR}/$CONTAINER_NAME.sif/etc/pki/tls;
mkdir ${SLURM_TMPDIR}/$CONTAINER_NAME.sif/etc/pki/tls/certs;
cp /etc/ssl/certs/ca-bundle.crt ${SLURM_TMPDIR}/$CONTAINER_NAME.sif/etc/pki/tls/certs/ca-bundle.crt;

singularity exec \
    --bind $PWD:/home/mozaffar \
    --bind $SLURM_TMPDIR:/tmp \
    --nv \
    ${SLURM_TMPDIR}/$CONTAINER_NAME.sif \
    mkdir -p /home/mozaffar/data

singularity shell \
    --bind $PWD:/home/mozaffar \
    --bind $SLURM_TMPDIR:/tmp \
    --bind $DATA_DIR:/home/mozaffar/data \
    --nv ${SLURM_TMPDIR}/$CONTAINER_NAME.sif 