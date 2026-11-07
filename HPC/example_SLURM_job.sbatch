#!/bin/bash
# Minimal SLURM job: runs a Python script with MPI in the SBO Apptainer image.
# Submit from the repository root:  sbatch HPC/example_SLURM_job.sbatch [script]
# The script defaults to the multi-point example. Adapt the #SBATCH lines and the module name to your cluster.
#SBATCH --job-name=sbo
#SBATCH --ntasks=4                  # number of MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
##SBATCH --partition=<partition>   # uncomment if your cluster needs these
##SBATCH --account=<account>

set -e

SCRIPT=${1:-examples/04_multi_point_optimisation.py}   # path relative to the repository root

module load openmpi                 # the OpenMPI module the image was built against (see build_apptainer.sh)
MPI_ROOT=$(ompi_info --path prefix -l 0 | awk '{print $NF}')
MPI_LIB=$(ompi_info --path libdir -l 0 | awk '{print $NF}')

# The non-system libraries libmpi.so needs, mounted at /host-dep-lib (as in build_apptainer.sh). The directory is in
# the repository, so it is reachable from every node of a multi-node job.
DEP_LIBS="$PWD/.dep-libs-$SLURM_JOB_ID"
mkdir -p "$DEP_LIBS"
trap 'rm -rf "$DEP_LIBS"' EXIT
ldd "$MPI_LIB/libmpi.so" | awk '{print $3}' | grep '^/' \
    | grep -vE 'libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|libutil\.so|libresolv\.so|libgcc_s\.so' \
    | grep -v "$MPI_LIB" | xargs -r cp -t "$DEP_LIBS"

# One container per rank; the repository is mounted at /app, and logs/ is written there
mpirun -n "$SLURM_NTASKS" apptainer exec \
    --bind "$PWD":/app \
    --bind "$MPI_ROOT":/host-mpi-root \
    --bind "$DEP_LIBS":/host-dep-lib \
    --pwd /app \
    HPC/SBO_apptainer.sif \
    python "/app/$SCRIPT"
