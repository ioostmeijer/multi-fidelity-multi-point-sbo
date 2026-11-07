#!/bin/bash
# Builds HPC/SBO_apptainer.sif from HPC/SBO_apptainer.def against the cluster's OpenMPI.
# Can be called from any directory: it switches to the repository root, because the
# definition file copies pyproject.toml and uv.lock from there.
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

# The modules of the cluster used for the thesis. On another cluster, load
# its OpenMPI module instead, and load the same module in the job script.
module purge
module load 2024r1
module load openmpi/4.1.6

MPI_LIB=$(ompi_info --path libdir -l 0 | awk '{print $NF}')
MPI_INC=$(ompi_info --path incdir -l 0 | awk '{print $NF}')
MPI_BIN=$(ompi_info --path bindir -l 0 | awk '{print $NF}')
MPI_SHARE=$(ompi_info --path datadir -l 0 | awk '{print $NF}')
MPI_ROOT=$(ompi_info --path prefix -l 0 | awk '{print $NF}')

# Stage all libmpi.so dependencies, excluding system libs that must
# come from the container (libc, libm, libpthread, libdl, librt, etc.)
DEP_STAGE=$(mktemp -d)

SYSTEM_LIBS="libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|libutil\.so|libresolv\.so|libgcc_s\.so"

ldd "$MPI_LIB/libmpi.so" | awk '{print $3}' | grep '^/' | while read lib; do
    # Skip system libs to avoid glibc collision in container
    basename "$lib" | grep -qE "$SYSTEM_LIBS" && continue
    # Skip libs already in MPI_LIB (covered by /host-mpi-lib)
    dirname "$lib" | grep -q "$MPI_LIB" && continue
    [ -f "$lib" ] && cp "$lib" "$DEP_STAGE/"
done

echo "MPI_SHARE: $MPI_SHARE"
echo "MPI_BIN: $MPI_BIN"
echo "MPI_INC: $MPI_INC"
echo "MPI_LIB: $MPI_LIB"
echo "DEP_STAGE contents:"
ls "$DEP_STAGE"

apptainer build \
    --bind "$MPI_ROOT":/host-mpi-root \
    --bind "$DEP_STAGE":/host-dep-lib \
    HPC/SBO_apptainer.sif HPC/SBO_apptainer.def

status=$?
rm -rf "$DEP_STAGE"
echo "Cleaned up staging dir"
echo "Build finished with exit code $status"
exit $status
