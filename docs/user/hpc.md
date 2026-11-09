# Running on HPC

Everything needed to run on a cluster is in the `HPC/` folder of the repository:

| File                           | Purpose                                                                |
| ------------------------------ | ---------------------------------------------------------------------- |
| `HPC/SBO_apptainer.def`        | Apptainer definition: Python 3.10 and the locked runtime dependencies. |
| `HPC/build_apptainer.sh`       | Builds `HPC/SBO_apptainer.sif` against the cluster's OpenMPI module.   |
| `HPC/example_SLURM_job.sbatch` | Minimal SLURM job that runs a script with MPI in the image.            |

The module names in the build script (`2024r1`, `openmpi/4.1.6`) are those of the cluster used for the thesis; adapt
them to yours.

## Running

The image does **not** contain the source code: bind-mount the repository at `/app` (`PYTHONPATH` inside the container
points to `/app/src`), so code changes take effect without rebuilding the image. The image also expects the host MPI at
the paths it was built against:

| Container path   | Host path                                                    |
| ---------------- | ------------------------------------------------------------ |
| `/app`           | The repository root.                                         |
| `/host-mpi-root` | The OpenMPI installation the image was built against.        |
| `/host-dep-lib`  | A directory with the non-system libraries `libmpi.so` needs. |

Launch with the host MPI, which starts one container per rank:

```bash
module load openmpi                 # adapt to your cluster
MPI_ROOT=$(ompi_info --path prefix -l 0 | awk '{print $NF}')

mpirun -n 4 apptainer exec \
    --bind "$PWD":/app \
    --bind "$MPI_ROOT":/host-mpi-root \
    --bind /path/to/dep-libs:/host-dep-lib \
    HPC/SBO_apptainer.sif \
    python /app/examples/04_multi_point_optimisation.py
```

The container limits BLAS/OpenMP threads to one per process (`OMP_NUM_THREADS=1` etc.), so the parallelism comes from
the MPI ranks only. See [Multi-point Optimisation](multi_point.md) for how the ranks are used.

### The Dependency Library Directory

`build_apptainer.sh` copies the libraries that `libmpi.so` needs (except system libraries such as `libc`) into a
temporary directory, which it deletes after the build. Recreate it for your jobs in the same way:

```bash
MPI_LIB=$(ompi_info --path libdir -l 0 | awk '{print $NF}')
DEP_LIBS=$HOME/sbo-dep-libs
mkdir -p "$DEP_LIBS"
ldd "$MPI_LIB/libmpi.so" | awk '{print $3}' | grep '^/' \
    | grep -vE 'libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|libutil\.so|libresolv\.so|libgcc_s\.so' \
    | grep -v "$MPI_LIB" | xargs -r cp -t "$DEP_LIBS"
```

### Job Script

`HPC/example_SLURM_job.sbatch` is a minimal job script. It stages the dependency libraries for the job, and then runs a
Python script on all ranks with the host MPI:

```bash
--8<-- "HPC/example_SLURM_job.sbatch"
```

Adapt the `#SBATCH` resources (and the partition and account, if your cluster needs them) and the OpenMPI module name.
Submit it from the repository root, optionally with the script to run:

```bash
sbatch HPC/example_SLURM_job.sbatch                               # runs examples/04_multi_point_optimisation.py
sbatch HPC/example_SLURM_job.sbatch examples/05_run_from_config/05_run_from_config.py
```

The log files are written per rank to `logs/` in the repository (see [Logging](results.md#logging)).

## Building the Image

From the repository root on a login node:

```bash
nohup ./HPC/build_apptainer.sh > build.log 2>&1 &
tail -f build.log                # follow the build
```

The image is written to `HPC/SBO_apptainer.sif`. Rebuild it whenever `uv.lock` changes.

### How the Image Is Built

The image installs exactly the runtime dependencies pinned in `uv.lock`, without test, docs or development tools, so it
matches the locked environment.

`mpi4py` is the exception: it is built from source inside the container against the **host's** OpenMPI, so MPI
communication goes through the cluster's own (interconnect-aware) MPI library. `build_apptainer.sh` therefore:

1. loads the cluster's OpenMPI module;
2. binds the OpenMPI installation into the build at `/host-mpi-root`, and the non-system libraries `libmpi.so` depends
    on at `/host-dep-lib`;
3. runs `apptainer build` from the repository root, so the definition file can copy in `pyproject.toml` and `uv.lock`.

!!! note "Licence of the image"

    The built image contains GPL-licensed software (the Linux base system). The definition file itself is part of this
    MIT-licensed repository; if you publish a built image, the GPL terms apply to it.
