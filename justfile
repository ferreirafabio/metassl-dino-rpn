# List available just commands
@list:
  just --list

# ---------------------------------------------------------------------------------------
# BASELINES - SIMSIAM ON CIFAR10
# ---------------------------------------------------------------------------------------

# Run pretraining DINO
@dino_pt EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_pretraining.sh

# Test pretraining DINO
@test_dino_pt EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_test_dino_pretraining.sh

# Run finetuning DINO
@dino_ft EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_finetuning.sh

# Test finetuning DINO
@test_dino_ft EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_test_dino_finetuning.sh

# Run NEPS DINO
@dino_neps EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps.sh

# Test NEPS DINO
@test_dino_neps EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_test_dino_neps.sh

# Run NEPS DINO Fabio
@dino_neps_fabio EXPERIMENT_NAME:
  #!/usr/bin/env zsh
  mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps_fabio.sh

# Test NEPS DINO Fabio
@dino_neps_fabio_test EXPERIMENT_NAME:
  #!/usr/bin/env zsh
  mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps_fabio_test.sh

# Test without NEPS DINO Fabio (Distributed fix)
@dino_wo_neps_fabio_test EXPERIMENT_NAME:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_wo_neps_fabio.sh

# Run NEPS DINO Fabio Sam (Distributed fix)
@dino_neps_distributed_fix EXPERIMENT_NAME:
  #!/usr/bin/env zsh
  mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps_fabio_sam.sh

# Test without NEPS DINO Finetuning Fabio (checking for semaphore issue)
@dino_wo_neps_finetuning_fabio_test EXPERIMENT_NAME:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_wo_neps_finetuning_fabio.sh
