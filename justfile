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


# region proposal dino
@dino_rpn EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS INVERT_GRADIENTS USE_RPN_OPTIMIZER SEPARATE_LOCAL_NET STN_MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},BATCH_SIZE={{BATCH_SIZE}},EPOCHS={{EPOCHS}},WARMUP_EPOCHS={{WARMUP_EPOCHS}},INVERT_GRADIENTS={{INVERT_GRADIENTS}},USE_RPN_OPTIMIZER={{USE_RPN_OPTIMIZER}},SEPARATE_LOCAL_NET={{SEPARATE_LOCAL_NET}},STN_MODE={{STN_MODE}} cluster/run_rpn.sh


# region proposal dino with separate localization backbones
@dino_rpn_separate_localization_backbones EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS INVERT_GRADIENTS USE_RPN_OPTIMIZER SEPARATE_LOCAL_NET STN_MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},BATCH_SIZE={{BATCH_SIZE}},EPOCHS={{EPOCHS}},WARMUP_EPOCHS={{WARMUP_EPOCHS}},INVERT_GRADIENTS={{INVERT_GRADIENTS}},USE_RPN_OPTIMIZER={{USE_RPN_OPTIMIZER}},SEPARATE_LOCAL_NET={{SEPARATE_LOCAL_NET}},STN_MODE={{STN_MODE}} cluster/run_rpn_separate_localization_head.sh

# region proposal dino evaluation
@dino_rpn_eval EXPERIMENT_NAME SEED:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}}  cluster/submit_imagenet_dino_linear_evaluation.sh
