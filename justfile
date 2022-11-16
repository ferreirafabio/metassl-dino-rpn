# List available just commands
@list:
  just --list

# ---------------------------------------------------------------------------------------
# BASELINES - SIMSIAM ON CIFAR10
# ---------------------------------------------------------------------------------------

# DINO RPN (NORMAL)
@dino_rpn EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS INVERT_GRADIENTS USE_RPN_OPTIMIZER SEPARATE_LOCAL_NET STN_MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},BATCH_SIZE={{BATCH_SIZE}},EPOCHS={{EPOCHS}},WARMUP_EPOCHS={{WARMUP_EPOCHS}},INVERT_GRADIENTS={{INVERT_GRADIENTS}},USE_RPN_OPTIMIZER={{USE_RPN_OPTIMIZER}},SEPARATE_LOCAL_NET={{SEPARATE_LOCAL_NET}},STN_MODE={{STN_MODE}} cluster/run_rpn.sh

# DINO RPN (NORMAL)
@dino_rpn_unbounded EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS INVERT_GRADIENTS USE_RPN_OPTIMIZER SEPARATE_LOCAL_NET STN_MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},BATCH_SIZE={{BATCH_SIZE}},EPOCHS={{EPOCHS}},WARMUP_EPOCHS={{WARMUP_EPOCHS}},INVERT_GRADIENTS={{INVERT_GRADIENTS}},USE_RPN_OPTIMIZER={{USE_RPN_OPTIMIZER}},SEPARATE_LOCAL_NET={{SEPARATE_LOCAL_NET}},STN_MODE={{STN_MODE}} cluster/run_rpn_unbounded.sh

# DINO RPN (NORMAL gtx3080)
@dino_rpn_3080 EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS INVERT_GRADIENTS USE_RPN_OPTIMIZER SEPARATE_LOCAL_NET STN_MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},BATCH_SIZE={{BATCH_SIZE}},EPOCHS={{EPOCHS}},WARMUP_EPOCHS={{WARMUP_EPOCHS}},INVERT_GRADIENTS={{INVERT_GRADIENTS}},USE_RPN_OPTIMIZER={{USE_RPN_OPTIMIZER}},SEPARATE_LOCAL_NET={{SEPARATE_LOCAL_NET}},STN_MODE={{STN_MODE}} cluster/run_rpn_3080.sh

# DINO RPN (SMALL RPNLR)
@dino_rpn_small_rpnlr EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS INVERT_GRADIENTS USE_RPN_OPTIMIZER SEPARATE_LOCAL_NET STN_MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},BATCH_SIZE={{BATCH_SIZE}},EPOCHS={{EPOCHS}},WARMUP_EPOCHS={{WARMUP_EPOCHS}},INVERT_GRADIENTS={{INVERT_GRADIENTS}},USE_RPN_OPTIMIZER={{USE_RPN_OPTIMIZER}},SEPARATE_LOCAL_NET={{SEPARATE_LOCAL_NET}},STN_MODE={{STN_MODE}} cluster/run_rpn_small_rpnlr.sh


# DINO RPN (DEEP)
@dino_rpn_deep EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS INVERT_GRADIENTS USE_RPN_OPTIMIZER SEPARATE_LOCAL_NET STN_MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},BATCH_SIZE={{BATCH_SIZE}},EPOCHS={{EPOCHS}},WARMUP_EPOCHS={{WARMUP_EPOCHS}},INVERT_GRADIENTS={{INVERT_GRADIENTS}},USE_RPN_OPTIMIZER={{USE_RPN_OPTIMIZER}},SEPARATE_LOCAL_NET={{SEPARATE_LOCAL_NET}},STN_MODE={{STN_MODE}} cluster/run_rpn_deep.sh

# DINO RPN (separate localization backbones)
@dino_rpn_separate_localization_backbones EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS INVERT_GRADIENTS USE_RPN_OPTIMIZER SEPARATE_LOCAL_NET STN_MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},BATCH_SIZE={{BATCH_SIZE}},EPOCHS={{EPOCHS}},WARMUP_EPOCHS={{WARMUP_EPOCHS}},INVERT_GRADIENTS={{INVERT_GRADIENTS}},USE_RPN_OPTIMIZER={{USE_RPN_OPTIMIZER}},SEPARATE_LOCAL_NET={{SEPARATE_LOCAL_NET}},STN_MODE={{STN_MODE}} cluster/run_rpn_separate_localization_backbones.sh

# region proposal dino evaluation
@dino_rpn_eval EXPERIMENT_NAME SEED:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_linear_evaluation.sh

# DINO RPN (full pretrained rpn)
@dino_rpn_full_train_pretrained_rpn EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS SEPARATE_LOCAL_NET STN_MODE RPN_PRETRAINED_WEIGHTS:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},BATCH_SIZE={{BATCH_SIZE}},EPOCHS={{EPOCHS}},WARMUP_EPOCHS={{WARMUP_EPOCHS}},SEPARATE_LOCAL_NET={{SEPARATE_LOCAL_NET}},STN_MODE={{STN_MODE}},RPN_PRETRAINED_WEIGHTS={{RPN_PRETRAINED_WEIGHTS}} cluster/run_rpn_full_train_pretrained_rpn.sh
