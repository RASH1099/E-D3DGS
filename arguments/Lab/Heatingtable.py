_base_ = './default.py'
ModelHiddenParams = dict(
    min_embeddings = 35,
    max_embeddings = 175,
    c2f_temporal_iter = 20000,
    total_num_frames = 345,
)

OptimizationParams = dict(
    maxtime = 345,
    iterations = 30_000,
    densify_until_iter = 30_000,
    position_lr_max_steps = 30_000,
    deformation_lr_max_steps = 30_000,
)