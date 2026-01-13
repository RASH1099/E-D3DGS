_base_ = './default.py'
ModelHiddenParams = dict(
    min_embeddings = 63,
    max_embeddings = 315,
    c2f_temporal_iter = 20000,
    total_num_frames = 624,
)

OptimizationParams = dict(
    maxtime = 624,
    iterations = 30_000,
    densify_until_iter = 30_000,
    position_lr_max_steps = 30_000,
    deformation_lr_max_steps = 30_000,
)