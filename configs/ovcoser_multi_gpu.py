has_test = True
deterministic = True
use_custom_worker_init = True
base_seed = 112358
log_interval = 20

# 多GPU设置
multi_gpu = True  # 启用多GPU训练
gpu_ids = [0, 1, 2, 3]  # 指定使用的GPU ID列表

# 每GPU的批次大小
__PER_GPU_BATCHSIZE = 2
__NUM_EPOCHS = 30

train = dict(
    batch_size=__PER_GPU_BATCHSIZE * len(gpu_ids),  # 总批次大小 = 每GPU批次大小 * GPU数量
    per_gpu_batch_size=__PER_GPU_BATCHSIZE,  # 每GPU的批次大小
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    num_iters=0,
    grad_acc_step=1,
    num_workers=4,  # 增加数据加载线程数以充分利用多GPU
    lr=0.00003 * len(gpu_ids),  # 根据GPU数量调整学习率
    sche_usebatch=False,
    optimizer=dict(
        mode="adamw",
        group_mode="finetune",
        set_to_none=True,
        cfg=dict(
            weight_decay=0.0005,
            diff_factor=0.1,
        ),
    ),
    scheduler=dict(
        warmup=dict(
            num_iters=0,
        ),
        mode="constant",
        cfg=dict(),
    ),
    input_hw=[384, 384],
)

test = dict(
    batch_size=__PER_GPU_BATCHSIZE * len(gpu_ids),  # 总批次大小 = 每GPU批次大小 * GPU数量
    per_gpu_batch_size=__PER_GPU_BATCHSIZE,  # 每GPU的批次大小
    num_workers=4,  # 增加数据加载线程数以充分利用多GPU
    save_results=True,
    input_hw=[384, 384],
)