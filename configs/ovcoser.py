has_test = True
deterministic = True
use_custom_worker_init = True
base_seed = 112358
log_interval = 20

use_prompts = True
use_visual_prompts = True

prompt_config = dict(
    LOCATION="prepend",  # 提示位置: "prepend" 或 "add"
    INITIATION="random",  # 初始化方式: "random"
    NUM_TOKENS=10,  # 提示token数量
    DEEP=True,  # 是否使用深层提示
    DEEP_SHARED=False,  # 是否共享深层提示
    DROPOUT=0.1,  # dropout率
    PROJECT=-1,  # 投影维度，-1表示不投影

    INSTANCE_PROMPT = True,  # 启用实例提示
    INSTANCE_P_LEN = 5,  # 实例提示长度
    INSTANCE_DROPOUT = 0.1,  # 实例提示dropout率
    
    # VFPT (Visual Fourier Prompt Tuning) 参数
    FOURIER_PERCENTAGE=0.5,  # 傅里叶变换百分比：0.0-1.0，1.0表示所有提示都应用傅里叶变换
    FOURIER_DIMENSION="all",  # 傅里叶变换维度："all"(所有维度)、"hidden"(隐藏维度)、"sequence"(序列维度)
    FOURIER_TYPE="fixed_linear",  # 傅里叶变换类型："fixed_linear"(固定线性变换)
    FOURIER_ADDITION=False,  # 是否添加额外的傅里叶提示
    FOURIER_ADDITION_NUM=0,  # 额外傅里叶提示的数量
    FOURIER_FIRST_LAYER=True,  # 是否在第一层应用傅里叶变换
    FOURIER_HALF="none",  # 半傅里叶变换模式："none"、"former"、"later"
    FOURIER_LOCATION="prepend",  # 傅里叶提示位置："prepend"(在前)、"random"(随机)、其他(在后)
    MIXED=False  # 是否混合使用傅里叶和普通提示
)


__BATCHSIZE =4
__NUM_EPOCHS =30

train = dict(
    batch_size=__BATCHSIZE,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    num_iters=0,
    grad_acc_step=1,
    num_workers=2,
    lr=0.00003,
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
    input_hw=[336, 336],
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    save_results=True,
    input_hw=[336, 336],
)
