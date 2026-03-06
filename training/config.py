"""
YOLO Pose 训练配置模块。

包含模型大小选项、硬件预设、训练超参数和数据增强参数。
所有参数可通过命令行或直接调用覆盖。
"""

from pathlib import Path

import torch

# ──────────────────────────────────────────────────────────────
# 模型大小选项
# ──────────────────────────────────────────────────────────────
PRETRAINED_DIR = Path(__file__).resolve().parent.parent / "yolo" / "pretrained"

MODEL_SIZES = {
    "nano":   str(PRETRAINED_DIR / "yolov8n-pose.pt"),
    "small":  str(PRETRAINED_DIR / "yolov8s-pose.pt"),
    "medium": str(PRETRAINED_DIR / "yolov8m-pose.pt"),
    "large":  str(PRETRAINED_DIR / "yolov8l-pose.pt"),
    "xlarge": str(PRETRAINED_DIR / "yolov8x-pose.pt"),
}

DEFAULT_MODEL_SIZE = "small"

# ──────────────────────────────────────────────────────────────
# 硬件预设
# ──────────────────────────────────────────────────────────────
HARDWARE_PROFILES = {
    "cpu": {
        "device": "cpu",
        "batch": 8,
        "workers": 2,
        "description": "Intel Mac / 无GPU 环境",
    },
    "mps": {
        "device": "mps",
        "batch": 16,
        "workers": 4,
        "description": "Apple Silicon Mac (M1/M2/M3/M4)",
    },
    "gpu_low": {
        "device": "0",
        "batch": 16,
        "workers": 4,
        "description": "NVIDIA GPU (4-6GB 显存)",
    },
    "gpu": {
        "device": "0",
        "batch": 32,
        "workers": 8,
        "description": "NVIDIA GPU (8GB+ 显存)",
    },
}


def detect_device():
    """自动检测最佳训练设备。

    检测优先级：CUDA > MPS > CPU。

    Returns:
        硬件预设名称（str）。
    """
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        return "gpu" if vram_gb >= 8 else "gpu_low"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_hardware_profile(profile_name=None):
    """获取硬件预设配置。

    Args:
        profile_name: 预设名称。None 则自动检测。

    Returns:
        dict，包含 device / batch / workers 等配置。
    """
    if profile_name is None:
        profile_name = detect_device()
    if profile_name not in HARDWARE_PROFILES:
        raise ValueError(
            f"未知硬件预设 '{profile_name}'，"
            f"可选：{list(HARDWARE_PROFILES.keys())}"
        )
    return HARDWARE_PROFILES[profile_name]


# ──────────────────────────────────────────────────────────────
# 训练超参数
# ──────────────────────────────────────────────────────────────
TRAINING_PARAMS = {
    # 训练轮数与早停
    "epochs": 200,
    "patience": 50,

    # 图像尺寸
    "imgsz": 640,

    # 优化器
    "optimizer": "AdamW",
    "lr0": 0.001,          # 初始学习率（AdamW 比 SGD 更小）
    "lrf": 0.01,           # 最终学习率 = lr0 * lrf
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # 学习率调度
    "warmup_epochs": 5.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,        # 余弦退火调度

    # 损失权重
    "box": 7.5,            # 边界框损失
    "cls": 0.5,            # 分类损失（单类任务可适当降低）
    "pose": 12.0,          # 关键点坐标损失
    "kobj": 1.0,           # 关键点目标性损失

    # 其他
    "close_mosaic": 10,    # 最后 N 轮关闭 mosaic
    "val": True,
    "plots": True,
    "save_period": 20,     # 每 N 轮保存一次 checkpoint
}

# ──────────────────────────────────────────────────────────────
# 数据增强参数
# ──────────────────────────────────────────────────────────────
AUGMENTATION_PARAMS = {
    # 翻转
    "fliplr": 0.5,         # 水平翻转概率（需配合正确 flip_idx）
    "flipud": 0.0,         # 不做垂直翻转（上下翻转后透视不合理）

    # Mosaic & Mixup
    "mosaic": 1.0,         # Mosaic 增强概率
    "mixup": 0.1,          # Mixup 小概率，增强泛化

    # 几何变换
    "degrees": 10.0,       # 旋转范围 ±10°
    "translate": 0.1,      # 平移比例
    "scale": 0.5,          # 缩放范围 [1-scale, 1+scale]
    "shear": 2.0,          # 剪切角度
    "perspective": 0.0005, # 透视变换（极小值）

    # 颜色空间
    "hsv_h": 0.015,        # 色调抖动
    "hsv_s": 0.7,          # 饱和度抖动
    "hsv_v": 0.4,          # 亮度抖动

    # 随机擦除
    "erasing": 0.3,        # 随机擦除概率（模拟球员遮挡）
}


def build_train_args(
    data_yaml,
    model_size=None,
    profile=None,
    project=None,
    name="badminton_court",
    resume=False,
    **overrides,
):
    """构建完整的 YOLO 训练参数字典。

    Args:
        data_yaml: data.yaml 文件路径。
        model_size: 模型大小名称（nano/small/medium/large/xlarge）。
        profile: 硬件预设名称。None 则自动检测。
        project: 训练输出目录。None 则使用默认（yolo/runs）。
        name: 训练运行名称。
        resume: 是否从上次中断处继续训练。
        **overrides: 覆盖任意参数。

    Returns:
        (model_weight, train_kwargs) 元组。
    """
    if model_size is None:
        model_size = DEFAULT_MODEL_SIZE
    if model_size not in MODEL_SIZES:
        raise ValueError(
            f"未知模型大小 '{model_size}'，"
            f"可选：{list(MODEL_SIZES.keys())}"
        )

    hw = get_hardware_profile(profile)

    args = {}
    args["data"] = str(data_yaml)
    args["device"] = hw["device"]
    args["batch"] = hw["batch"]
    args["workers"] = hw["workers"]
    if project is None:
        project = str(Path(__file__).resolve().parent.parent / "yolo" / "runs")
    args["project"] = project
    args["name"] = name
    args["exist_ok"] = True
    args["resume"] = resume

    args.update(TRAINING_PARAMS)
    args.update(AUGMENTATION_PARAMS)
    args.update(overrides)

    model_weight = MODEL_SIZES[model_size]
    return model_weight, args
