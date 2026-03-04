"""
YOLO Pose 主训练脚本。

用法示例：
    # 默认配置（small 模型，自动检测设备）
    python -m training.train --dataset-dir BadmintonCourtDetection.yolov8

    # 指定模型大小和硬件预设
    python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 \
        --model-size medium --profile mps

    # 覆盖特定参数
    python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 \
        --epochs 100 --batch 8

    # 恢复训练
    python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 --resume
"""

import argparse
import sys
from pathlib import Path

from training.config import (
    DEFAULT_MODEL_SIZE,
    HARDWARE_PROFILES,
    MODEL_SIZES,
    build_train_args,
    detect_device,
)
from training.prepare_data import fix_data_yaml, print_report, validate_dataset
from training.split_dataset import split_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Pose 关键点训练（球场/球网）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 球场模型训练
  python -m training.train --dataset-dir BadmintonCourtDetection.yolov8
  python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 --model-size medium --profile mps

  # 球网模型训练
  python -m training.train --dataset-dir net.v1i.yolov8 --model-type net
        """,
    )

    # 必需参数
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="数据集根目录路径",
    )

    # 模型类型
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["court", "net"],
        default="court",
        help="模型类型：court(球场22点) 或 net(球网4点)，默认 court",
    )

    # 模型配置
    parser.add_argument(
        "--model-size",
        type=str,
        default=DEFAULT_MODEL_SIZE,
        choices=list(MODEL_SIZES.keys()),
        help=f"模型大小 (默认: {DEFAULT_MODEL_SIZE})",
    )

    # 硬件配置
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=list(HARDWARE_PROFILES.keys()),
        help="硬件预设 (默认: 自动检测)",
    )

    # 训练参数覆盖
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch", type=int, default=None, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=None, help="输入图像大小")
    parser.add_argument("--lr0", type=float, default=None, help="初始学习率")
    parser.add_argument("--patience", type=int, default=None, help="早停轮数")
    parser.add_argument("--workers", type=int, default=None, help="数据加载线程数")

    # 输出配置
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="训练输出目录 (默认: YOLO 自动设置为 runs/pose)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="训练运行名称 (默认: 根据 model-type 自动设置)",
    )

    # 控制选项
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次中断处继续训练",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="冒烟测试模式（epochs=5, 仅验证训练能正常启动）",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过数据集完整性验证",
    )
    parser.add_argument(
        "--no-fix-yaml",
        action="store_true",
        help="不自动修正 data.yaml 中的 flip_idx",
    )

    # 数据拆分选项
    parser.add_argument(
        "--split",
        action="store_true",
        help="训练前先执行数据集随机拆分",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.77,
        help="训练集比例 (默认: 0.77，需配合 --split 使用)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.17,
        help="验证集比例 (默认: 0.17，需配合 --split 使用)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.06,
        help="测试集比例 (默认: 0.06，需配合 --split 使用)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42，需配合 --split 使用)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    data_yaml_path = dataset_dir / "data.yaml"

    # 根据 model-type 设置默认 run name
    if args.name is None:
        args.name = "badminton_net" if args.model_type == "net" else "badminton_court"

    # 加载模型类型对应的关键点配置
    from training.prepare_data import _load_model_config
    import training.prepare_data as _pd
    _pd.NUM_KEYPOINTS, _pd.DATASET_FLIP_IDX, _pd.EXPECTED_FIELDS_PER_LINE = (
        _load_model_config(args.model_type)
    )
    print(f"[模型类型] {args.model_type} ({_pd.NUM_KEYPOINTS}个关键点)")

    if not dataset_dir.exists():
        print(f"[错误] 数据集目录不存在: {dataset_dir}")
        sys.exit(1)
    if not data_yaml_path.exists():
        print(f"[错误] data.yaml 不存在: {data_yaml_path}")
        sys.exit(1)

    # ── Step 0: 数据集拆分（可选）──
    if args.split:
        print("=" * 50)
        print("Step 0: 数据集随机拆分")
        print("=" * 50)
        split_dataset(
            dataset_dir=dataset_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        # 拆分后需要重新读取 data.yaml（路径已更新为 txt 文件）
        print()

    # ── Step 1: 修正 data.yaml ──
    if not args.no_fix_yaml:
        print("=" * 50)
        print("Step 1: 检查并修正 data.yaml")
        print("=" * 50)
        fix_data_yaml(data_yaml_path, in_place=True)
        print()

    # ── Step 2: 验证数据集 ──
    if not args.skip_validation:
        print("=" * 50)
        print("Step 2: 验证数据集完整性")
        print("=" * 50)
        report = validate_dataset(dataset_dir)
        ok = print_report(report)
        if not ok:
            print("\n[错误] 数据集验证未通过，请修复后重试")
            print("提示: 使用 --skip-validation 跳过验证（不推荐）")
            sys.exit(1)
        print()

    # ── Step 3: 构建训练参数 ──
    print("=" * 50)
    print("Step 3: 构建训练配置")
    print("=" * 50)

    # 收集命令行覆盖参数
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.lr0 is not None:
        overrides["lr0"] = args.lr0
    if args.patience is not None:
        overrides["patience"] = args.patience
    if args.workers is not None:
        overrides["workers"] = args.workers

    # 冒烟测试模式：最少数据 + 最少轮数，仅验证流程可走通
    if args.smoke_test:
        overrides["epochs"] = overrides.get("epochs", 2)
        overrides["patience"] = overrides.get("patience", 2)
        overrides["fraction"] = overrides.get("fraction", 0.05)
        overrides["batch"] = overrides.get("batch", 2)
        smoke_fraction = overrides["fraction"]
        print(
            f"[冒烟测试] epochs={overrides['epochs']}, "
            f"fraction={smoke_fraction}, batch={overrides['batch']}"
        )

    model_weight, train_kwargs = build_train_args(
        data_yaml=data_yaml_path.resolve(),
        model_size=args.model_size,
        profile=args.profile,
        project=args.project,
        name=args.name,
        resume=args.resume,
        **overrides,
    )

    detected = detect_device()
    actual_profile = args.profile or detected
    print(f"  模型: {args.model_size} ({model_weight})")
    print(f"  设备: {train_kwargs['device']} (预设: {actual_profile})")
    print(f"  批次大小: {train_kwargs['batch']}")
    print(f"  轮数: {train_kwargs['epochs']}")
    print(f"  图像尺寸: {train_kwargs['imgsz']}")
    print(f"  学习率: {train_kwargs['lr0']}")
    print(f"  早停: {train_kwargs['patience']} 轮")
    project_display = args.project or "runs/pose"
    print(f"  输出: {project_display}/{args.name}")
    print()

    # ── Step 4: 启动训练 ──
    print("=" * 50)
    print("Step 4: 启动训练")
    print("=" * 50)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics 库")
        print("请运行: pip install ultralytics>=8.1.0")
        sys.exit(1)

    model = YOLO(model_weight)
    results = model.train(**train_kwargs)

    # ── 训练完成 ──
    print()
    print("=" * 50)
    print("训练完成")
    print("=" * 50)

    # 从训练结果中获取实际保存路径
    save_dir = getattr(results, "save_dir", None) or Path(project_display) / args.name
    save_dir = Path(save_dir)
    best_model = save_dir / "weights" / "best.pt"
    if best_model.exists():
        print(f"  最佳模型: {best_model}")
    last_model = save_dir / "weights" / "last.pt"
    if last_model.exists():
        print(f"  最新模型: {last_model}")
    print(f"  完整结果: {save_dir}/")

    return results


if __name__ == "__main__":
    main()
