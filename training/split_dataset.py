"""
数据集合并与随机拆分模块。

功能：
1. 将 train/valid/test 下所有数据合并到 all/ 目录
2. 根据可配置比例随机拆分，生成 train.txt / val.txt / test.txt 文件列表
3. 更新 data.yaml 指向文本文件列表

YOLO 支持在 data.yaml 中用 .txt 文件列表代替目录路径，
因此只需要 all/ 一个数据目录 + 三个 txt 文件即可定义拆分。

用法示例：
    python -m training.split_dataset \
        --dataset-dir BadmintonCourtDetection.yolov8 \
        --train-ratio 0.77 --valid-ratio 0.17 --test-ratio 0.06

    # 使用不同随机种子
    python -m training.split_dataset \
        --dataset-dir BadmintonCourtDetection.yolov8 --seed 123
"""

import random
import shutil
import sys
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ["train", "valid", "test"]


def merge_splits_to_all(dataset_dir):
    """将 train/valid/test 中的数据合并到 all/ 目录。

    使用复制方式将所有文件移动到 all/images/ 和 all/labels/。

    Args:
        dataset_dir: 数据集根目录。

    Returns:
        合并后的文件数量。
    """
    dataset_dir = Path(dataset_dir)
    all_images_dir = dataset_dir / "all" / "images"
    all_labels_dir = dataset_dir / "all" / "labels"
    all_images_dir.mkdir(parents=True, exist_ok=True)
    all_labels_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in SPLIT_NAMES:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"

        if not images_dir.exists():
            continue

        count = 0
        for img_file in images_dir.iterdir():
            if img_file.is_symlink():
                continue
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dest_img = all_images_dir / img_file.name
            if dest_img.exists():
                continue
            shutil.copy2(img_file, dest_img)

            label_file = labels_dir / (img_file.stem + ".txt")
            if label_file.exists() and not label_file.is_symlink():
                shutil.copy2(label_file, all_labels_dir / label_file.name)

            count += 1

        print(f"  [{split}] 合并了 {count} 个样本")
        total += count

    return total


def collect_pairs(all_dir):
    """收集 all/ 目录中所有 image-label 配对。

    Args:
        all_dir: all/ 目录路径。

    Returns:
        list of (image_filename, label_filename) 元组。
    """
    images_dir = Path(all_dir) / "images"
    labels_dir = Path(all_dir) / "labels"

    pairs = []
    for img_file in sorted(images_dir.iterdir()):
        if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_file = labels_dir / (img_file.stem + ".txt")
        if label_file.exists():
            pairs.append((img_file.name, label_file.name))
        else:
            print(f"  [警告] 图片 {img_file.name} 缺少标注文件，跳过")

    return pairs


def split_by_ratio(pairs, train_ratio, valid_ratio, test_ratio):
    """按比例切分数据。

    Args:
        pairs: (image_filename, label_filename) 列表（已打乱）。
        train_ratio: 训练集比例。
        valid_ratio: 验证集比例。
        test_ratio: 测试集比例。

    Returns:
        (train_pairs, valid_pairs, test_pairs) 三个列表。
    """
    total = len(pairs)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_pairs = pairs[:train_end]
    valid_pairs = pairs[train_end:valid_end]
    test_pairs = pairs[valid_end:]

    return train_pairs, valid_pairs, test_pairs


def write_split_lists(dataset_dir, train_pairs, valid_pairs, test_pairs):
    """生成 train.txt / val.txt / test.txt 文件列表。

    每个文件每行一个图片路径（相对于 dataset_dir）。
    YOLO 自动将路径中的 'images' 替换为 'labels' 来查找对应标注。

    Args:
        dataset_dir: 数据集根目录。
        train_pairs: 训练集 (image_filename, label_filename) 列表。
        valid_pairs: 验证集列表。
        test_pairs: 测试集列表。
    """
    dataset_dir = Path(dataset_dir)
    splits = {
        "train.txt": train_pairs,
        "val.txt": valid_pairs,
        "test.txt": test_pairs,
    }
    for filename, pairs in splits.items():
        filepath = dataset_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for img_name, _ in pairs:
                f.write(f"./all/images/{img_name}\n")


def update_data_yaml(dataset_dir):
    """更新 data.yaml，将 train/val/test 路径改为 txt 文件列表。

    Args:
        dataset_dir: 数据集根目录。
    """
    dataset_dir = Path(dataset_dir)
    data_yaml_path = dataset_dir / "data.yaml"

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["train"] = "train.txt"
    data["val"] = "val.txt"
    data["test"] = "test.txt"

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=None, allow_unicode=True)

    print(f"[data.yaml] 已更新: train→train.txt, val→val.txt, test→test.txt")


def split_dataset(dataset_dir, train_ratio=0.77, valid_ratio=0.17,
                  test_ratio=0.06, seed=42):
    """执行数据集合并与随机拆分的完整流程。

    流程：
    1. 合并：如果 all/ 不存在，将 train/valid/test 数据合并到 all/
    2. 收集 all/ 中所有 image-label 配对
    3. 随机打乱并按比例切分
    4. 生成 train.txt / val.txt / test.txt 文件列表
    5. 更新 data.yaml 指向 txt 文件

    Args:
        dataset_dir: 数据集根目录。
        train_ratio: 训练集比例。
        valid_ratio: 验证集比例。
        test_ratio: 测试集比例。
        seed: 随机种子。

    Returns:
        dict，包含拆分统计信息。
    """
    dataset_dir = Path(dataset_dir)
    all_dir = dataset_dir / "all"

    # 验证比例
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        print(f"[错误] 比例之和 = {ratio_sum:.4f}，应为 1.0")
        sys.exit(1)

    # Step 1: 合并
    if all_dir.exists() and (all_dir / "images").exists():
        existing_count = sum(
            1 for f in (all_dir / "images").iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"[合并] all/ 目录已存在，包含 {existing_count} 张图片，跳过合并")
    else:
        print("[合并] 将 train/valid/test 合并到 all/ ...")
        total = merge_splits_to_all(dataset_dir)
        print(f"[合并] 完成，共 {total} 个样本")

    # Step 2: 收集配对
    pairs = collect_pairs(all_dir)
    print(f"\n[拆分] 共 {len(pairs)} 个有效样本")
    print(f"[拆分] 随机种子: {seed}")
    print(f"[拆分] 比例: train={train_ratio}, valid={valid_ratio}, test={test_ratio}")

    # Step 3: 随机打乱
    random.seed(seed)
    random.shuffle(pairs)

    # Step 4: 按比例切分
    train_pairs, valid_pairs, test_pairs = split_by_ratio(
        pairs, train_ratio, valid_ratio, test_ratio
    )

    # Step 5: 生成 txt 文件列表
    write_split_lists(dataset_dir, train_pairs, valid_pairs, test_pairs)

    # Step 6: 更新 data.yaml
    update_data_yaml(dataset_dir)

    # Step 7: 打印统计
    total = len(pairs)
    print(f"\n{'=' * 50}")
    print("拆分结果")
    print(f"{'=' * 50}")
    splits = [
        ("train", train_pairs),
        ("valid", valid_pairs),
        ("test", test_pairs),
    ]
    for split_name, split_pairs in splits:
        pct = len(split_pairs) / total * 100 if total > 0 else 0
        print(f"  {split_name:6s}: {len(split_pairs):5d} ({pct:5.1f}%)")
    print(f"  {'合计':6s}: {total:5d}")
    print(f"{'=' * 50}")

    return {
        "total": total,
        "train": len(train_pairs),
        "valid": len(valid_pairs),
        "test": len(test_pairs),
        "seed": seed,
    }


def main():
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="数据集合并与随机拆分工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m training.split_dataset --dataset-dir BadmintonCourtDetection.yolov8
  python -m training.split_dataset --dataset-dir BadmintonCourtDetection.yolov8 \\
      --train-ratio 0.8 --valid-ratio 0.15 --test-ratio 0.05 --seed 123
        """,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="数据集根目录路径",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.77,
        help="训练集比例 (默认: 0.77)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.17,
        help="验证集比例 (默认: 0.17)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.06,
        help="测试集比例 (默认: 0.06)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"[错误] 数据集目录不存在: {dataset_dir}")
        sys.exit(1)

    split_dataset(
        dataset_dir=dataset_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
