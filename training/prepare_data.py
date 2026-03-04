"""
数据准备模块。

功能：
1. 修正 data.yaml 中的 flip_idx
2. 验证数据集完整性（图片-标注配对、关键点数量一致性）
3. 输出验证报告

支持两种数据集格式：
- 目录格式：train/images/, valid/images/, test/images/
- txt 文件列表格式：train.txt, val.txt, test.txt（指向 all/images/）
"""

import sys
from pathlib import Path

import yaml


def _load_model_config(model_type):
    """根据模型类型加载对应的关键点配置。

    Args:
        model_type: "court" 或 "net"。

    Returns:
        (num_keypoints, flip_idx, expected_fields) 元组。
    """
    if model_type == "net":
        from training.keypoint_mapping_net import (
            DATASET_FLIP_IDX,
            NUM_KEYPOINTS,
        )
    else:
        from training.keypoint_mapping import (
            DATASET_FLIP_IDX,
            NUM_KEYPOINTS,
        )
    expected_fields = 1 + 4 + NUM_KEYPOINTS * 3
    return NUM_KEYPOINTS, DATASET_FLIP_IDX, expected_fields


# 默认使用球场模型配置（向后兼容）
NUM_KEYPOINTS, DATASET_FLIP_IDX, EXPECTED_FIELDS_PER_LINE = _load_model_config("court")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def fix_data_yaml(data_yaml_path, in_place=True):
    """修正 data.yaml 中的 flip_idx。

    Args:
        data_yaml_path: data.yaml 文件路径。
        in_place: True 直接修改文件，False 仅返回修正后的内容。

    Returns:
        修正后的 yaml 内容（dict）。
    """
    data_yaml_path = Path(data_yaml_path)
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    old_flip = data.get("flip_idx", [])
    data["flip_idx"] = DATASET_FLIP_IDX

    changed = old_flip != DATASET_FLIP_IDX
    if changed:
        print(f"[fix_data_yaml] flip_idx 已更新")
        print(f"  旧值: {old_flip}")
        print(f"  新值: {DATASET_FLIP_IDX}")
    else:
        print(f"[fix_data_yaml] flip_idx 已经是正确的，无需修改")

    if in_place and changed:
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=None, allow_unicode=True)
        print(f"[fix_data_yaml] 已写入 {data_yaml_path}")

    return data


def _resolve_split_images(dataset_dir, split_path_str):
    """解析一个 split 的图片列表。

    支持两种格式：
    - 如果 split_path 是 .txt 文件，从中读取图片路径列表
    - 如果 split_path 是目录，直接扫描目录中的图片

    Args:
        dataset_dir: 数据集根目录。
        split_path_str: data.yaml 中的路径字符串。

    Returns:
        list of Path，图片文件的绝对路径列表。
    """
    dataset_dir = Path(dataset_dir)
    split_path = dataset_dir / split_path_str

    if split_path.suffix == ".txt" and split_path.is_file():
        # txt 文件列表格式
        images = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path = (dataset_dir / line).resolve()
                if img_path.exists():
                    images.append(img_path)
        return images
    elif split_path.is_dir():
        # 目录格式
        return [
            f for f in sorted(split_path.iterdir())
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        return []


def validate_dataset(dataset_dir):
    """验证数据集完整性。

    自动检测数据集格式（txt 文件列表 或 目录），然后验证：
    - 每个 split 的图片-标注配对
    - 标注格式正确性（字段数量 = 71）
    - 所有关键点 visibility 值合法（0/1/2）

    Args:
        dataset_dir: 数据集根目录路径。

    Returns:
        dict，包含验证结果和统计信息。
    """
    dataset_dir = Path(dataset_dir)
    data_yaml_path = dataset_dir / "data.yaml"

    report = {
        "dataset_dir": str(dataset_dir),
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "errors": [],
    }

    # 读取 data.yaml 获取 split 路径
    split_paths = {}
    if data_yaml_path.exists():
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        split_paths = {
            "train": data.get("train", ""),
            "valid": data.get("val", ""),
            "test": data.get("test", ""),
        }
    else:
        # 回退到默认目录结构
        split_paths = {
            "train": "train/images",
            "valid": "valid/images",
            "test": "test/images",
        }

    for split_name, split_path_str in split_paths.items():
        split_report = _validate_split_from_list(
            dataset_dir, split_name, split_path_str, report["errors"]
        )
        report["splits"][split_name] = split_report
        report["total_images"] += split_report["num_images"]
        report["total_labels"] += split_report["num_labels"]

    report["ok"] = len(report["errors"]) == 0
    return report


def _validate_split_from_list(dataset_dir, split_name, split_path_str, errors):
    """验证单个 split。

    Args:
        dataset_dir: 数据集根目录。
        split_name: split 名称。
        split_path_str: data.yaml 中的路径字符串。
        errors: 错误列表（追加）。

    Returns:
        split 报告字典。
    """
    split_report = {
        "split": split_name,
        "num_images": 0,
        "num_labels": 0,
        "missing_labels": [],
        "missing_images": [],
        "format_errors": [],
    }

    if not split_path_str:
        return split_report

    images = _resolve_split_images(dataset_dir, split_path_str)
    if not images:
        errors.append(f"[{split_name}] 未找到图片: {split_path_str}")
        return split_report

    split_report["num_images"] = len(images)

    # 检查每张图片的标注文件
    label_count = 0
    for img_path in images:
        # YOLO 通过将路径中 'images' 替换为 'labels' 来查找标注
        label_path = Path(str(img_path).replace("/images/", "/labels/"))
        label_path = label_path.with_suffix(".txt")

        if label_path.exists():
            label_count += 1
            _validate_label_file(label_path, split_name, split_report, errors)
        else:
            split_report["missing_labels"].append(img_path.stem)
            if len(split_report["missing_labels"]) <= 10:
                errors.append(
                    f"[{split_name}] 图片 {img_path.name} 缺少标注文件"
                )

    split_report["num_labels"] = label_count
    missing_count = len(split_report["missing_labels"])
    if missing_count > 10:
        errors.append(
            f"[{split_name}] 共 {missing_count} 张图片缺少标注文件"
        )
    split_report["missing_labels"] = split_report["missing_labels"][:10]

    return split_report


def _validate_label_file(label_file, split_name, split_report, errors):
    """验证单个标注文件的格式。"""
    with open(label_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line_idx, line in enumerate(lines):
        fields = line.split()
        if len(fields) != EXPECTED_FIELDS_PER_LINE:
            msg = (
                f"[{split_name}] {label_file.name} 第{line_idx+1}行："
                f"字段数 {len(fields)} != 预期 {EXPECTED_FIELDS_PER_LINE}"
            )
            split_report["format_errors"].append(msg)
            errors.append(msg)
            continue

        # 检查 visibility 值
        for kp_idx in range(NUM_KEYPOINTS):
            vis_offset = 5 + kp_idx * 3 + 2  # class(1) + bbox(4) + kp*(x,y,v)
            vis_val = float(fields[vis_offset])
            if vis_val not in (0.0, 1.0, 2.0):
                msg = (
                    f"[{split_name}] {label_file.name} 第{line_idx+1}行："
                    f"关键点{kp_idx} visibility={vis_val}，预期 0/1/2"
                )
                split_report["format_errors"].append(msg)
                errors.append(msg)


def print_report(report):
    """打印验证报告。"""
    print("\n" + "=" * 60)
    print("数据集验证报告")
    print("=" * 60)
    print(f"目录: {report['dataset_dir']}")
    print(f"总图片数: {report['total_images']}")
    print(f"总标注数: {report['total_labels']}")
    print()

    for split_name, sr in report["splits"].items():
        status = "✓" if not sr["missing_labels"] and not sr["format_errors"] else "✗"
        print(f"  [{status}] {split_name}: {sr['num_images']} 图片, {sr['num_labels']} 标注")
        if sr["missing_labels"]:
            print(f"      缺少标注: {len(sr['missing_labels'])} 个")
        if sr["missing_images"]:
            print(f"      缺少图片: {len(sr['missing_images'])} 个")
        if sr["format_errors"]:
            print(f"      格式错误: {len(sr['format_errors'])} 个")

    print()
    if report["ok"]:
        print("结果: ✓ 数据集验证通过")
    else:
        print(f"结果: ✗ 发现 {len(report['errors'])} 个问题")
        for err in report["errors"][:20]:
            print(f"  - {err}")
        if len(report["errors"]) > 20:
            print(f"  ... 还有 {len(report['errors']) - 20} 个问题未显示")

    print("=" * 60)
    return report["ok"]


def main():
    """命令行入口：验证数据集并修正 data.yaml。"""
    import argparse

    parser = argparse.ArgumentParser(description="数据准备与验证工具")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="数据集根目录路径",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["court", "net"],
        default="court",
        help="模型类型：court(球场22点) 或 net(球网4点)，默认 court",
    )
    parser.add_argument(
        "--fix-yaml",
        action="store_true",
        help="修正 data.yaml 中的 flip_idx",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证，不修改任何文件",
    )
    args = parser.parse_args()

    # 根据模型类型重新加载配置
    global NUM_KEYPOINTS, DATASET_FLIP_IDX, EXPECTED_FIELDS_PER_LINE
    NUM_KEYPOINTS, DATASET_FLIP_IDX, EXPECTED_FIELDS_PER_LINE = _load_model_config(
        args.model_type
    )
    print(f"[模型类型] {args.model_type} ({NUM_KEYPOINTS}个关键点, "
          f"每行{EXPECTED_FIELDS_PER_LINE}字段)")

    dataset_dir = Path(args.dataset_dir)
    data_yaml_path = dataset_dir / "data.yaml"

    # 验证数据集
    report = validate_dataset(dataset_dir)
    ok = print_report(report)

    # 修正 data.yaml
    if args.fix_yaml and not args.validate_only:
        if data_yaml_path.exists():
            print()
            fix_data_yaml(data_yaml_path, in_place=True)
        else:
            print(f"\n[警告] 未找到 data.yaml: {data_yaml_path}")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
