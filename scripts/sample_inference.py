"""
从数据集中随机抽样图片，使用训练好的 YOLO 模型进行推理，保存可视化结果。

用法：
    python -m scripts.sample_inference
"""

import random
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_COUNT = 10
SEED = 42


def sample_images(image_dir: Path, n: int, seed: int) -> list[Path]:
    """从目录中随机抽样 n 张图片。"""
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    random.seed(seed)
    return random.sample(images, min(n, len(images)))


def run_inference(model_path: Path, images: list[Path], save_dir: Path):
    """对图片列表执行推理并保存可视化结果。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(img) for img in images],
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        conf=0.25,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    print(f"  保存 {len(results)} 张结果到 {save_dir}/")


def main():
    # ── 球场模型推理 ──
    print("=" * 50)
    print("球场关键点模型 (court_yolo.pt) 推理")
    print("=" * 50)
    court_images = sample_images(
        PROJECT_ROOT / "yolo" / "datasets" / "court" / "all" / "images",
        SAMPLE_COUNT,
        SEED,
    )
    print(f"  抽样 {len(court_images)} 张图片")
    run_inference(
        model_path=PROJECT_ROOT / "yolo" / "weights" / "court_yolo.pt",
        images=court_images,
        save_dir=PROJECT_ROOT / "yolo" / "sample_results" / "court",
    )

    # ── 球网模型推理 ──
    print()
    print("=" * 50)
    print("球网关键点模型 (net_yolo.pt) 推理")
    print("=" * 50)
    net_images = sample_images(
        PROJECT_ROOT / "yolo" / "datasets" / "net" / "train" / "images",
        SAMPLE_COUNT,
        SEED,
    )
    print(f"  抽样 {len(net_images)} 张图片")
    run_inference(
        model_path=PROJECT_ROOT / "yolo" / "weights" / "net_yolo.pt",
        images=net_images,
        save_dir=PROJECT_ROOT / "yolo" / "sample_results" / "net",
    )

    print()
    print("完成！请检查以下文件夹：")
    print(f"  球场结果: yolo/sample_results/court/")
    print(f"  球网结果: yolo/sample_results/net/")


if __name__ == "__main__":
    main()
