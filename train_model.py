import argparse
import sys

from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Train TeaGuard YOLOv8 model')
    parser.add_argument(
        "--data",
        default="Tea Leaves Dataset",
        help = 'Path to "Tea Leaves Dataset" folder containing data.yaml'
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help = "Override epoch count from config.yaml"
    )
    args = parser.parse_args()
    if args.epochs:
        import yaml
        cfg_path = Path("config.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        cfg["yolo"]["epochs"] = args.epochs
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f,default_flow_style=False)
        print(f"Epochs set to {args.epochs}")

    from core.trainer import train_model, evaluate_model, validate_dataset
    from tqdm import tqdm

    print(f"\n{'=' * 55}")
    print("  TeaGuard AI — YOLOv8 Detection Training")
    print(f"  Dataset : {Path(args.data).resolve()}")
    print(f"  Classes : T1 / T2 / T3 / T4  (4 classes)")
    print(f"  Format  : Roboflow detection (images + labels)")
    print(f"{'=' * 55}\n")

    print("Validating dataset ...")
    if not validate_dataset(args.data):
        print("Dataset validation failed. Check folder structure.")
        sys.exit(1)

    pbar = None

    def epoch_cb(epoch, total, metrics):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Training", unit="epoch")
        pbar.update(1)
        metric_str = "  ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
            if isinstance(v, (int, float))
        )
        if metric_str:
            pbar.set_postfix_str(metric_str)

    weights = train_model(args.data, progress_callback=epoch_cb)
    if pbar:
       pbar.close()

    print(f"\n✓ Best weights: {weights}")
    print("\nEvaluating on test split ...")
    try:
        metrics = evaluate_model(weights)
        print(f"\nTest Results:")
        print(f"  mAP@50     : {metrics['mAP50'] * 100:.2f}%")
        print(f"  mAP@50-95  : {metrics['mAP50-95'] * 100:.2f}%")
        print(f"  Precision  : {metrics['precision'] * 100:.2f}%")
        print(f"  Recall     : {metrics['recall'] * 100:.2f}%")
    except Exception as e:
        print(f"  Evaluation skipped: {e}")

    print(f"\nLaunch app:  python main.py")

if __name__ == "__main__":
    main()
