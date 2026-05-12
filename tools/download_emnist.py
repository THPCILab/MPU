from pathlib import Path

from torchvision import datasets


ROOT = Path(__file__).resolve().parents[1]


def main():
    data_root = ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    datasets.EMNIST(root=str(data_root), split="byclass", train=True, download=True)
    datasets.EMNIST(root=str(data_root), split="byclass", train=False, download=True)
    print(f"EMNIST ByClass is ready under {data_root}")


if __name__ == "__main__":
    main()
