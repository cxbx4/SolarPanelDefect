import register_modules
register_modules.register()

from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/runs/train/yolo26n_patconv_lwga/weights/best.pt")
    results = model.val(
        data="configs/data.yaml",
        split="test",
        workers=0,
    )
    print(results)


if __name__ == "__main__":
    main()