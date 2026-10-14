import argparse
import torch.nn as nn

import register_modules
register_modules.register()

from ultralytics import YOLO


def replace_conv_with_patconv(module, pat_cls, conv_cls):
    replaced = 0
    for name, child in module.named_children():
        if (
            isinstance(child, conv_cls)
            and child.conv.kernel_size == (3, 3)
            and child.conv.groups == 1
        ):
            c1 = child.conv.in_channels
            c2 = child.conv.out_channels
            k = child.conv.kernel_size[0]
            s = child.conv.stride[0]
            p = child.conv.padding[0]
            g = child.conv.groups
            d = child.conv.dilation[0]
            new_layer = pat_cls(c1, c2, k, s, p, g, d)
            setattr(module, name, new_layer)
            replaced += 1
        else:
            replaced += replace_conv_with_patconv(child, pat_cls, conv_cls)
    return replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--cfg", type=str, default="yolo26n_patconv_lwga.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="yolo26n_patconv_lwga")
    parser.add_argument("--no-val", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.cfg)

    from custom_modules import PATConv
    from ultralytics.nn.modules import Conv

    n_replaced = replace_conv_with_patconv(model.model, PATConv, Conv)
    print(f"[Info] Replaced {n_replaced} standard 3x3 Conv layers with PATConv.")

    model.model.info()

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=False,
        seed=args.seed,
        project=args.project,
        name=args.name,
        optimizer="MuSGD",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        mosaic=1.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        amp=True,
        cache="ram",
    )

    if not args.no_val:
        results = model.val(data=args.data, split="val")
        print("Val results:", results)


if __name__ == "__main__":
    main()