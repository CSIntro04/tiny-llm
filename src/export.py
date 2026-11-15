"""导出"仅权重"的 checkpoint，用于分发给组员（网盘），不用重新训练就能生成和打开界面。

用法：
    python src/export.py --runs runs/s10m runs/s50m runs/s100m --out release/

训练时保存的 ckpt_best.pt 里有 AdamW 的两个状态 m、v（各和权重一样大），所以文件约是权重的 3 倍。
生成只需要权重和配置，导出时去掉 optimizer、随机数状态，s100m 从 1.27GB 降到约 0.42GB。
导出的文件只能用来生成、评估，不能续训（没有优化器状态）。

输出：release/<run>/ckpt_best.pt + config.yaml + meta.json + 分词器文件 + log.csv + loss_curve.png，
最后打包成 release/tiny-llm-weights.tar。解压到项目根目录的 runs/ 下即可被 sample.py 和界面使用。
不会修改 runs/ 下已有的任何文件。
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import torch

# 生成和界面用到的键：sample.load_model 读 model/config，app.ckpt_meta 读 iter/best_val
KEEP_KEYS = ("model", "iter", "best_val", "config", "meta")
COPY_FILES = ("config.yaml", "meta.json", "log.csv", "loss_curve.png")


def strip_checkpoint(src: Path, dst: Path) -> None:
    """只保留 KEEP_KEYS，写到 dst。权重保持 fp32，导出前后生成结果逐位相同。"""
    state = torch.load(src, map_location="cpu", weights_only=False)
    slim = {k: state[k] for k in KEEP_KEYS if k in state}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)


def export_run(run_dir: Path, out_dir: Path, ckpt: str = "ckpt_best.pt") -> Path:
    """导出一个 run，返回导出目录。分词器文件一起复制，导出目录可以单独使用。"""
    dst = out_dir / run_dir.name
    strip_checkpoint(run_dir / ckpt, dst / ckpt)
    for name in COPY_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, dst / name)
    # meta.json 记着分词器文件名；runs/ 里没有这个文件，要从训练数据目录复制
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    state = torch.load(dst / ckpt, map_location="cpu", weights_only=False)
    data_dir = Path(state["config"]["data"]["dir"])
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[1] / data_dir
    shutil.copy2(data_dir / meta["tokenizer_file"], dst / meta["tokenizer_file"])
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description="导出仅权重的 checkpoint")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="release")
    p.add_argument("--ckpt", default="ckpt_best.pt")
    p.add_argument("--no_tar", action="store_true", help="不打包 tar")
    args = p.parse_args()

    out = Path(args.out)
    dirs = []
    for run in args.runs:
        d = export_run(Path(run), out, args.ckpt)
        size = (d / args.ckpt).stat().st_size / 1e6
        print(f"{run} -> {d}/{args.ckpt}（{size:.0f} MB）")
        dirs.append(d)
    if not args.no_tar:
        tar_path = out / "tiny-llm-weights.tar"
        # 不压缩：权重是随机浮点数，gzip 几乎压不动，反而很慢
        with tarfile.open(tar_path, "w") as tar:
            for d in dirs:
                tar.add(d, arcname=d.name)
        print(f"打包：{tar_path}（{tar_path.stat().st_size / 1e6:.0f} MB）"
              f"；在项目根目录执行 tar xf {tar_path} -C runs/ 解压")


if __name__ == "__main__":
    main()
