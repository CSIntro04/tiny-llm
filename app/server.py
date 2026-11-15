"""网页界面的后端（FastAPI）：给 React + shadcn/ui 前端（app/web/）提供 JSON 接口，并托管前端构建产物。

用法：
    python app/server.py                  # 浏览器打开 http://127.0.0.1:7860
    python app/server.py --preload all    # 答辩用：启动时就把三个正式模型都加载好

架构：浏览器（app/web/dist 里的静态页面） ⇄ 本文件的 /api/* 接口 ⇄ src/sample.py。
生成接口用 SSE（Server-Sent Events）流式返回：HTTP 响应不关闭，每生成一个 token 就写一行
"data: {...}\\n\\n"，前端边收边显示，所以文字是一个一个"打"出来的。
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common import (COMPARE_RUNS, DEFAULT_CKPT, FIGURES, GPU_LOCK, RESULT_TABLES, get_model,
                    list_checkpoints, list_runs, model_stats, preload, resolve_seed, run_curve)
from eval_prompts import PROMPTS  # noqa: E402  （common 已把 src/ 加进 sys.path）
from sample import generate_stream  # noqa: E402

DIST = Path(__file__).resolve().parent / "web" / "dist"

app = FastAPI(title="Tiny LLM")


class GenerateRequest(BaseModel):
    """生成参数。范围和界面上的滑块一致；top_k=0 / top_p=1.0 表示不限，seed=-1 表示随机。"""
    ckpt: str
    prompt: str = Field("", max_length=2000)
    max_new_tokens: int = Field(200, ge=1, le=500)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=500)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    seed: int = -1


def _check_ckpt(ckpt: str) -> None:
    """只接受 list_checkpoints() 里列出来的路径，防止用 ../ 读任意文件。"""
    if ckpt not in list_checkpoints():
        raise HTTPException(404, f"找不到 checkpoint：{ckpt}")


# ---------------------------------------------------------------- 模型与生成


@app.get("/api/checkpoints")
def api_checkpoints() -> dict:
    items = list_checkpoints()
    default = DEFAULT_CKPT if DEFAULT_CKPT in items else (items[0] if items else None)
    examples = PROMPTS["story"][:3] + PROMPTS["expository"][:3]
    return {"default": default, "items": items, "examples": examples}


@app.get("/api/model")
def api_model(ckpt: str) -> dict:
    _check_ckpt(ckpt)
    return model_stats(ckpt)


@app.get("/api/compare-models")
def api_compare_models() -> list[dict]:
    """三模型对比页的三列：run 名、checkpoint 路径、参数量（找不到文件的标 missing）。"""
    available = set(list_checkpoints())
    out = []
    for run in COMPARE_RUNS:
        ckpt = f"runs/{run}/ckpt_best.pt"
        ok = ckpt in available
        out.append({"run": run, "ckpt": ckpt, "missing": not ok,
                    "params": get_model(ckpt)[0].num_params(False) if ok else None})
    return out


def _sse(obj: dict) -> str:
    """一条 SSE 消息：'data: <json>' 加一个空行。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _stream(req: GenerateRequest) -> Iterator[str]:
    """逐 token 生成并包装成 SSE。拿着 GPU_LOCK：多个请求排队，不会互相打乱随机种子。"""
    seed = resolve_seed(req.seed)
    with GPU_LOCK:
        model, tok, _ = get_model(req.ckpt)
        t0, text = time.time(), ""
        for delta in generate_stream(model, tok, req.prompt, req.max_new_tokens, req.temperature,
                                     req.top_k or None, req.top_p if req.top_p < 1.0 else None, seed):
            text += delta
            yield _sse({"delta": delta})
        dt = time.time() - t0
    n = len(tok.encode(text))
    yield _sse({"done": True, "seed": seed, "n_tokens": n, "seconds": dt, "tok_s": n / max(dt, 1e-6)})


@app.post("/api/generate")
def api_generate(req: GenerateRequest) -> StreamingResponse:
    _check_ckpt(req.ckpt)
    # 普通 def（不是 async）：FastAPI 会放到线程池里跑，生成时不会卡住其他请求
    return StreamingResponse(_stream(req), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- 训练过程


@app.get("/api/runs")
def api_runs() -> list[str]:
    return list_runs()


@app.get("/api/runs/{run}/curve")
def api_run_curve(run: str) -> FileResponse:
    # 只返回 list_runs() 里那些 run 的 loss_curve.png；不把整个 runs/ 挂出去（里面有 checkpoint）
    if run not in list_runs():
        raise HTTPException(404, f"找不到 run：{run}")
    return FileResponse(run_curve(run))


@app.get("/api/figures")
def api_figures() -> list[str]:
    """docs/figures/ 下的对比图（界面截图除外），前端用 /figures/<文件名> 访问。"""
    return [p.name for p in sorted(FIGURES.glob("*.png")) if not p.name.startswith("app_")]


@app.get("/api/results")
def api_results() -> list[dict]:
    """结果表的 Markdown 原文，前端用 react-markdown 渲染。"""
    return [{"title": title, "markdown": (FIGURES / name).read_text(encoding="utf-8")}
            for name, title in RESULT_TABLES if (FIGURES / name).exists()]


app.mount("/figures", StaticFiles(directory=FIGURES), name="figures")

# 前端页面：放在最后挂载，/api/* 先匹配
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
else:
    @app.get("/", response_class=HTMLResponse)
    def no_frontend() -> str:
        return ("<h3>前端还没构建</h3><p>在 app/web 下运行 <code>npm install && npm run build</code>，"
                "或者先用 Gradio 版：<code>python app/app.py</code></p>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="网页界面后端（FastAPI + React 前端）")
    ap.add_argument("--preload", choices=["none", "default", "all"], default="default")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    preload(args.preload)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
