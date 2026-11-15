"""网页界面后端（app/server.py）的测试：接口字段、流式生成与 generate_text 一致、路径校验。

运行：pytest tests/test_server.py   （需要 runs/s10m/ckpt_best.pt，没有就跳过）
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))

CKPT = "runs/s10m/ckpt_best.pt"
pytestmark = pytest.mark.skipif(not (ROOT / CKPT).exists(), reason=f"没有 {CKPT}")

from fastapi.testclient import TestClient  # noqa: E402

from common import get_model  # noqa: E402
from sample import generate_text  # noqa: E402
from server import app  # noqa: E402

client = TestClient(app)


def _events(resp) -> list[dict]:
    """把 SSE 响应拆成一条条 JSON。"""
    return [json.loads(line[len("data: "):]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_checkpoints_and_model() -> None:
    r = client.get("/api/checkpoints").json()
    assert CKPT in r["items"] and len(r["examples"]) == 6
    m = client.get("/api/model", params={"ckpt": CKPT}).json()
    assert m["n_layer"] == 6 and m["vocab_size"] == 8192 and m["params"] > m["params_non_emb"]


@pytest.mark.parametrize("temperature,top_k,top_p", [(0.8, 50, 1.0), (0.0, 0, 1.0), (1.2, 0, 0.9)])
def test_stream_matches_generate_text(temperature: float, top_k: int, top_p: float) -> None:
    """同一个 seed：流式拼起来的文本 == generate_text 一次性生成的文本。"""
    body = {"ckpt": CKPT, "prompt": "Once upon a time", "max_new_tokens": 60,
            "temperature": temperature, "top_k": top_k, "top_p": top_p, "seed": 123}
    ev = _events(client.post("/api/generate", json=body))
    assert ev[-1]["done"] and ev[-1]["seed"] == 123
    streamed = "".join(e["delta"] for e in ev[:-1])
    model, tok, _ = get_model(CKPT)
    expected = generate_text(model, tok, "Once upon a time", max_new_tokens=60, temperature=temperature,
                             top_k=top_k or None, top_p=top_p if top_p < 1 else None, seed=123)[0]
    assert streamed == expected


def test_rejects_unknown_paths() -> None:
    assert client.get("/api/model", params={"ckpt": "../etc/passwd"}).status_code == 404
    assert client.post("/api/generate", json={"ckpt": "runs/s10m/../../x.pt"}).status_code == 404
    assert client.get("/api/runs/nope/curve").status_code == 404
    assert client.post("/api/generate", json={"ckpt": CKPT, "temperature": 5}).status_code == 422


def test_training_endpoints() -> None:
    runs = client.get("/api/runs").json()
    assert "s10m" in runs
    assert client.get("/api/runs/s10m/curve").headers["content-type"] == "image/png"
    figs = client.get("/api/figures").json()
    assert figs and not any(f.startswith("app_") for f in figs)
    assert client.get(f"/figures/{figs[0]}").status_code == 200
