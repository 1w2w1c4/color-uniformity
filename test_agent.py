"""agent 測試：工具直接呼叫判定程式；Claude 以假 client 取代，不需要 API key。"""
import json
from types import SimpleNamespace

import anthropic
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import agent
import main
from sites import SiteRegistry

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    cfg = tmp_path / "sites.yaml"
    cfg.write_text("default: {threshold: 3.0, spot_threshold: null, min_spot_size: 0.5, rows: 5, "
                   "cols: 5, margin: 0.1, reference: null}\nsites:\n  strict: {threshold: 1.0}\n",
                   encoding="utf-8")
    monkeypatch.setattr(main, "registry", SiteRegistry(cfg))
    monkeypatch.setattr(main, "RESULT_DIR", tmp_path / "result")
    img_dir = tmp_path / "images"
    (img_dir / "line1").mkdir(parents=True)
    arr = np.full((200, 300, 3), (200, 22, 34), np.uint8)
    Image.fromarray(arr).save(img_dir / "uniform.png")
    arr[90:110, 140:160] = (250, 240, 240)  # 中央白點
    Image.fromarray(arr).save(img_dir / "line1" / "spot.png")
    (img_dir / "notes.txt").write_text("x")
    (tmp_path / "secret.png").write_bytes(b"x")
    monkeypatch.setattr(agent, "IMAGE_DIR", img_dir.resolve())
    monkeypatch.setattr(agent, "sessions", agent.SessionStore())


# ---------------------------------------------------------------- 工具

def test_list_images_only_images_recursive():
    assert agent.list_images() == ["line1/spot.png", "uniform.png"]
    assert agent.list_images("line1") == ["line1/spot.png"]


def test_check_uniformity_pass_and_fail():
    ok = agent.check_uniformity("uniform.png")
    assert ok["passed"] is True and ok["heatmap_path"].endswith("uniform_heatmap.png")
    bad = agent.check_uniformity("line1/spot.png", site="strict")
    assert bad["passed"] is False and bad["spots"]["count"] >= 1
    assert bad["applied_settings"]["threshold"] == 1.0
    assert "delta_e_grid" not in bad  # 大陣列不送給 Claude


@pytest.mark.parametrize("name,args,msg", [
    ("check_uniformity", {"path": "../secret.png"}, "影像資料夾內"),
    ("check_uniformity", {"path": "/etc/passwd"}, "影像資料夾內"),
    ("check_uniformity", {"path": "missing.png"}, "找不到影像"),
    ("check_uniformity", {"path": "uniform.png", "site": "nope"}, "未知的 site"),
    ("check_uniformity", {"path": "uniform.png", "roi": "0,0,9999,10"}, "roi 超出"),
    ("check_uniformity", {"path": "notes.txt"}, "無法辨識"),
    ("check_pressure", {"path": "uniform.png", "roi": "a,b"}, "roi 格式"),
    ("list_images", {"subdir": ".."}, "影像資料夾內"),
    ("check_uniformity", {"file": "uniform.png"}, "參數錯誤"),
    ("rm_rf", {}, "未知的工具"),
])
def test_tool_errors_returned_to_model(name, args, msg):
    content, is_error = agent.execute_tool(name, args)
    assert is_error and msg in content


def test_check_pressure_runs():
    content, is_error = agent.execute_tool("check_pressure", {"path": "uniform.png", "dpi": 300})
    r = json.loads(content)
    assert not is_error and "passed" in r and r["report_path"].endswith("uniform_pressure.png")


# ---------------------------------------------------------------- agent 循環（假 client）

def block(type, **kw):
    return SimpleNamespace(type=type, **kw, to_dict=lambda: {"type": type, **kw})


def resp(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append({**kw, "messages": json.loads(json.dumps(kw["messages"], default=str))})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def two_tools_then_answer():
    return FakeClient(
        resp("tool_use",
             block("tool_use", id="t1", name="check_uniformity", input={"path": "uniform.png"}),
             block("tool_use", id="t2", name="check_uniformity", input={"path": "line1/spot.png"})),
        resp("end_turn", block("text", text="uniform.png 合格；spot.png 不合格。")),
    )


def test_run_agent_executes_parallel_tools(monkeypatch):
    fake = two_tools_then_answer()
    monkeypatch.setattr(agent, "_client", fake)
    r = agent.run_agent([], "檢查全部")
    assert r.reply == "uniform.png 合格；spot.png 不合格。" and r.stop_reason == "end_turn"
    assert [c["input"]["path"] for c in r.tool_calls] == ["uniform.png", "line1/spot.png"]
    # 兩個結果放在同一則 user 訊息，id 對應
    results = fake.calls[1]["messages"][-1]["content"]
    assert [x["tool_use_id"] for x in results] == ["t1", "t2"]
    assert json.loads(results[0]["content"])["passed"] is True
    assert json.loads(results[1]["content"])["passed"] is False
    assert [m["role"] for m in r.new_messages] == ["user", "assistant", "user", "assistant"]
    assert fake.calls[0]["model"] == agent.MODEL and fake.calls[0]["tools"] == agent.TOOLS


def test_run_agent_step_limit(monkeypatch):
    loop = resp("tool_use", block("tool_use", id="t", name="list_sites", input={}))
    monkeypatch.setattr(agent, "_client", FakeClient(*[loop] * agent.MAX_STEPS))
    assert agent.run_agent([], "x").stop_reason == "max_steps"


# ---------------------------------------------------------------- HTTP 端點

def test_endpoint_keeps_session_history(monkeypatch):
    fake = two_tools_then_answer()
    fake.responses.append(resp("end_turn", block("text", text="剛才 spot.png 有一個白點。")))
    monkeypatch.setattr(agent, "_client", fake)

    r1 = client.post("/v1/agent", json={"message": "檢查全部"}).json()
    assert r1["reply"].startswith("uniform.png") and len(r1["tool_calls"]) == 2
    r2 = client.post("/v1/agent", json={"message": "剛才哪張有問題？", "session_id": r1["session_id"]}).json()
    assert r2["session_id"] == r1["session_id"] and r2["tool_calls"] == []
    assert len(fake.calls[2]["messages"]) == 5  # 第二輪帶著第一輪完整歷史

    assert client.delete(f"/v1/agent/{r1['session_id']}").status_code == 200
    assert client.delete(f"/v1/agent/{r1['session_id']}").status_code == 404


def test_endpoint_api_error_does_not_corrupt_history(monkeypatch):
    req = httpx.Request("POST", "https://api.anthropic.com")
    err = anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
    monkeypatch.setattr(agent, "_client", FakeClient(err))
    r = client.post("/v1/agent", json={"message": "hi", "session_id": "s1"})
    assert r.status_code == 503
    assert agent.sessions.get_or_create("s1")[1].messages == []


def test_endpoint_validation():
    assert client.post("/v1/agent", json={"message": ""}).status_code == 422


# ---------------------------------------------------------------- config.yaml

def test_project_config_loads():
    cfg = agent.load_config(agent.ROOT / "config.yaml")
    assert (cfg["provider"], cfg["model"], cfg["base_url"]) == ("anthropic", "claude-opus-5-5", None)
    assert cfg["api_key_env"] == "ANTHROPIC_API_KEY"
    assert cfg["image_dir"] == (agent.ROOT / "images").resolve()


def test_config_defaults_and_env_override(tmp_path, monkeypatch):
    assert agent.load_config(tmp_path / "missing.yaml")["max_steps"] == 25
    p = tmp_path / "c.yaml"
    p.write_text("agent: {provider: anthropic, model: claude-sonnet-5-5}\n"
                 "providers:\n  anthropic: {effort: high, fallbacks: false, api_key_env: MY_KEY,"
                 " base_url: 'http://gw.local'}\n  openai: {api_key: ignored}\n"
                 "paths: {image_dir: /data/panels}\nsession: {ttl: 60}\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_EFFORT", "low")
    monkeypatch.setenv("MY_KEY", "sk-test")
    cfg = agent.load_config(p)
    assert (cfg["model"], cfg["effort"], cfg["fallbacks"]) == ("claude-sonnet-5-5", "low", False)
    assert (cfg["api_key"], cfg["base_url"]) == ("sk-test", "http://gw.local")
    assert str(cfg["image_dir"]) == "/data/panels" and cfg["session_ttl"] == 60


@pytest.mark.parametrize("text,msg", [
    ("providers: {anthropic: {effort: turbo}}", "effort"),
    ("agent: {provider: openai}", "尚未實作"),
    ("agent: {max_steps: 0}", "max_steps"),
    ("agent: [unclosed", "格式錯誤"),
])
def test_config_invalid(tmp_path, monkeypatch, text, msg):
    monkeypatch.delenv("AGENT_EFFORT", raising=False)
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(agent.AgentConfigError, match=msg):
        agent.load_config(p)
