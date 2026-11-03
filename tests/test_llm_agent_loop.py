"""The agent loop, checked with a scripted fake client: no network, no cost."""

from types import SimpleNamespace as NS

from wavecheck.hacking import llm_agent


class FakeEnv:
    def __init__(self):
        self.calls = []

    def simulate(self, eps_real, eps_imag, x):
        self.calls.append((eps_real, eps_imag, x))
        return {"q_sca": 1.0}


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self  # so client.messages.create(...) works

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


def tool_use(i, **tool_input):
    block = NS(type="tool_use", id=f"toolu_{i}", name="simulate", input=tool_input)
    return NS(content=[block], stop_reason="tool_use", model_dump=lambda: {"stop": "tool_use"})


def final(text):
    block = NS(type="text", text=text)
    return NS(content=[block], stop_reason="end_turn", model_dump=lambda: {"stop": "end_turn"})


def test_loop_runs_tools_and_returns_the_final_report():
    script = [
        tool_use(1, eps_real=-3.0, eps_imag=0.1, x=0.1),
        tool_use(2, eps_real=-2.1, eps_imag=0.01, x=0.1),
        final("DESIGN: eps_real=-2.1, eps_imag=0.01, x=0.1\nQ_SCA: 1.0\nNOTES: none"),
    ]
    client, env = FakeClient(script), FakeEnv()
    result = llm_agent.run_episode(client, "fake-model", env, max_tool_calls=10)

    assert env.calls == [(-3.0, 0.1, 0.1), (-2.1, 0.01, 0.1)]
    assert result.n_tool_calls == 2 and not result.hit_budget
    assert result.final_text.startswith("DESIGN")
    sent = client.requests[-1]["messages"]
    ids = [
        c["tool_use_id"]
        for m in sent
        if m["role"] == "user" and isinstance(m["content"], list)
        for c in m["content"]
    ]
    assert ids == ["toolu_1", "toolu_2"]


def test_loop_enforces_the_tool_budget():
    script = [tool_use(i, eps_real=-3.0, eps_imag=0.1, x=0.1) for i in range(5)] + [final("done")]
    client, env = FakeClient(script), FakeEnv()
    result = llm_agent.run_episode(client, "fake-model", env, max_tool_calls=2)

    assert len(env.calls) == 2
    assert result.n_tool_calls == 2 and result.hit_budget


def test_monitor_tool_is_offered_only_in_the_monitor_condition():
    for condition, expected in [
        ("neutral", ["simulate"]),
        ("monitor", ["simulate", "check_physical_limits"]),
    ]:
        client = FakeClient([final("done")])
        llm_agent.run_episode(client, "fake-model", FakeEnv(), condition=condition)
        assert [t["name"] for t in client.requests[0]["tools"]] == expected
