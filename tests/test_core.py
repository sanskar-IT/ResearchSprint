import json
import unittest

from researchsprint.core import (
    DEFAULT_FALLBACK_MODEL,
    ResearchAgent,
    ResearchSprintOrchestrator,
    format_team_prompt,
    internet_search,
    select_target_model,
)


class FakeModelClient:
    def __init__(self, response="ok", should_fail=False):
        self.response = response
        self.should_fail = should_fail
        self.calls = []

    def generate(self, prompt, *, generation_config=None, tools=None, system_instruction=None):
        self.calls.append(
            {
                "prompt": prompt,
                "generation_config": generation_config,
                "tools": tools,
                "system_instruction": system_instruction,
            }
        )
        if self.should_fail:
            raise RuntimeError("boom")
        return self.response


class FakeDDGS:
    def __init__(self):
        self.text_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query, max_results=5):
        self.text_calls.append((query, max_results))
        return [{"title": "result", "query": query, "max": max_results}]


class TestCoreHelpers(unittest.TestCase):
    def test_format_team_prompt_contains_context_and_task(self):
        prompt = format_team_prompt("task", "ctx")
        self.assertIn("[CONTEXT FROM TEAM]:", prompt)
        self.assertIn("ctx", prompt)
        self.assertIn("[YOUR CURRENT TASK]:", prompt)
        self.assertIn("task", prompt)

    def test_select_target_model_prefers_exact(self):
        selected = select_target_model(["models/gemini-2.5-flash", "models/gemini-2.0-flash"])
        self.assertEqual(selected, "gemini-2.5-flash")

    def test_select_target_model_chooses_flash_2_5_variant(self):
        selected = select_target_model(["models/gemini-2.5-flash-lite-preview-09-2025"])
        self.assertEqual(selected, "gemini-2.5-flash-lite-preview-09-2025")

    def test_select_target_model_falls_back(self):
        selected = select_target_model(["models/gemini-2.0-pro"])
        self.assertEqual(selected, DEFAULT_FALLBACK_MODEL)

    def test_internet_search_serializes_results(self):
        out = internet_search("adk", max_results=3, ddgs_client_factory=FakeDDGS)
        parsed = json.loads(out)
        self.assertEqual(parsed[0]["query"], "adk")
        self.assertEqual(parsed[0]["max"], 3)


class TestResearchAgent(unittest.TestCase):
    def test_process_sends_generation_with_role_and_tools(self):
        client = FakeModelClient(response="done")
        tools = [lambda _: "x"]
        agent = ResearchAgent(
            name="Scout",
            role="role text",
            model_client=client,
            generation_config={"temperature": 0.7},
            tools=tools,
        )

        output = agent.process("collect data", "existing context")

        self.assertEqual(output, "done")
        self.assertEqual(client.calls[0]["system_instruction"], "role text")
        self.assertEqual(client.calls[0]["tools"], tools)
        self.assertIn("collect data", client.calls[0]["prompt"])
        self.assertIn("existing context", client.calls[0]["prompt"])

    def test_process_returns_error_message_on_failure(self):
        client = FakeModelClient(should_fail=True)
        agent = ResearchAgent(name="Critic", role="r", model_client=client)
        output = agent.process("x")
        self.assertIn("Error in Critic", output)


class TestOrchestrator(unittest.TestCase):
    @staticmethod
    def _build_agent(name, response):
        return ResearchAgent(name=name, role=f"{name} role", model_client=FakeModelClient(response=response))

    def test_log_step_accumulates_history_and_context(self):
        orch = ResearchSprintOrchestrator(agents={})
        orch.log_step("A", "B")
        self.assertEqual(len(orch.history), 1)
        self.assertIn("A Output", orch.full_context)
        self.assertIn("B", orch.full_context)

    def test_run_sprint_executes_all_stages_and_returns_scribe_output(self):
        agents = {
            "planner": self._build_agent("planner", "plan1"),
            "scout": self._build_agent("scout", "research1"),
            "ideation": self._build_agent("ideation", "ideas1"),
            "critique": self._build_agent("critique", "critique1"),
            "integration": self._build_agent("integration", "refined1"),
            "scribe": self._build_agent("scribe", "final1"),
            "memory": self._build_agent("memory", "summary1"),
        }
        orch = ResearchSprintOrchestrator(agents=agents)

        result = orch.run_sprint("Build X")

        self.assertEqual(result, "final1")
        self.assertEqual(len(orch.history), 7)
        self.assertIn("Sprint Planner Output", orch.full_context)
        self.assertIn("Scribe Agent Output", orch.full_context)
        self.assertIn("Memory Agent Output", orch.full_context)

        logged_agents = [entry.split(" Output ---", 1)[0].replace("--- ", "") for entry in orch.history]
        self.assertEqual(
            logged_agents,
            [
                "Sprint Planner",
                "Knowledge Scout",
                "Ideation Generator",
                "Critique Agent",
                "Integration Agent",
                "Scribe Agent",
                "Memory Agent",
            ],
        )

        scout_prompt = agents["scout"].model_client.calls[0]["prompt"]
        self.assertIn("plan1", scout_prompt)
        self.assertIn("Sprint Planner Output", scout_prompt)

    def test_run_sprint_requires_all_agents(self):
        orch = ResearchSprintOrchestrator(agents={"planner": self._build_agent("planner", "x")})
        with self.assertRaises(ValueError):
            orch.run_sprint("objective")


if __name__ == "__main__":
    unittest.main()
