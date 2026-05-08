"""Core logic for ResearchSprint multi-agent orchestration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence

DEFAULT_PREFERRED_MODEL = "gemini-2.5-flash"
DEFAULT_FALLBACK_MODEL = "gemini-1.5-flash"
logger = logging.getLogger(__name__)
PLANNER_TASK_TEMPLATE = "Create a research plan for: {user_objective}"
SCOUT_TASK_TEMPLATE = "Execute the research steps defined in this plan: {plan}"
IDEATION_TASK = "Generate innovative concepts based on the research data."
CRITIQUE_TASK = "Critically review these concepts. Find flaws and suggest fixes."
INTEGRATION_TASK = "Select the best concept and refine it using the critique details."
SCRIBE_TASK = "Write the final comprehensive report/proposal based on the refined strategy."
MEMORY_TASK = "Create a concise executive summary of this entire session for the archives."


class ModelClient(Protocol):
    """Minimal model client protocol used by agents (ADK-compatible abstraction)."""

    def generate(
        self,
        prompt: str,
        *,
        generation_config: Optional[dict] = None,
        tools: Optional[Sequence[Callable]] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate text output for a prompt."""


def format_team_prompt(task_input: str, context: str = "") -> str:
    """Create the standardized team prompt for all agents."""

    return (
        "[CONTEXT FROM TEAM]:\n"
        f"{context}\n\n"
        "[YOUR CURRENT TASK]:\n"
        f"{task_input}\n\n"
        "Perform your role strictly."
    )


def select_target_model(
    available_models: Sequence[str],
    preferred_model: str = DEFAULT_PREFERRED_MODEL,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
) -> str:
    """Select target model from available model names."""

    normalized = [m.replace("models/", "") for m in available_models]
    if preferred_model in normalized:
        return preferred_model

    for candidate in normalized:
        lower = candidate.lower()
        if "flash" in lower and "2.5" in lower:
            return candidate

    return fallback_model


def internet_search(
    query: str,
    *,
    max_results: int = 5,
    ddgs_client_factory: Optional[Callable] = None,
) -> str:
    """Search the internet and return JSON serialized results."""

    if ddgs_client_factory is None:
        try:
            from ddgs import DDGS  # package rename target
        except ImportError:  # backward compatibility
            from duckduckgo_search import DDGS

        ddgs_client_factory = DDGS

    with ddgs_client_factory() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return json.dumps(results)


@dataclass
class ResearchAgent:
    """Single research agent with a model-backed process method."""

    name: str
    role: str
    model_client: ModelClient
    generation_config: Dict = field(default_factory=dict)
    tools: Optional[Sequence[Callable]] = None

    def process(self, task_input: str, context: str = "") -> str:
        prompt = format_team_prompt(task_input=task_input, context=context)
        try:
            return self.model_client.generate(
                prompt,
                generation_config=self.generation_config,
                tools=self.tools,
                system_instruction=self.role,
            )
        except (RuntimeError, TimeoutError, ConnectionError, ValueError) as exc:
            logger.exception("Agent '%s' failed during generation", self.name)
            return f"Error in {self.name}: {exc}"


@dataclass
class ResearchSprintOrchestrator:
    """Sequential orchestrator for the ResearchSprint workflow."""

    agents: Mapping[str, ResearchAgent]
    history: List[str] = field(default_factory=list)
    full_context: str = ""
    latest_archive_summary: str = ""

    def _require_agent(self, key: str) -> ResearchAgent:
        if key not in self.agents:
            raise ValueError(f"Missing required agent: {key}")
        return self.agents[key]

    def log_step(self, agent_name: str, output: str) -> None:
        entry = f"--- {agent_name} Output ---\n{output}\n"
        self.history.append(entry)
        self.full_context += entry

    def run_sprint(self, user_objective: str) -> str:
        planner = self._require_agent("planner")
        scout = self._require_agent("scout")
        ideation = self._require_agent("ideation")
        critique = self._require_agent("critique")
        integration = self._require_agent("integration")
        scribe = self._require_agent("scribe")
        memory = self._require_agent("memory")

        plan = planner.process(
            task_input=PLANNER_TASK_TEMPLATE.format(user_objective=user_objective),
            context=self.full_context,
        )
        self.log_step("Sprint Planner", plan)

        research_data = scout.process(
            task_input=SCOUT_TASK_TEMPLATE.format(plan=plan),
            context=self.full_context,
        )
        self.log_step("Knowledge Scout", research_data)

        concepts = ideation.process(
            task_input=IDEATION_TASK,
            context=self.full_context,
        )
        self.log_step("Ideation Generator", concepts)

        critique_result = critique.process(
            task_input=CRITIQUE_TASK,
            context=self.full_context,
        )
        self.log_step("Critique Agent", critique_result)

        refined_strategy = integration.process(
            task_input=INTEGRATION_TASK,
            context=self.full_context,
        )
        self.log_step("Integration Agent", refined_strategy)

        final_deliverable = scribe.process(
            task_input=SCRIBE_TASK,
            context=self.full_context,
        )
        self.log_step("Scribe Agent", final_deliverable)

        archive_summary = memory.process(
            task_input=MEMORY_TASK,
            context=self.full_context,
        )
        self.latest_archive_summary = archive_summary
        self.log_step("Memory Agent", archive_summary)

        return final_deliverable
