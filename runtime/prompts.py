"""Reflection and curator prompt templates.

The reflector uses the reference's no-GT variant (appworld_react_reflector_no_gt_prompt.txt)
since our framework evaluates without access to ground truth code.

Memory-specific curation is implemented in Methods.
"""

from __future__ import annotations


_REFLECTION_TASK_MAX_CHARS = 12_000
_REFLECTION_TRAJECTORY_MAX_CHARS = 60_000
_REFLECTION_DETAILS_MAX_CHARS = 12_000
_REFLECTION_PLAYBOOK_MAX_CHARS = 12_000


def _truncate_middle(text: str, max_chars: int, *, label: str) -> str:
    """Keep the beginning and end of long prompt fields within a fixed budget."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 200:
        return text[:max_chars]

    omitted = len(text) - max_chars
    marker = f"\n\n...[{label} truncated: omitted {omitted} chars]...\n\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[:max_chars]

    head = int(keep * 0.6)
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def prepare_reflection_inputs(
    *,
    task: str,
    trajectory: str,
    outcome: str,
    details: str,
    playbook: str = "",
) -> dict[str, str]:
    """Clamp large reflection fields so prompt size stays comfortably bounded."""
    return {
        "task": _truncate_middle(task, _REFLECTION_TASK_MAX_CHARS, label="task"),
        "trajectory": _truncate_middle(
            trajectory, _REFLECTION_TRAJECTORY_MAX_CHARS, label="trajectory"
        ),
        "outcome": outcome,
        "details": _truncate_middle(
            details or "[none]", _REFLECTION_DETAILS_MAX_CHARS, label="details"
        ),
        "playbook": _truncate_middle(
            playbook or "N/A", _REFLECTION_PLAYBOOK_MAX_CHARS, label="playbook"
        ),
    }

# ---------------------------------------------------------------------------
# Reflector prompt (no ground truth) — verbatim from reference, minus GT fields
# ---------------------------------------------------------------------------

REFLECTOR_PROMPT = """\
You are an expert AppWorld coding agent and educator. Your job is to diagnose the \
current trajectory: identify what went wrong (or could be better), grounded in \
execution feedback, API usage, and unit test results.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account to understand the gap between predicted and actual outcome
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help avoid this mistake in the future
- Identify root causes: wrong source of truth, bad filters (timeframe/direction/identity), \
formatting issues, or missing authentication — and how to correct them
- Provide concrete, step-by-step corrections the model should take
- Be specific about what the model should have done differently
- Explicitly curate from the environment feedback the output format/schema of APIs used \
when unclear or mismatched with expectations \
(e.g., `apis.blah.show_contents()` returns a list of content_ids (strings), not content objects)

**Inputs:**

- Ground truth code (reference, known-correct):
<<<GROUND_TRUTH_CODE_START>>>
[Ground Truth not applicable]
<<<GROUND_TRUTH_CODE_END>>>

- Generated code / conversation trace (candidate to critique):
<<<GENERATED_CODE_START>>>
{trajectory}
<<<GENERATED_CODE_END>>>

- Execution result / test report:
<<<TEST_REPORT>>>
Outcome: {outcome}
Details: {details}
<<<TEST_REPORT>>>

- Playbook (strategies used by the agent):
<<<PLAYBOOK_GUIDE>>>
{playbook}
<<<PLAYBOOK_GUIDE>>>

- Prior reflection (from a previous review pass):
<<<PRIOR_REFLECTION>>>
N/A
<<<PRIOR_REFLECTION>>>

**Answer in this exact JSON format (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought — detailed analysis of what happened and why]",
  "error_identification": "[What specifically went wrong?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the agent have done instead?]",
  "key_insight": "[The generalizable strategy or principle to remember to avoid this error]"
}}"""


BROWSECOMP_REFLECTOR_PROMPT = """\
You are an expert BrowseComp-Plus research agent and evaluator. Your job is to \
diagnose the current trajectory for a document-grounded question answering task: \
identify what went wrong (or could be better), grounded in the retrieved \
documents, citations used in the answer, and the judge feedback.

**Instructions:**
- Carefully analyze whether the answer was supported by the retrieved documents
- Check whether citations matched the claims they were attached to
- Identify retrieval failures separately from reasoning or answer synthesis failures
- Pay attention to unsupported claims, missing evidence, overconfident answers, \
and incorrect final answers
- Provide actionable guidance that would improve future BrowseComp-Plus attempts
- Keep the guidance general enough to help on similar document-grounded tasks

**Inputs:**

- User question:
<<<QUESTION>>>
{task}
<<<QUESTION>>>

- Agent trajectory and outputs:
<<<GENERATED_ATTEMPT>>>
{trajectory}
<<<GENERATED_ATTEMPT>>>

- Evaluation result:
<<<JUDGE_REPORT>>>
Outcome: {outcome}
Details: {details}
<<<JUDGE_REPORT>>>

- Current playbook:
<<<PLAYBOOK_GUIDE>>>
{playbook}
<<<PLAYBOOK_GUIDE>>>

**Answer in this exact JSON format (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought — detailed analysis of what happened and why]",
  "error_identification": "[What specifically went wrong?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the agent have done instead?]",
  "key_insight": "[The generalizable strategy or principle to remember to avoid this error]"
}}"""


SWE_BENCH_REFLECTOR_PROMPT = """\
You are an expert SWE-bench software engineering agent and evaluator. Your job is to \
diagnose the current trajectory for a repository bug-fixing task: identify what went \
wrong (or could be better), grounded in the issue description, code-edit trajectory, \
git diff outcome, and evaluation feedback.

**Instructions:**
- Carefully analyze whether the attempted code change actually addressed the issue
- Distinguish repository exploration mistakes from patch-authoring mistakes
- Pay attention to empty diffs, edits to the wrong files, partial fixes, and invalid patch structure
- Identify whether the agent stopped too early, searched the wrong areas, or made an unsupported code change
- Provide actionable debugging and patch-writing guidance that would help on similar code-fix tasks
- Keep the guidance general enough to improve future SWE-bench attempts

**Inputs:**

- User question / issue description:
<<<QUESTION>>>
{task}
<<<QUESTION>>>

- Agent trajectory and outputs:
<<<GENERATED_ATTEMPT>>>
{trajectory}
<<<GENERATED_ATTEMPT>>>

- Evaluation result:
<<<JUDGE_REPORT>>>
Outcome: {outcome}
Details: {details}
<<<JUDGE_REPORT>>>

- Current playbook:
<<<PLAYBOOK_GUIDE>>>
{playbook}
<<<PLAYBOOK_GUIDE>>>

**Answer in this exact JSON format (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought — detailed analysis of what happened and why]",
  "error_identification": "[What specifically went wrong?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the agent have done instead?]",
  "key_insight": "[The generalizable strategy or principle to remember to avoid this error]"
}}"""


TERMINAL_BENCH_REFLECTOR_PROMPT = """\
You are an expert Terminal-Bench agent evaluator. Your job is to diagnose the \
current trajectory for a terminal task executed inside a sandboxed environment: \
identify what went wrong (or could be better), grounded in the task instruction, \
terminal actions, verifier reward, and execution logs.

**Instructions:**
- Carefully analyze whether the agent made progress toward the stated terminal task
- Distinguish environment setup mistakes from command-selection or debugging mistakes
- Pay attention to incomplete workflows, wrong files or directories, and cases where the \
agent failed to validate its work before stopping
- Use the verifier reward and logs to infer whether the final state actually satisfied \
the task requirements
- Provide actionable workflow advice that would help on similar multi-step terminal tasks
- Keep the guidance general enough to improve future terminal-bench attempts

**Inputs:**

- User task / terminal objective:
<<<QUESTION>>>
{task}
<<<QUESTION>>>

- Agent trajectory and outputs:
<<<GENERATED_ATTEMPT>>>
{trajectory}
<<<GENERATED_ATTEMPT>>>

- Evaluation result:
<<<JUDGE_REPORT>>>
Outcome: {outcome}
Details: {details}
<<<JUDGE_REPORT>>>

- Current playbook:
<<<PLAYBOOK_GUIDE>>>
{playbook}
<<<PLAYBOOK_GUIDE>>>

**Answer in this exact JSON format (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought — detailed analysis of what happened and why]",
  "error_identification": "[What specifically went wrong?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the agent have done instead?]",
  "key_insight": "[The generalizable strategy or principle to remember to avoid this error]"
}}"""


def build_reflection_prompt(
    task: str,
    trajectory: str,
    outcome: str,
    details: str,
    playbook: str = "",
    benchmark: str | None = None,
) -> str:
    prepared = prepare_reflection_inputs(
        task=task,
        trajectory=trajectory,
        outcome=outcome,
        details=details,
        playbook=playbook,
    )
    prompt_template = REFLECTOR_PROMPT
    if benchmark == "browsecomp":
        prompt_template = BROWSECOMP_REFLECTOR_PROMPT
    elif benchmark == "swebench":
        prompt_template = SWE_BENCH_REFLECTOR_PROMPT
    elif benchmark == "terminalbench":
        prompt_template = TERMINAL_BENCH_REFLECTOR_PROMPT
    return prompt_template.format(**prepared)


"""Generation prompt templates."""

GENERATION_PROMPT = """\
You are solving a task in a simulated app environment.{env_context_block}

Here is relevant context from your past experience:

{retrieved_context}

--- Task ---
{task_description}

Write imperative Python code (NOT a function definition) that completes the task.
Think step by step, then write clean, correct code.
Wrap your solution in a ```python ... ``` code block."""

GENERATION_PROMPT_NO_CONTEXT = """\
You are solving a task in a simulated app environment.{env_context_block}

--- Task ---
{task_description}

Write imperative Python code (NOT a function definition) that completes the task.
Think step by step, then write clean, correct code.
Wrap your solution in a ```python ... ``` code block."""


def build_generation_prompt(
    task_description: str,
    retrieved_context: str,
    env_context: str = "",
) -> str:
    env_context_block = f"\n\n{env_context.strip()}" if env_context.strip() else ""
    if retrieved_context.strip():
        return GENERATION_PROMPT.format(
            env_context_block=env_context_block,
            retrieved_context=retrieved_context,
            task_description=task_description,
        )
    return GENERATION_PROMPT_NO_CONTEXT.format(
        env_context_block=env_context_block,
        task_description=task_description,
    )
