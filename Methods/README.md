# Methods

Core memory implementations for **Why Do Self-Improving Agents Fail?**
These are adaptations within a shared state interface, not complete upstream
agent runtimes. They preserve the original project's default memory algorithms
and prompts. Benchmark-specific mitigations and experimental ablations are
excluded from this package.

| Method | Module / class | Memory and learning |
| --- | --- | --- |
| ACE | `ace.py` / `PlaybookState` | Curator adds numbered rules to a sectioned playbook; retrieval returns the full playbook. |
| Agent Workflow Memory | `awm.py` / `AWMState` | Induces reusable workflows from batches of successful trajectories; retrieves similar workflows. |
| ReasoningBank | `reasoning_bank.py` / `ReasoningBankState` | Extracts reasoning items from successes and failures; retrieves items belonging to similar historical task queries. |
| Hermes | `hermes.py` / `HermesState` | Maintains `SKILL.md` documents and support files in memory; exposes an index and loads relevant skill bodies. |

## Installation

Python 3.11 or newer. From the repository root:

```bash
python -m pip install -r Methods/requirements.txt
```

Runtime dependencies are NumPy and PyYAML. Supply model and embedding clients
that implement the structural contracts in `interfaces.py`:

- `llm.generate(prompt, max_tokens=..., system=None) -> str`
- `embedder.embed(texts) -> numpy.ndarray`, with L2-normalized rows
- `embedder.top_k_indices(query, candidates, k) -> list[int]`

ReasoningBank requires an embedder. AWM and Hermes support keyword retrieval
when it is absent; ACE's full-playbook retrieval does not use embeddings.

## Integration

```python
from Methods import ReasoningBankState

# llm and embedder are supplied by the calling application.
memory = ReasoningBankState(llm=llm, embedder=embedder)
context = memory.retrieve("The next task description", k=1)

# Solve and evaluate the task using context, then supply the observed result.
memory.update({
    "task_description": "The completed task description",
    "trajectory": "The agent's actual interaction trace",
    "outcome": "success",  # or "failure"
    "reflection_text": "An explanation of what worked or failed",
    "reflection_json": {},  # optional structured reflection
    "metadata": {"task_id": "example-task"},
})
snapshot = memory.serialize()
```

All four classes expose `retrieve`, `update`, `serialize`, `load_serialized`,
and `stats`. Evaluation and reflection are caller responsibilities; this
package contains only the memory methods. State budgets use an approximate
four characters per token.

## Method details

- **ACE:** preserves the project's ADD-only curator and oldest-bullet pruning.
  The class name and snapshot type remain `PlaybookState` and `playbook`.
  Append-only ablations and threshold-based selective retrieval are excluded.
- **AWM:** learns only from successful trajectories, with an induction window
  of three by default. Snapshots restore workflows and counters; as in the
  original implementation, pending induction history resets on load.
- **ReasoningBank:** uses the caller's evaluated outcome instead of a separate
  LLM judge, and can supplement extraction with the supplied reflection.
  Default `k=1` selects one historical task entry, which may contain up to
  three memory items. Oldest entries are pruned when over budget.
- **Hermes:** supports create, edit, patch, delete, write_file, and remove_file
  actions on an in-memory skill library. Support files are serialized text;
  they are not written to disk or executed. The existing default category
  `why-fail` is retained for generated-skill compatibility.

Constructors take method settings; snapshots are not a substitute for the
experiment configuration. Reuse the same settings when loading a snapshot.
Mitigation-enabled snapshots are outside this package's supported scope.

## Offline tests

```bash
python -m pip install pytest
python -m pytest -q Methods/tests
```

Tests use fake clients and make no model API calls.

## Implementation references

- ACE: `ace-agent/ace-appworld`, reflected in the original project's curator prompt.
- AWM: <https://github.com/zorazrw/agent-workflow-memory>
- ReasoningBank: <https://github.com/google-research/reasoning-bank>
- Hermes: <https://github.com/NousResearch/hermes-agent>
