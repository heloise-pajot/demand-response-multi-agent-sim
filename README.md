# Demand-response multi-agent negotiation simulator

A working prototype exploring how multi-agent LLM systems can make demand-response
negotiations on the electricity grid auditable — not just functional.

## Why this exists

Regulated infrastructure (energy, in this case) can't deploy an agentic system that
makes decisions no one can explain after the fact. This prototype simulates a
demand-response event — a grid operator asking industrial consumers to curtail load
during a peak — through four Mistral-powered agents, each with a distinct role and
incentive:

- **DSO (grid operator)** — issues the curtailment request
- **Supplier** — proposes compensation terms
- **Aggregator** — represents industrial consumers, negotiates on their behalf
- **Regulator** — checks the final agreement against compliance rules and validates or flags it

Every step is logged with the agent's full reasoning, producing an audit trail that
attributes each decision to a specific agent and a specific rationale. That's the
operational answer to "how do you make an agentic system auditable for a regulator" —
the architecture *is* the governance mechanism, not a layer bolted on afterward.

This extends the multi-agent architecture explored in two papers co-authored at
ECAI/EUMAS 2025 on LLM-based agents applied to demand-response simulation, moving
from research finding to a runnable prototype.

## Running it

```bash
pip install mistralai
export MISTRAL_API_KEY="your-key-here"
python negotiation_sim.py
```

Or run offline with deterministic mock responses, no API key needed:

```bash
python negotiation_sim.py --mock
```

Both write a full audit log to `negotiation_log.json` — see the file in this repo
for a sample run.

## Scope

This is a scoped prototype, not a production system: single scenario, three
negotiation rounds max, no persistence layer, no live grid data feed. The point is
the architecture and the audit trail pattern, not production hardening.
