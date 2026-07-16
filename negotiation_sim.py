"""
Demand-Response Multi-Agent Negotiation Simulator
===================================================

Simulates a demand-response event on the electricity grid using four
Mistral-powered agents that negotiate with each other, plus a governance
layer that logs every reasoning step for auditability.

Scenario: a winter demand peak forces the grid operator (DSO) to request
a load curtailment. An industrial consumer aggregator and an energy
supplier negotiate the terms (MW curtailed, duration, compensation).
A regulator/auditor agent checks the final agreement against compliance
rules and either validates or flags it.

Architecture note (why 4 agents, not a single LLM call):
Each agent has a distinct role, incentive, and information set — this
mirrors how real DR markets work, and it means each negotiation step
produces an independently inspectable reasoning trace. That trace is
the audit trail: every decision can be attributed to a specific agent,
with the reasoning that produced it, which is the operational answer to
"how do you make an agentic system auditable for a regulator".

Usage:
    export MISTRAL_API_KEY="your-key-here"
    python negotiation_sim.py                  # live run against Mistral API
    python negotiation_sim.py --mock            # offline dry-run, no API calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


MODEL = "mistral-large-latest"


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    step: int
    timestamp: str
    agent: str
    role: str
    action: str
    reasoning: str
    output: dict
    compliance_flag: Optional[str] = None


@dataclass
class AuditLog:
    scenario: str
    entries: list = field(default_factory=list)

    def record(self, step, agent, role, action, reasoning, output, compliance_flag=None):
        entry = AuditEntry(
            step=step,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=agent,
            role=role,
            action=action,
            reasoning=reasoning,
            output=output,
            compliance_flag=compliance_flag,
        )
        self.entries.append(entry)
        return entry

    def to_json(self, path):
        data = {"scenario": self.scenario, "entries": [asdict(e) for e in self.entries]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------

class Agent:
    """
    Thin wrapper around a Mistral chat call. Each agent has a fixed system
    prompt (its role, incentives, and constraints) and returns structured
    JSON so the negotiation loop and the audit log can consume it directly.
    """

    def __init__(self, name: str, role: str, system_prompt: str, client=None, mock=False):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.mock = mock

    def act(self, context: str, response_schema_hint: str) -> dict:
        if self.mock:
            return self._mock_response(context)

        response = self.client.chat.complete(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        f"Respond ONLY with a JSON object matching this shape: "
                        f"{response_schema_hint}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def _mock_response(self, context: str) -> dict:
        """Deterministic offline stand-in, used for --mock dry runs and demos."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Scenario-specific agents
# ---------------------------------------------------------------------------

class DSOAgent(Agent):
    def __init__(self, client=None, mock=False):
        super().__init__(
            name="DSO",
            role="Grid Operator (Distribution System Operator)",
            system_prompt=(
                "You are the Distribution System Operator. You monitor grid load and "
                "issue demand-response requests when peak load threatens grid stability. "
                "You must specify the MW to curtail, the time window, and the urgency "
                "level. You are risk-averse: you request a safety margin above the "
                "strict technical minimum."
            ),
            client=client,
            mock=mock,
        )

    def _mock_response(self, context):
        return {
            "mw_requested": 18,
            "window": "17:00-19:00",
            "urgency": "high",
            "reasoning": "Forecast winter evening peak exceeds substation capacity by 14MW; "
                         "requesting 18MW to keep a 4MW safety margin.",
        }


class SupplierAgent(Agent):
    def __init__(self, client=None, mock=False):
        super().__init__(
            name="Supplier",
            role="Energy Supplier",
            system_prompt=(
                "You are the Energy Supplier. You receive a curtailment request from the "
                "DSO and must propose compensation terms to the industrial aggregator "
                "that are commercially viable for you while meeting the DSO's request. "
                "You try to minimize compensation cost while still making the offer "
                "attractive enough to be accepted."
            ),
            client=client,
            mock=mock,
        )

    def _mock_response(self, context):
        return {
            "mw_offered": 15,
            "compensation_eur_per_mwh": 180,
            "reasoning": "Offering 15MW at 180 EUR/MWh, below the DSO's full 18MW ask, "
                         "as an opening position to test aggregator's flexibility.",
        }


class AggregatorAgent(Agent):
    def __init__(self, client=None, mock=False):
        super().__init__(
            name="Aggregator",
            role="Industrial Consumer Aggregator",
            system_prompt=(
                "You represent a pool of industrial consumers who can curtail load. "
                "You evaluate the Supplier's offer against your consumers' operational "
                "constraints and push back if compensation is too low or the MW ask "
                "exceeds what your pool can safely deliver without disrupting production."
            ),
            client=client,
            mock=mock,
        )
        self._mock_call_count = 0

    def _mock_response(self, context):
        self._mock_call_count += 1
        if self._mock_call_count == 1:
            return {
                "counter_mw": 15,
                "counter_compensation_eur_per_mwh": 230,
                "accepted": False,
                "reasoning": "15MW is deliverable without halting production lines, but "
                             "180 EUR/MWh undervalues the disruption cost; countering at 230.",
            }
        return {
            "counter_mw": 15,
            "counter_compensation_eur_per_mwh": 230,
            "accepted": True,
            "reasoning": "Supplier matched the 230 EUR/MWh counter-offer; terms now "
                         "cover the disruption cost, accepting to lock in the agreement "
                         "before the DSO's urgency deadline.",
        }


class RegulatorAgent(Agent):
    def __init__(self, client=None, mock=False):
        super().__init__(
            name="Regulator",
            role="Compliance Auditor",
            system_prompt=(
                "You are the compliance auditor for demand-response transactions. Given "
                "the final agreed terms, you check them against three rules: "
                "(1) curtailment must not exceed 20% of the aggregator's declared "
                "flexible capacity in a single event, "
                "(2) compensation must be within the regulated price corridor of "
                "100-250 EUR/MWh, "
                "(3) the event duration must not exceed 4 hours without a renewal step. "
                "You either VALIDATE the agreement or FLAG it with the specific rule "
                "violated."
            ),
            client=client,
            mock=mock,
        )

    def _mock_response(self, context):
        return {
            "status": "VALIDATED",
            "reasoning": "15MW is within the aggregator's declared 100MW flexible pool "
                         "(15%), 230 EUR/MWh sits inside the 100-250 corridor, and the "
                         "2-hour window is under the 4-hour renewal threshold.",
        }


# ---------------------------------------------------------------------------
# Negotiation orchestration
# ---------------------------------------------------------------------------

def run_negotiation(mock: bool = False, max_rounds: int = 3) -> AuditLog:
    client = None
    if not mock:
        try:
            from mistralai import Mistral
        except ImportError:
            print("mistralai SDK not usable in this environment; run with --mock, "
                  "or install/verify the SDK locally.", file=sys.stderr)
            sys.exit(1)
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            print("Set MISTRAL_API_KEY to run live. Use --mock for an offline dry run.",
                  file=sys.stderr)
            sys.exit(1)
        client = Mistral(api_key=api_key)

    dso = DSOAgent(client, mock)
    supplier = SupplierAgent(client, mock)
    aggregator = AggregatorAgent(client, mock)
    regulator = RegulatorAgent(client, mock)

    log = AuditLog(scenario="Winter evening peak — substation capacity exceeded by 14MW")
    step = 1

    # Step 1: DSO issues the curtailment request
    dso_out = dso.act(
        context="A winter evening peak is forecast to exceed substation capacity. "
                "Issue a demand-response request.",
        response_schema_hint='{"mw_requested": number, "window": string, '
                              '"urgency": string, "reasoning": string}',
    )
    log.record(step, dso.name, dso.role, "issue_curtailment_request",
               dso_out.get("reasoning", ""), dso_out)
    step += 1

    # Step 2: Supplier proposes terms to meet the DSO request
    supplier_out = supplier.act(
        context=f"The DSO has requested: {json.dumps(dso_out)}. Propose compensation terms.",
        response_schema_hint='{"mw_offered": number, "compensation_eur_per_mwh": number, '
                              '"reasoning": string}',
    )
    log.record(step, supplier.name, supplier.role, "propose_terms",
               supplier_out.get("reasoning", ""), supplier_out)
    step += 1

    # Steps 3..N: Aggregator negotiates until acceptance or max_rounds reached
    accepted = False
    current_offer = supplier_out
    for round_num in range(max_rounds):
        agg_out = aggregator.act(
            context=f"The Supplier has offered: {json.dumps(current_offer)}. "
                    f"Evaluate and either accept or counter.",
            response_schema_hint='{"counter_mw": number, '
                                  '"counter_compensation_eur_per_mwh": number, '
                                  '"accepted": boolean, "reasoning": string}',
        )
        log.record(step, aggregator.name, aggregator.role,
                   "evaluate_and_counter" if not agg_out.get("accepted") else "accept_terms",
                   agg_out.get("reasoning", ""), agg_out)
        step += 1

        if agg_out.get("accepted"):
            accepted = True
            final_terms = current_offer
            break

        # Supplier responds to the counter (mock keeps this simple: accept the counter
        # on round 2 to keep the demo transcript short and legible)
        current_offer = {
            "mw_offered": agg_out.get("counter_mw"),
            "compensation_eur_per_mwh": agg_out.get("counter_compensation_eur_per_mwh"),
        }
        if round_num >= 1:
            accepted = True
            final_terms = current_offer
            log.record(step, supplier.name, supplier.role, "accept_counter",
                       "Accepting counter-offer to close the negotiation within the "
                       "operating window before the DSO's urgency deadline.",
                       current_offer)
            step += 1
            break

    if not accepted:
        final_terms = current_offer

    # Final step: Regulator validates
    reg_out = regulator.act(
        context=f"Final agreed terms: {json.dumps(final_terms)}. Window: "
                f"{dso_out.get('window')}. Aggregator declared flexible pool: 100MW. "
                f"Check compliance.",
        response_schema_hint='{"status": string, "reasoning": string}',
    )
    log.record(step, regulator.name, regulator.role, "compliance_check",
               reg_out.get("reasoning", ""), reg_out,
               compliance_flag=reg_out.get("status"))

    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                         help="Run offline with deterministic mock responses, no API calls.")
    parser.add_argument("--out", default="negotiation_log.json",
                         help="Path to write the audit log JSON.")
    args = parser.parse_args()

    log = run_negotiation(mock=args.mock)
    data = log.to_json(args.out)
    print(f"Negotiation complete. {len(log.entries)} steps logged to {args.out}")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
