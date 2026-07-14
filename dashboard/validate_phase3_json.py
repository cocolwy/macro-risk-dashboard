#!/usr/bin/env python3
"""Read-only self-audit for phase3 / metric_exploration JSON vs frontend expectations."""

import json
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"


def audit_phase3(path: Path) -> list[str]:
    d = json.loads(path.read_text())
    errors = []
    exps = d.get("experiments", [])
    names = {e["name"] for e in exps}

    for bad in ("LR Ext+Regime9", "LR Ext+Regime2", "GBDT Ext+Regime2", "LR Regime2", "LR Regime9"):
        if any(bad in n for n in names):
            errors.append(f"Regime2/9 experiment still present: {bad}")

    required_ctx = {"LR Regime-Conditional", "LR Slim+Interact", "GBDT Slim+Interact", "LR Regime-PostCal"}
    missing_ctx = required_ctx - names
    if missing_ctx:
        errors.append(f"Missing regime-as-context experiments: {sorted(missing_ctx)}")

    if "correlation_analysis" not in d:
        errors.append("Missing correlation_analysis block")
    else:
        ca = d["correlation_analysis"]
        if not ca.get("high_corr_pairs"):
            errors.append("correlation_analysis.high_corr_pairs is empty")
        if not ca.get("vif"):
            errors.append("correlation_analysis.vif is empty")

    for pair in d.get("pairwise", []):
        for role in ("baseline", "challenger"):
            ref = pair.get(role, "")
            if ref and not any(ref in n for n in names):
                errors.append(f"pairwise '{pair.get('id')}' references missing '{ref}'")

    for exp in exps:
        if "practical_metrics" not in exp and ("LR" in exp["name"] or "GBDT" in exp["name"]):
            errors.append(f"'{exp['name']}' missing practical_metrics")

    return errors


def main() -> int:
    print("=== Phase3 JSON Self-Audit ===")
    errors = []
    for fname in ("phase3_metrics.json", "metric_exploration.json"):
        p = DATA / fname
        if not p.exists():
            errors.append(f"Missing {p}")
            continue
        print(f"  {fname}: {p.stat().st_size / 1024:.0f} KB")
        if fname == "phase3_metrics.json":
            errors.extend(audit_phase3(p))

    if errors:
        print(f"\nFAIL ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
