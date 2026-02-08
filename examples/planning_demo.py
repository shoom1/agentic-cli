#!/usr/bin/env python
"""Standalone demo for the PlanStore planning system.

This demo shows the simplified flat-markdown planning pattern:
1. Agent creates a plan with markdown checkboxes
2. Agent updates the plan as tasks complete
3. Framework just stores the string

Usage:
    conda run -n agenticcli python examples/planning_demo.py
"""

from agentic_cli.tools.planning_tools import PlanStore


def demo_basic_plan():
    """Demo creating and retrieving a plan."""
    print("\n" + "=" * 60)
    print("Basic Plan Demo")
    print("=" * 60)

    store = PlanStore()

    # Agent creates a plan
    plan = (
        "## Research Plan: Python History\n"
        "\n"
        "- [ ] Search for key milestones in Python development\n"
        "- [ ] Identify major version releases and their features\n"
        "- [ ] Find notable Python community events\n"
        "- [ ] Write summary document\n"
    )
    store.save(plan)

    print("  Created plan:")
    print(store.get())
    print()


def demo_progress_tracking():
    """Demo updating plan with progress."""
    print("\n" + "=" * 60)
    print("Progress Tracking Demo")
    print("=" * 60)

    store = PlanStore()

    # Initial plan
    store.save(
        "## ML Pipeline\n"
        "- [ ] Prepare dataset\n"
        "- [ ] Train model\n"
        "- [ ] Evaluate performance\n"
        "- [ ] Deploy to production\n"
    )
    print("  Initial plan:")
    print(store.get())

    # Agent completes first two tasks
    store.save(
        "## ML Pipeline\n"
        "- [x] Prepare dataset\n"
        "- [x] Train model (accuracy: 94.2%)\n"
        "- [ ] Evaluate performance\n"
        "- [ ] Deploy to production\n"
    )
    print("\n  After completing 2 tasks:")
    print(store.get())

    # Agent adds notes and completes more
    store.save(
        "## ML Pipeline\n"
        "- [x] Prepare dataset\n"
        "- [x] Train model (accuracy: 94.2%)\n"
        "- [x] Evaluate performance\n"
        "  - Precision: 0.93, Recall: 0.95\n"
        "  - Exceeds baseline by 12%\n"
        "- [ ] Deploy to production\n"
    )
    print("\n  After evaluation with notes:")
    print(store.get())
    print()


def demo_plan_revision():
    """Demo revising a plan mid-execution."""
    print("\n" + "=" * 60)
    print("Plan Revision Demo")
    print("=" * 60)

    store = PlanStore()

    # Original plan
    store.save(
        "## API Integration\n"
        "- [x] Design API schema\n"
        "- [ ] Implement endpoints\n"
        "- [ ] Write tests\n"
    )
    print("  Original plan:")
    print(store.get())

    # Agent discovers need for auth — revises plan
    store.save(
        "## API Integration\n"
        "- [x] Design API schema\n"
        "- [x] Implement endpoints\n"
        "- [ ] Add authentication (discovered requirement)\n"
        "- [ ] Write tests\n"
        "- [ ] Update documentation\n"
    )
    print("\n  Revised plan (added auth + docs):")
    print(store.get())
    print()


def demo_clear():
    """Demo clearing the plan."""
    print("\n" + "=" * 60)
    print("Clear Plan Demo")
    print("=" * 60)

    store = PlanStore()
    store.save("- [ ] Some task")
    print(f"  Has plan: {not store.is_empty()}")

    store.clear()
    print(f"  After clear, is empty: {store.is_empty()}")
    print()


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  PlanStore Planning System Demo")
    print("#" * 60)

    demo_basic_plan()
    demo_progress_tracking()
    demo_plan_revision()
    demo_clear()

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
