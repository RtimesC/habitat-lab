"""Summary artifacts for the grounded Habitat test-scene engineering demo."""

import json
import os
from typing import Any, Mapping, Sequence


def write_grounded_demo_artifacts(
    run_dir: str,
    episode_summaries: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Write machine-readable and human-readable grounded demo summaries."""
    if not episode_summaries:
        raise ValueError("at least one episode summary is required")

    json_path = os.path.join(run_dir, "episode_summary.json")
    markdown_path = os.path.join(run_dir, "summary.md")
    payload = dict(episode_summaries[0])
    payload["episode_count"] = len(episode_summaries)
    payload["episodes"] = [dict(summary) for summary in episode_summaries]

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("# Grounded Official NaVIDA demo\n\n")
        handle.write(
            "This is a Habitat test-scene engineering demo. "
            "`benchmark_comparable=false`.\n\n"
        )
        handle.write(
            "It must not be reported as an R2R, RxR, MP3D, or HM3D "
            "benchmark result.\n\n"
        )
        handle.write(
            "Oracle validation is an environment/route upper-bound check; "
            "it is separate from the Official NaVIDA result below.\n\n"
        )
        handle.write(
            "| episode_id | instruction_source | steps | stopped | failed | "
            "success | SPL | final_distance | video_generated | video_error | "
            "failure_reason |\n"
        )
        handle.write("|---|---|---:|---|---|---:|---:|---:|---|---|---|\n")
        for summary in episode_summaries:
            handle.write(
                f"| {summary['episode_id']} | {summary['instruction_source']} | "
                f"{summary['steps']} | {summary['stopped']} | "
                f"{summary['failed']} | {summary['success']} | "
                f"{summary['spl']} | {summary['final_distance']} | "
                f"{summary['video_generated']} | "
                f"{summary['video_error']} | "
                f"{summary['failure_reason']} |\n"
            )

    return json_path, markdown_path
