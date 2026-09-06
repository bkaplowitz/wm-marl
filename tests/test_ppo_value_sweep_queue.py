"""Promotion must use complete seed sets and reject a large map regression."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_ppo_value_sweep as sweep


def fixture_jobs():
    performance = {
        "base": (0.25, 0.45),
        "fact": (0.40, 0.60),
        "factrep": (0.45, 0.70),
        "actor1": (0.99, 0.10),
    }
    jobs = []
    for seed in (0, 1):
        for map_index, map_name in enumerate(("2s3z", "3s_vs_3z")):
            for arm in sweep.ARMS:
                job = sweep.make_job(sweep.run_spec(arm, map_name, seed), len(jobs), 0)
                job.update(
                    status="complete",
                    summary={
                        "win_rate": performance.get(arm, (0.10, 0.10))[map_index],
                    },
                )
                jobs.append(job)
    return jobs


def test_promotion_uses_balanced_complete_results_and_preserves_indices():
    jobs = fixture_jobs()
    chosen, _, _ = sweep.select_candidates(jobs)
    assert chosen == ["factrep", "fact"]
    original = [job["name"] for job in jobs]
    state = dict(jobs=jobs, phase="screen")
    sweep.advance(state)
    assert state["phase"] == "validation"
    assert len(state["jobs"]) == 72
    assert [job["name"] for job in jobs[:48]] == original
    assert len({job["name"] for job in jobs}) == 72
    assert [job["index"] for job in jobs] == list(range(72))
    assert all(job["stage"] == 1 for job in jobs[48:])


def test_incomplete_reference_blocks_automatic_promotion():
    jobs = fixture_jobs()
    next(job for job in jobs if job["arm"] == "base")["status"] = "failed"
    chosen, _, reason = sweep.select_candidates(jobs)
    assert not chosen
    assert "Reference incomplete" in reason
