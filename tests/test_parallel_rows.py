"""A seed whose episode raises comes back as a failed row with its error, instead of discarding the whole run."""
from tenplaces import parallel_eval


class BrokenPolicy:
    def reset(self):
        pass

    def select_action(self, obs):
        raise RuntimeError("policy exploded")


def test_table_job_turns_an_exception_into_a_failed_row():
    parallel_eval._W.update(policy=BrokenPolicy(), checker=None)
    row = parallel_eval._table_job((7, 0, None, None))
    assert row["seed"] == 7 and row["success"] is False and row["subtasks_done"] == 0
    assert all(row[k] is False for k in parallel_eval.STEP_KEYS)
    assert "policy exploded" in row["error"]
