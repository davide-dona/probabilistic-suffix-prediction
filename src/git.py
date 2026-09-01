import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GitState:
    """Which commit produced a run, and whether it was clean.

    Carried inside a checkpoint and a W&B run's config, so a set of weights is never left
    unable to say what code wrote them.
    """

    commit: str
    dirty: bool


def current_git_state() -> GitState:
    """The commit checked out where this process runs, and whether it has uncommitted changes.

    Returns:
        The current `GitState`.
    """
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ['git', 'status', '--porcelain'], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    return GitState(commit=commit, dirty=dirty)
