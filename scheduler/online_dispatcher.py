from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class DispatchContext:
    leader_id: int
    action: int
    vacancy: int
    group: Sequence[int]


class OnlineDispatcher:
    def select_members(self, context: DispatchContext, env) -> List[int]:
        raise NotImplementedError


class BaselineRandomDispatcher(OnlineDispatcher):
    """Behavior-compatible with the existing random follower selection."""

    def select_members(self, context: DispatchContext, env) -> List[int]:
        peers = [aid for aid in context.group if aid != context.leader_id]
        n_followers = int(max(0, min(context.vacancy - 1, len(peers))))
        followers = []
        if n_followers > 0:
            followers = env.random_choice(peers, n_followers, False).tolist()
        return [int(context.leader_id)] + [int(x) for x in followers]
