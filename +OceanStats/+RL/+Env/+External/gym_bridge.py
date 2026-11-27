"""Thin helper so MATLAB exchanges only plain floats / numpy arrays with Gymnasium."""
import warnings
import numpy as np
import gymnasium as gym

warnings.filterwarnings("ignore")


def _obs(o):
    return np.atleast_1d(np.asarray(o, dtype=np.float64)).ravel()


def _mask(info, n):
    m = info.get("action_mask") if isinstance(info, dict) else None
    return np.ones(n) if m is None else np.asarray(m, dtype=np.float64).ravel()


class Bridge:
    def __init__(self, name):
        self.env = gym.make(name)
        self.n = int(self.env.action_space.n)

    def spaces(self):
        sp = self.env.observation_space
        if isinstance(sp, gym.spaces.Discrete):
            return "discrete", float(sp.n), np.zeros(1), np.zeros(1), float(self.n)
        return "box", 0.0, sp.low.astype(np.float64).ravel(), sp.high.astype(np.float64).ravel(), float(self.n)

    def reset(self, seed):
        o, info = self.env.reset(seed=None if seed < 0 else int(seed))
        return _obs(o), _mask(info, self.n)

    def step(self, a):
        o, r, term, trunc, info = self.env.step(int(a))
        return _obs(o), float(r), bool(term), bool(trunc), _mask(info, self.n)

    def model(self):
        """Dense copy of env.unwrapped.P: arrays S x A x K (padded with prob 0)."""
        P = self.env.unwrapped.P
        S, A = len(P), self.n
        K = max(len(P[s][a]) for s in range(S) for a in range(A))
        nxt = np.zeros((S, A, K)); prob = np.zeros((S, A, K)); rew = np.zeros((S, A, K)); done = np.zeros((S, A, K))
        for s in range(S):
            for a in range(A):
                for k, (p, s2, r, d) in enumerate(P[s][a]):
                    nxt[s, a, k], prob[s, a, k], rew[s, a, k], done[s, a, k] = s2, p, r, d
        valid = np.ones((S, A))
        if hasattr(self.env.unwrapped, "action_mask"):
            valid = np.array([self.env.unwrapped.action_mask(s) for s in range(S)], dtype=np.float64)
        return nxt, prob, rew, done, valid


def reference(name, seed, actions, resets):
    """Raw Gymnasium rollout, no adapter: rows [obs..., reward, terminated, truncated]; reset (unseeded) when resets[t]."""
    env = gym.make(name)
    o, _ = env.reset(seed=int(seed))
    rows = [np.concatenate([_obs(o), [0.0, 0.0, 0.0]])]
    for a, z in zip(actions, resets):
        if z:
            o, _ = env.reset()
            rows.append(np.concatenate([_obs(o), [0.0, 0.0, 0.0]]))
            continue
        o, r, term, trunc, _ = env.step(int(a))
        rows.append(np.concatenate([_obs(o), [float(r), float(term), float(trunc)]]))
    return np.array(rows)
