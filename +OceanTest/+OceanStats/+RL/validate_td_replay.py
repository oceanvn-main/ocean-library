"""Independent replay validation of the MATLAB TD learners.

export_td_replay.m runs many TDLearner configurations, RECORDING every episode (states, actions, rewards,
terminated / truncated, double-learning coin flips), and dumps the learner's final tables to td_replay_cases.json.
This script re-implements every update rule from the textbook definitions (Sutton & Barto 2018: eqs. 6.2, 6.7-6.9,
7.1-7.2, 12.17, algorithms on pp. 147, 305, 309), replays the SAME transitions with numpy and compares the final tables.
Because the transitions are fixed, no random-number stream is involved: any difference is a difference in the update math.

Run:  py validate_td_replay.py            (after OceanTest.OceanStats.RL.export_td_replay())
Exit status is non-zero if any case differs by more than TOL.
"""
import json
import os
import sys

import numpy as np

TOL = 1e-10
here = os.path.dirname(os.path.abspath(__file__))
cases = json.load(open(os.path.join(here, "td_replay_cases.json"), encoding="utf-8"))
if isinstance(cases, dict):
    cases = [cases]


def alpha_fn(cfg):
    ss = cfg["StepSize"]
    if ss["kind"] == "constant":
        return lambda n: ss["alpha"]
    return lambda n: min(1.0, ss["scale"] / max(n, 1) ** ss["power"])


def eps_greedy(q, valid, eps):
    p = np.zeros_like(q, dtype=float)
    nv = valid.sum()
    p[valid] = eps / nv
    best = valid & (q >= q[valid].max())
    p[best] += (1 - eps) / best.sum()
    return p


def masked_max(q, valid):
    return q[valid].max()


def replay_control(case):
    cfg = case["cfg"]
    valid = np.array(case["valid"], dtype=bool)
    S, A = valid.shape
    g, lam, n = cfg["Gamma"], cfg["Lambda"], cfg["N"]
    alpha = alpha_fn(cfg)
    Q = np.full((S, A), float(cfg["InitialValue"]))
    Q[~valid] = 0.0
    Q2 = Q.copy()
    cnt = np.zeros((S, A))
    cnt2 = np.zeros((S, A))
    E = np.zeros((S, A))
    target, mode, eps = cfg["Target"], cfg["Mode"], cfg["Exploration"]
    for ep in case["episodes"]:
        Sx = np.atleast_1d(np.array(ep["S"], dtype=int)) - 1
        Ax = np.atleast_1d(np.array(ep["A"], dtype=int)) - 1
        R = np.atleast_1d(np.array(ep["R"], dtype=float))
        T = len(R)
        term = bool(ep["Terminated"])
        coin = np.atleast_1d(np.array(ep["Coin"], dtype=bool))

        def is_last_terminal(t):
            return term and t == T - 1

        if cfg["Double"]:
            for t in range(T):
                s, a, s2 = Sx[t], Ax[t], Sx[t + 1]
                first, second = (Q, Q2) if coin[t] else (Q2, Q)
                if is_last_terminal(t):
                    y = R[t]
                else:
                    qq = np.where(valid[s2], first[s2], -np.inf)
                    astar = int(np.argmax(qq))
                    y = R[t] + g * second[s2, astar]
                c = cnt if coin[t] else cnt2                          # one visit count per table
                c[s, a] += 1
                first[s, a] += alpha(c[s, a]) * (y - first[s, a])
        elif mode == "onestep":
            for t in range(T):
                s, a, s2 = Sx[t], Ax[t], Sx[t + 1]
                if is_last_terminal(t):
                    y = R[t]
                elif target == "sarsa":
                    y = R[t] + g * Q[s2, Ax[t + 1]]
                elif target == "qlearning":
                    y = R[t] + g * masked_max(Q[s2], valid[s2])
                elif target == "expectedsarsa":
                    p = eps_greedy(Q[s2], valid[s2], eps)
                    y = R[t] + g * float((p[valid[s2]] * Q[s2][valid[s2]]).sum())
                else:
                    raise ValueError(target)
                cnt[s, a] += 1
                Q[s, a] += alpha(cnt[s, a]) * (y - Q[s, a])
        elif mode == "nstep":
            def boot(sn, an):
                if target == "sarsa":
                    return Q[sn, an]
                if target == "expectedsarsa":
                    p = eps_greedy(Q[sn], valid[sn], eps)
                    return float((p[valid[sn]] * Q[sn][valid[sn]]).sum())
                raise ValueError(target)
            for tau in range(T):
                k = min(n, T - tau)
                G = sum(g ** i * R[tau + i] for i in range(k))
                if tau + n < T:
                    G += g ** n * boot(Sx[tau + n], Ax[tau + n])
                elif not term:                                        # truncated: bootstrap from S_T
                    G += g ** k * boot(Sx[T], Ax[T])
                s, a = Sx[tau], Ax[tau]
                cnt[s, a] += 1
                Q[s, a] += alpha(cnt[s, a]) * (G - Q[s, a])
        elif mode == "trace":
            E[:] = 0.0
            replacing = cfg["Trace"] == "replacing"
            for t in range(T):
                s, a, s2 = Sx[t], Ax[t], Sx[t + 1]
                if is_last_terminal(t):
                    y = R[t]
                elif target == "sarsa":
                    y = R[t] + g * Q[s2, Ax[t + 1]]
                elif target == "qlearning":
                    y = R[t] + g * masked_max(Q[s2], valid[s2])
                elif target == "expectedsarsa":
                    p = eps_greedy(Q[s2], valid[s2], eps)
                    y = R[t] + g * float((p[valid[s2]] * Q[s2][valid[s2]]).sum())
                else:
                    raise ValueError(target)
                # Watkins: the action A' was drawn from the Q BEFORE this step's update -> judge greediness on that Q
                cut = (target == "qlearning" and not is_last_terminal(t)
                       and Q[s2, Ax[t + 1]] < masked_max(Q[s2], valid[s2]))
                delta = y - Q[s, a]
                if replacing:
                    E[s, a] = 1.0
                else:
                    E[s, a] += 1.0
                cnt[s, a] += 1
                Q += alpha(cnt[s, a]) * delta * E
                E *= g * lam
                if cut:
                    E[:] = 0.0
        else:
            raise ValueError(mode)
    if cfg["Double"]:
        Q = (Q + Q2) / 2
    return Q


def replay_prediction(case):
    cfg = case["cfg"]
    valid = np.array(case["valid"], dtype=bool)
    S = valid.shape[0]
    g, lam, n = cfg["Gamma"], cfg["Lambda"], cfg["N"]
    alpha = alpha_fn(cfg)
    V = np.full(S, float(cfg["InitialValue"]))
    cnt = np.zeros(S)
    E = np.zeros(S)
    mode = cfg["Mode"]
    for ep in case["episodes"]:
        Sx = np.atleast_1d(np.array(ep["S"], dtype=int)) - 1
        R = np.atleast_1d(np.array(ep["R"], dtype=float))
        T = len(R)
        term = bool(ep["Terminated"])
        if mode == "onestep":
            for t in range(T):
                s, s2 = Sx[t], Sx[t + 1]
                y = R[t] if (term and t == T - 1) else R[t] + g * V[s2]
                cnt[s] += 1
                V[s] += alpha(cnt[s]) * (y - V[s])
        elif mode == "nstep":
            for tau in range(T):
                k = min(n, T - tau)
                G = sum(g ** i * R[tau + i] for i in range(k))
                if tau + n < T:
                    G += g ** n * V[Sx[tau + n]]
                elif not term:
                    G += g ** k * V[Sx[T]]
                s = Sx[tau]
                cnt[s] += 1
                V[s] += alpha(cnt[s]) * (G - V[s])
        elif mode == "trace":
            E[:] = 0.0
            offline = bool(cfg["Offline"])
            replacing = cfg["Trace"] == "replacing"
            dV = np.zeros(S)
            for t in range(T):
                s, s2 = Sx[t], Sx[t + 1]
                y = R[t] if (term and t == T - 1) else R[t] + g * V[s2]
                delta = y - V[s]
                if replacing:
                    E[s] = 1.0
                else:
                    E[s] += 1.0
                cnt[s] += 1
                step = alpha(cnt[s]) * delta * E
                if offline:
                    dV += step
                else:
                    V += step
                E *= g * lam
            if offline:
                V += dV
        elif mode == "lambdareturn":
            V0 = V.copy()                                             # frozen during the episode
            G = np.zeros(T)
            bootT = 0.0 if term else V0[Sx[T]]
            G[T - 1] = R[T - 1] + g * bootT
            for t in range(T - 2, -1, -1):
                G[t] = R[t] + g * ((1 - lam) * V0[Sx[t + 1]] + lam * G[t + 1])
            dV = np.zeros(S)
            for t in range(T):
                s = Sx[t]
                cnt[s] += 1
                dV[s] += alpha(cnt[s]) * (G[t] - V0[s])
            V += dV
        else:
            raise ValueError(mode)
    return V


bad = 0
for case in cases:
    if case["cfg"]["Task"] == "prediction":
        got = replay_prediction(case)
        want = np.array(case["V"], dtype=float).reshape(-1)
    else:
        got = replay_control(case)
        want = np.array(case["Q"], dtype=float)
        got = got.reshape(want.shape)
    diff = float(np.max(np.abs(got - want)))
    ok = diff <= TOL
    bad += not ok
    n_steps = sum(np.atleast_1d(e["R"]).size for e in case["episodes"])
    print(f"[{'PASS' if ok else 'FAIL'}] {case['name']:<32s} {len(case['episodes']):4d} episodes {n_steps:6d} steps   max|MATLAB - replay| = {diff:.2e}")
print(f"\n{len(cases) - bad}/{len(cases)} cases agree to {TOL:g}")
sys.exit(1 if bad else 0)
