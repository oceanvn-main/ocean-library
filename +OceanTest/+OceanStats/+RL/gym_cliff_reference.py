"""Gymnasium CliffWalking-v1 transition table -> gym_cliff_reference.json, for compare_td_envs_in_matlab.m.
Format: for every state s (0-based row*12+col) and action a (0 up, 1 right, 2 down, 3 left):
[next_state, reward, terminated] taken from env.unwrapped.P[s][a][0]. Run: py gym_cliff_reference.py"""
import json
import os

import gymnasium as gym

env = gym.make("CliffWalking-v1").unwrapped
P = env.P
cols = lambda k, f: [[f(P[s][a][0][k]) for a in range(4)] for s in range(env.nS)]
out = {"nS": int(env.nS), "nA": int(env.nA), "next": cols(1, int), "reward": cols(2, float), "terminated": cols(3, int)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_cliff_reference.json"), "w"))
print("states", env.nS, "actions", env.nA)
