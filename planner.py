import argparse, sys
import numpy as np


class MDP:
    def __init__(self, S, A, T, R, gamma):
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma
        assert T.shape == (S, A, S)
        assert R.shape == (S, A, S)


class continuing_MDP(MDP):
    def __init__(self, S, A, T, R, gamma):
        super().__init__(S, A, T, R, gamma)


class episodic_MDP(MDP):
    def __init__(self, S, A, T, R, gamma, end_states):
        super().__init__(S, A, T, R, gamma)
        self.end_states = end_states


def parse_mdp(mdp_file):
    f = open(mdp_file, "r")

    line = f.readline().strip().split()
    assert line[0] == "numStates"
    S = int(line[1])

    line = f.readline().strip().split()
    assert line[0] == "numActions"
    A = int(line[1])

    line = f.readline().strip().split()
    assert line[0] == "end"
    end_states = [int(x) for x in line[1:]]
    if end_states[0] == -1:
        end_states = None

    line = f.readline().strip().split()
    T = np.zeros((S, A, S))
    R = np.zeros((S, A, S))
    while line[0] == "transition":
        s1 = int(line[1])
        a = int(line[2])
        s2 = int(line[3])
        R[s1][a][s2] = float(line[4])
        T[s1][a][s2] = float(line[5])
        line = f.readline().strip().split()

    assert line[0] == "mdptype"
    mdptype = line[1]
    assert mdptype == "episodic" or mdptype == "continuing"

    line = f.readline().strip().split()
    assert line[0] == "discount"
    gamma = float(line[1])

    if mdptype == "episodic":
        return episodic_MDP(S, A, T, R, gamma, end_states)
    else:
        return continuing_MDP(S, A, T, R, gamma)


def parse_policy(policy_file):
    f = open(policy_file, "r")
    p = [int(x.strip().split()[0]) for x in f]
    return np.array(p)


if __name__ == "__main__":
    args = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", type=str)
    parser.add_argument("--algorithm", type=str, default="vi")
    parser.add_argument("--policy", type=str, default=None)
    args = parser.parse_args()

    mdp = parse_mdp(args.mdp)
    policy = None
    if args.policy != None:
        policy = parse_policy(args.policy)

    print(args.mdp)
    print(args.algorithm)
    print(args.policy)

    print(mdp.S)
    print(mdp.A)
    print(mdp.T)
    print(mdp.R)
    print(mdp.gamma)
