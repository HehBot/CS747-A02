import argparse, sys
import numpy as np


epsilon = 1e-9


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
        assert end_states != None, "End states not specified for episodic MDP"
        return episodic_MDP(S, A, T, R, gamma, end_states)
    else:
        assert end_states == None, "End states not specified for continuing MDP"
        return continuing_MDP(S, A, T, R, gamma)


def print_mdp(mdp: MDP):
    print(f"numStates {mdp.S}")
    print(f"numActions {mdp.A}")
    print("end ", end="")
    if type(mdp) == episodic_MDP:
        for e in mdp.end_states:
            print(e, "", end="")
        print()
    else:
        print("-1")
    for s1 in range(mdp.S):
        for a in range(mdp.A):
            for s2 in range(mdp.S):
                if mdp.T[s1][a][s2] > epsilon:
                    print(
                        f"transition {s1} {a} {s2} {mdp.R[s1][a][s2]} {mdp.T[s1][a][s2]}"
                    )
    print("mdptype ", end="")
    if type(mdp) == episodic_MDP:
        print("episodic")
    else:
        print("continuing")
    print(f"discount {mdp.gamma}")


def parse_policy(policy_file):
    f = open(policy_file, "r")
    p = [int(x.strip().split()[0]) for x in f]
    return np.array(p)


class algorithm:
    def get_optimal_value_policy(self, mdp):
        raise NotImplementedError

    def evaluate_policy(self, mdp, policy):
        T = mdp.T[np.arange(mdp.S), policy, :]
        R = mdp.R[np.arange(mdp.S), policy, :]
        return np.linalg.inv(np.identity(mdp.S) - mdp.gamma * T) @ ((T * R).sum(axis=1))


class value_iteration(algorithm):
    def get_optimal_value_policy(self, mdp):
        V = np.random.randn(mdp.S)
        while True:
            Vt = (
                (mdp.T * (mdp.R + mdp.gamma * V.reshape((1, 1, -1))))
                .sum(axis=2)
                .max(axis=1)
            )
            if (np.abs(Vt - V) < epsilon).all():
                break
            V = Vt
        p = (
            (mdp.T * (mdp.R + mdp.gamma * V.reshape((1, 1, -1))))
            .sum(axis=2)
            .argmax(axis=1)
        )
        return (V, p)


class howard_policy_iteration(algorithm):
    def get_optimal_value_policy(self, mdp):
        p = np.random.randint(0, mdp.A, (mdp.S,))
        Vp = self.evaluate_policy(mdp, p)
        while True:
            Qp = (mdp.T * (mdp.R + mdp.gamma * Vp.reshape((1, 1, -1)))).sum(axis=2)

            z = np.where(Qp > Vp.reshape((-1, 1)) + epsilon)
            IA = [(s, z[1][z[0] == s]) for s in range(mdp.S) if (z[0] == s).any()]
            if len(IA) == 0:
                break
            for s, ia in IA:
                p[s] = np.random.choice(ia)

            Vp = self.evaluate_policy(mdp, p)

        return (Vp, p)


class linear_programming(algorithm):
    def get_optimal_value_policy(self, mdp):
        import pulp

        V = np.array([pulp.LpVariable(f"V_s{s}") for s in range(mdp.S)])
        problem = pulp.LpProblem("mdp_lp", sense=pulp.LpMinimize)
        for s in range(mdp.S):
            for a in range(mdp.A):
                problem += V[s] >= (mdp.T[s][a] * (mdp.R[s][a] + mdp.gamma * V)).sum()
        problem += V.sum()
        problem.solve(pulp.PULP_CBC_CMD(msg=0))

        V = np.array([pulp.value(vs) for vs in list(V)])
        p = (
            (mdp.T * (mdp.R + mdp.gamma * V.reshape((1, 1, -1))))
            .sum(axis=2)
            .argmax(axis=1)
        )

        return (V, p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", type=str)
    parser.add_argument("--algorithm", type=str, default="hpi")
    parser.add_argument("--policy", type=str, default=None)
    args = parser.parse_args()

    assert args.algorithm in ["vi", "hpi", "lp"]

    mdp: MDP = parse_mdp(args.mdp)

    alg = None
    if args.algorithm == "vi":
        alg = value_iteration()
    elif args.algorithm == "hpi":
        alg = howard_policy_iteration()
    else:
        alg = linear_programming()

    ans = None
    if args.policy == None:
        ans = alg.get_optimal_value_policy(mdp)
    else:
        policy = parse_policy(args.policy)
        ans = (alg.evaluate_policy(mdp, policy), policy)

    ans = list(zip(list(ans[0]), list(ans[1])))
    for vi, pi in ans:
        print("%.6f" % (vi), pi)
