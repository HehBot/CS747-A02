import numpy as np
import scipy.sparse as sp


epsilon = 1e-9


class MDP:
    def __init__(self, S, A, T, R, gamma):
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma
        assert T.shape == (S,)
        assert R.shape == (S,)
        for s in range(S):
            assert T[s].shape == (A, S)
            assert R[s].shape == (A, S)
        if self.S**2 * self.A < 1e8:
            self.small = True
            self.T_numpy = np.array([self.T[s].toarray() for s in range(self.S)])
            self.R_numpy = np.array([self.R[s].toarray() for s in range(self.S)])
        else:
            self.small = False
            self.T_numpy = None
            self.R_numpy = None

    def evaluate_policy(self, policy):
        if self.small:
            T = self.T_numpy[np.arange(self.S), policy, :]
            R = self.R_numpy[np.arange(self.S), policy, :]
            return np.linalg.inv(np.identity(self.S) - self.gamma * T) @ (T * R).sum(
                axis=1
            )
        else:
            T = sp.csc_array(
                [self.T[s][[policy[s]], :].toarray().flatten() for s in range(self.S)]
            )
            R = sp.csc_array(
                [self.R[s][[policy[s]], :].toarray().flatten() for s in range(self.S)]
            )
            return sp.linalg.inv(sp.identity(self.S).tocsc() - self.gamma * T) @ (
                T * R
            ).sum(axis=1)

    def get_state_action_function(self, V):
        if self.small:
            return (
                self.T_numpy * (self.R_numpy + self.gamma * V.reshape((1, 1, -1)))
            ).sum(axis=2)
        else:
            Vsp = V.reshape((1, -1)) + np.zeros((self.A, self.S))
            Vsp = sp.csc_array(Vsp)
            return np.array(
                [
                    (self.T[s] * (self.R[s] + self.gamma * Vsp)).sum(axis=1)
                    for s in range(self.S)
                ]
            )

    def print(self):
        raise NotImplementedError

    def print_transitions(self):
        for s1 in range(self.S):
            coo = sp.coo_array(self.T[s1])
            for a, s2, t in zip(coo.row, coo.col, coo.data):
                print(
                    "transition %d %d %d %0.8f %0.8f"
                    % (s1, a, s2, self.R[s1][a, s2], t)
                )


class continuing_MDP(MDP):
    def __init__(self, S, A, T, R, gamma):
        super().__init__(S, A, T, R, gamma)

    def print(self):
        print("numStates %d" % (self.S))
        print("numActions %d" % (self.A))
        print("end -1")
        self.print_transitions()
        print("mdptype continuing")
        print("discount %f" % (self.gamma))


class episodic_MDP(MDP):
    def __init__(self, S, A, T, R, gamma, end_states):
        super().__init__(S, A, T, R, gamma)
        self.end_states = end_states

    def print(self):
        print("numStates %d" % (self.S))
        print("numActions %d" % (self.A))
        print("end ", end="")
        for e in self.end_states:
            print(e, end=" ")
        print()
        self.print_transitions()
        print("mdptype episodic")
        print("discount %f" % (self.gamma))


def parse_mdp(mdp_file):
    f = None
    if mdp_file == None:
        f = open(0, "r")
    else:
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
    T = np.array([sp.lil_array((A, S)) for s in range(S)])
    R = np.array([sp.lil_array((A, S)) for s in range(S)])
    while line[0] == "transition":
        s1 = int(line[1])
        a = int(line[2])
        s2 = int(line[3])
        R[s1][a, s2] = float(line[4])
        T[s1][a, s2] = float(line[5])
        line = f.readline().strip().split()

    assert line[0] == "mdptype"
    mdptype = line[1]
    assert mdptype == "episodic" or mdptype == "continuing"

    line = f.readline().strip().split()
    assert line[0] == "discount"
    gamma = float(line[1])

    f.close()

    if mdptype == "episodic":
        assert end_states != None, "End states not specified for episodic MDP"
        return episodic_MDP(S, A, T, R, gamma, end_states)
    else:
        assert end_states == None, "End states specified for continuing MDP"
        return continuing_MDP(S, A, T, R, gamma)


def parse_policy(policy_file):
    f = open(policy_file, "r")
    p = [int(x.strip().split()[0]) for x in f]
    f.close()
    return np.array(p)


class algorithm:
    def get_optimal_value_policy(self, mdp):
        raise NotImplementedError


class value_iteration(algorithm):
    def get_optimal_value_policy(self, mdp):
        V = np.random.randn(mdp.S)
        while True:
            Vt = mdp.get_state_action_function(V).max(axis=1)
            if np.abs(Vt - V).max() < epsilon:
                break
            V = Vt
        p = mdp.get_state_action_function(V).argmax(axis=1)
        return (V, p)


class howard_policy_iteration(algorithm):
    def get_optimal_value_policy(self, mdp):
        p = np.random.randint(0, mdp.A, (mdp.S,))
        V = mdp.evaluate_policy(p)
        while True:
            Q = mdp.get_state_action_function(V)
            z = np.where(Q > V.reshape((-1, 1)) + epsilon)
            IA = [(s, z[1][z[0] == s]) for s in range(mdp.S) if (z[0] == s).any()]
            if len(IA) == 0:
                break
            for s, ia in IA:
                p[s] = np.random.choice(ia)
            V = mdp.evaluate_policy(p)
        return (V, p)


class linear_programming(algorithm):
    def get_optimal_value_policy(self, mdp):
        import pulp

        V = np.array([pulp.LpVariable(f"V_s{s}") for s in range(mdp.S)])
        problem = pulp.LpProblem("mdp_lp", sense=pulp.LpMinimize)
        Q = mdp.get_state_action_function(V)
        for s in range(mdp.S):
            for a in range(mdp.A):
                problem += V[s] >= Q[s][a]
        problem += V.sum()
        problem.solve(pulp.PULP_CBC_CMD(msg=0))

        V = np.array([pulp.value(vs) for vs in V])
        p = mdp.get_state_action_function(V).argmax(axis=1)
        return (V, p)


def main(mdp_file, algorithm, policy_file):
    assert algorithm in ["vi", "hpi", "lp"]
    alg = None
    if algorithm == "vi":
        alg = value_iteration()
    elif algorithm == "hpi":
        alg = howard_policy_iteration()
    else:
        alg = linear_programming()

    mdp: MDP = parse_mdp(mdp_file)

    ans = None
    if policy_file == None:
        ans = alg.get_optimal_value_policy(mdp)
    else:
        policy = parse_policy(policy_file)
        ans = (mdp.evaluate_policy(policy), policy)

    ans = list(zip(list(ans[0]), list(ans[1])))
    for vi, pi in ans:
        print("%.6f" % (vi), pi)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", type=str, default=None, help="MDP_FILE")
    parser.add_argument("--algorithm", type=str, default="hpi", help="ALGORITHM")
    parser.add_argument("--policy", type=str, default=None, help="POLICY_FILE")
    args = parser.parse_args()

    main(args.mdp, args.algorithm, args.policy)
