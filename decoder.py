import argparse
import numpy as np
from football_util import states, index_to_coords, state_to_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", type=str)
    parser.add_argument("--value-policy", type=str)
    args = parser.parse_args()
    assert args.opponent != None, "Opponent file not specified"
    assert args.value_policy != None, "Value-policy file not specified"
    return (args.opponent, args.value_policy)


def parse_value_policy(value_policy_file):
    f = open(value_policy_file, "r")
    V = []
    P = []
    for l in f:
        v, a = l.strip().split()
        v = float(v)
        a = int(a)
        V.append(v)
        P.append(a)
    return (np.array(V), np.array(P))


def main():
    opponent_file, value_policy_file = parse_args()
    value, policy = parse_value_policy(value_policy_file)

    f = open(opponent_file)
    f.readline()
    for l in f:
        state_string = l.strip().split()[0]
        b0_index = int(state_string[:2])
        b1_index = int(state_string[2:4])
        opp_index = int(state_string[4:6])
        poss = int(state_string[6]) - 1
        if b0_index > b1_index:
            b0_index, b1_index = b1_index, b0_index
            poss = 1 - poss
        b = (index_to_coords(b0_index), index_to_coords(b1_index))
        opp = index_to_coords(opp_index)
        state = (b, opp, poss)
        s = state_to_index(state)
        print("%s %d %0.6f" % (state_string, policy[s], value[s]))


if __name__ == "__main__":
    main()
