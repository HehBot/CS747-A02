import argparse
import numpy as np
from planner import episodic_MDP, print_mdp
from football_util import (
    S,
    A,
    A_opp,
    index_to_coords,
    coords_to_index,
    state_to_index,
    states,
    parse_opponent,
)


def move(pos, action):
    if action == 0:
        return (pos[0] - 1, pos[1])
    elif action == 1:
        return (pos[0] + 1, pos[1])
    elif action == 2:
        return (pos[0], pos[1] + 1)
    elif action == 3:
        return (pos[0], pos[1] - 1)
    assert False, "Illegal action"


def out_of_bounds(pos):
    return pos[0] < 0 or pos[0] > 3 or pos[1] < 0 or pos[1] > 3


def chess_distance(pos1, pos2):
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def in_line(pos1, pos_middle, pos2):
    if (pos1[0] - pos_middle[0]) * (pos2[0] - pos_middle[0]) > 0:
        return False
    if (pos1[1] - pos_middle[1]) * (pos2[1] - pos_middle[1]) > 0:
        return False
    return (pos1[0] - pos_middle[0]) * (pos2[1] - pos_middle[1]) == (
        pos1[1] - pos_middle[1]
    ) * (pos2[0] - pos_middle[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", type=str)
    parser.add_argument("--p", type=float)
    parser.add_argument("--q", type=float)
    args = parser.parse_args()
    assert args.opponent != None, "Opponent file not specified"
    assert args.p != None, "p not specified"
    assert args.q != None, "q not specified"
    assert args.p >= 0 and args.p <= 0.5, "p must lie in interval [0.0, 0.5]"
    assert args.q >= 0.6 and args.q <= 1.0, "q must lie in interval [0.6, 1.0]"
    return (args.opponent, args.p, args.q)


def main():
    opponent_file, p, q = parse_args()

    loss_s, win_s = state_to_index("loss"), state_to_index("win")

    opp_state_action_probs = parse_opponent(opponent_file)

    T = np.zeros((S, A, S))

    for s in range(S):
        if s == loss_s or s == win_s:
            continue
        b, opp, poss = states[s]
        opp_action_probs = opp_state_action_probs[s]

        for opp_action in range(A_opp):
            new_opp = move(opp, opp_action)
            opp_prob = opp_action_probs[opp_action]
            cond_prob = None

            if out_of_bounds(new_opp):
                continue

            # player 0 moves
            for action in range(0, 4):
                new_b = (move(b[0], action), b[1])
                new_poss = poss

                if out_of_bounds(new_b[0]):
                    T[s][action][loss_s] = 1
                else:
                    if coords_to_index(new_b[0]) > coords_to_index(new_b[1]):
                        new_b = (new_b[1], new_b[0])
                        new_poss = 1 - new_poss

                    if new_b[new_poss] == new_opp or (
                        new_b[new_poss] == opp and b[poss] == new_opp
                    ):
                        # tackling
                        cond_prob = 0.5 - p
                    else:
                        # movement
                        if poss == 0:
                            cond_prob = 1 - 2 * p
                        else:
                            cond_prob = 1 - p

                    T[s][action][state_to_index((new_b, new_opp, new_poss))] += (
                        cond_prob * opp_prob
                    )
                    T[s][action][loss_s] += (1 - cond_prob) * opp_prob

            # player 1 moves
            for action in range(4, 8):
                new_b = (b[0], move(b[1], action - 4))
                new_poss = poss

                if out_of_bounds(new_b[1]):
                    T[s][action][loss_s] = 1
                else:
                    if coords_to_index(new_b[0]) > coords_to_index(new_b[1]):
                        new_b = (new_b[1], new_b[0])
                        new_poss = 1 - new_poss

                    if new_b[new_poss] == new_opp or (
                        new_b[new_poss] == opp and b[poss] == new_opp
                    ):
                        # tackling
                        cond_prob = 0.5 - p
                    else:
                        # movement
                        if poss == 1:
                            cond_prob = 1 - 2 * p
                        else:
                            cond_prob = 1 - p

                    T[s][action][state_to_index((new_b, new_opp, new_poss))] += (
                        cond_prob * opp_prob
                    )
                    T[s][action][loss_s] += (1 - cond_prob) * opp_prob

            # pass
            action = 8
            new_b = b
            new_poss = 1 - poss

            cond_prob = q - 0.1 * chess_distance(new_b[0], new_b[1])
            if in_line(new_b[0], new_opp, new_b[1]):
                cond_prob /= 2

            T[s][action][state_to_index((new_b, new_opp, new_poss))] += (
                cond_prob * opp_prob
            )
            T[s][action][loss_s] += (1 - cond_prob) * opp_prob

            # shoot
            action = 9
            cond_prob = q - 0.2 * (3 - b[poss][0])
            if opp[0] == 3 and (opp[1] == 1 or opp[1] == 2):
                cond_prob /= 2

            T[s][action][win_s] += cond_prob * opp_prob
            T[s][action][loss_s] += (1 - cond_prob) * opp_prob

    R = np.zeros((S, A, S))
    for s in range(S):
        if s == win_s or s == loss_s:
            continue
        for a in range(A):
            R[s][a][win_s] = 1

    mdp = episodic_MDP(S, A, T, R, 1.0, [loss_s, win_s])
    print_mdp(mdp)


if __name__ == "__main__":
    main()
