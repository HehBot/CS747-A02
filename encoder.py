import numpy as np
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


def chess_distance(pos0, pos1):
    return max(abs(pos0[0] - pos1[0]), abs(pos0[1] - pos1[1]))


def in_line(pos0, pos_middle, pos1):
    if (pos0[0] - pos_middle[0]) * (pos1[0] - pos_middle[0]) > 0:
        return False
    if (pos0[1] - pos_middle[1]) * (pos1[1] - pos_middle[1]) > 0:
        return False
    return (
        (pos0[0] == pos_middle[0] and pos1[0] == pos_middle[0])
        or (pos0[1] == pos_middle[1] and pos1[1] == pos_middle[1])
        or (
            (pos0[0] - pos_middle[0]) * (pos1[1] - pos_middle[1])
            == (pos0[1] - pos_middle[1]) * (pos1[0] - pos_middle[0])
            and abs(pos0[0] - pos_middle[0]) == abs(pos0[1] - pos_middle[1])
            and abs(pos1[0] - pos_middle[0]) == abs(pos1[1] - pos_middle[1])
        )
    )


def main(opponent_file, p, q):
    loss_s, win_s = state_to_index("loss"), state_to_index("win")

    print("numStates %d" % (S))
    print("numActions %d" % (A))
    print("end %d %d" % (loss_s, win_s))

    opp_state_action_probs = parse_opponent(opponent_file)

    for s in range(S):
        if s == loss_s or s == win_s:
            continue

        T = np.zeros((A, S))

        old_b, old_opp, old_poss = states[s]
        opp_action_probs = opp_state_action_probs[s]

        for opp_action in range(A_opp):
            new_opp = move(old_opp, opp_action)
            opp_prob = opp_action_probs[opp_action]
            cond_prob = None

            if out_of_bounds(new_opp):
                continue

            # player 0 moves
            for action in range(0, 4):
                new_b = (move(old_b[0], action), old_b[1])
                new_poss = old_poss

                if out_of_bounds(new_b[0]):
                    T[action][loss_s] += 1.0 * opp_prob
                else:
                    # movement
                    if old_poss == 0:
                        cond_prob = 1.0 - 2.0 * p
                        # tackling
                        if new_b[0] == new_opp or (
                            new_b[0] == old_opp and old_b[0] == new_opp
                        ):
                            cond_prob *= 0.5
                    else:
                        cond_prob = 1.0 - p

                    T[action][state_to_index((new_b, new_opp, new_poss))] += (
                        cond_prob * opp_prob
                    )
                    T[action][loss_s] += (1.0 - cond_prob) * opp_prob

            # player 1 moves
            for action in range(4, 8):
                new_b = (old_b[0], move(old_b[1], action - 4))
                new_poss = old_poss

                if out_of_bounds(new_b[1]):
                    T[action][loss_s] += 1.0 * opp_prob
                else:
                    # movement
                    if old_poss == 1:
                        cond_prob = 1.0 - 2.0 * p
                        # tackling
                        if new_b[1] == new_opp or (
                            new_b[1] == old_opp and old_b[1] == new_opp
                        ):
                            cond_prob *= 0.5
                    else:
                        cond_prob = 1.0 - p

                    T[action][state_to_index((new_b, new_opp, new_poss))] += (
                        cond_prob * opp_prob
                    )
                    T[action][loss_s] += (1.0 - cond_prob) * opp_prob

            # pass
            action = 8
            new_poss = 1 - old_poss
            cond_prob = q - 0.1 * chess_distance(old_b[0], old_b[1])
            if in_line(old_b[0], new_opp, old_b[1]):
                cond_prob *= 0.5

            T[action][state_to_index((old_b, new_opp, new_poss))] += (
                cond_prob * opp_prob
            )
            T[action][loss_s] += (1.0 - cond_prob) * opp_prob

            # shoot
            action = 9
            cond_prob = q - 0.2 * (3 - old_b[old_poss][0])
            if new_opp[0] == 3 and (new_opp[1] == 1 or new_opp[1] == 2):
                cond_prob *= 0.5

            T[action][win_s] += cond_prob * opp_prob
            T[action][loss_s] += (1.0 - cond_prob) * opp_prob

        R = np.zeros((A, S))
        for a in range(A):
            R[a][win_s] = 1.0

        for a in range(A):
            for s2 in range(S):
                if T[a][s2] > 1e-7:
                    print(
                        "transition %d %d %d %0.7f %0.7f"
                        % (s, a, s2, R[a][s2], T[a][s2])
                    )

    print("mdptype episodic")
    print("discount 1.0")


if __name__ == "__main__":
    import argparse

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

    main(args.opponent, args.p, args.q)
