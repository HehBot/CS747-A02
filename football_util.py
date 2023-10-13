import numpy as np

"""
01(0,3) 02(1,3) 03(2,3) 04(3,3)
05(0,2) 06(1,2) 07(2,2) 08(3,2)
09(0,1) 10(1,1) 11(2,1) 12(3,1)
13(0,0) 14(1,0) 15(2,0) 16(3,0)
"""


def index_to_coords(i):
    return ((i - 1) % 4, 3 - ((i - 1) // 4))


def coords_to_index(pos):
    return 4 * (3 - pos[1]) + pos[0] + 1


S = 2 + 16 * 16 * 16
A = 10
A_opp = 4


def state_to_index(s):
    if s == "loss":
        return S - 2
    elif s == "win":
        return S - 1
    b, opp, poss = s
    b0 = coords_to_index(b[0]) - 1
    b1 = coords_to_index(b[1]) - 1
    opp = coords_to_index(opp) - 1

    if b0 > b1:
        b0, b1 = b1, b0
        poss = 1 - poss
    elif b0 == b1:
        poss = 0

    return (16 * 16) * opp + b1 * b1 + 2 * b0 + poss


states = [None for s in range(S)]
states[state_to_index("loss")] = "loss"
states[state_to_index("win")] = "win"
for opp in range(1, 16 + 1):
    oppc = index_to_coords(opp)
    for b1 in range(1, 16 + 1):
        b1c = index_to_coords(b1)
        for b0 in range(1, b1):
            b0c = index_to_coords(b0)
            for poss in range(2):
                state = ((b0c, b1c), oppc, poss)
                states[state_to_index(state)] = state
        state = ((b1c, b1c), oppc, 0)
        states[state_to_index(state)] = state


def parse_opponent(opponent_file):
    opp_state_action_probs = np.zeros((S, A_opp))
    with open(opponent_file, "r") as f:
        f.readline()
        for l in f:
            l = l.strip().split()
            b0_index = int(l[0][:2])
            b1_index = int(l[0][2:4])
            b0 = index_to_coords(b0_index)
            b1 = index_to_coords(b1_index)
            opp = index_to_coords(int(l[0][4:6]))
            poss = int(l[0][6]) - 1
            if b0_index <= b1_index:
                state = ((b0, b1), opp, poss)
                s = state_to_index(state)
                for a in range(A_opp):
                    opp_state_action_probs[s][a] = float(l[a + 1])
    return opp_state_action_probs
