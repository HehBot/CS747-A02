from football_util import state_to_index, index_to_coords
import matplotlib.pyplot as plt
import subprocess


def graph1(req_s, outfilename, outfileformat, opponent_file, p_list, q):
    args_planner = ("python", "planner.py")
    goals = []
    processes = []
    for p in p_list:
        args = (
            "python",
            "encoder.py",
            "--opponent",
            opponent_file,
            "--p",
            str(p),
            "--q",
            str(q),
        )
        enc = subprocess.Popen(args, stdout=subprocess.PIPE)
        processes.append(enc)
    for i in range(len(p_list)):
        proc = processes[i]
        vp = subprocess.check_output(args_planner, stdin=proc.stdout)
        proc.wait()
        ans = vp.decode().strip().split("\n")[req_s]
        ans = float(ans.split()[0])
        goals.append(ans)
        print(p_list[i], q, ans)

    fig, ax = plt.subplots()
    ax.plot(p_list, goals, label=("$q=%f$" % (q)))
    ax.set_title("$\mathbb{E}[G]$ vs $p$ for $q=%f$" % (q))
    ax.set_xlabel("$p$")
    ax.set_ylabel("$\mathbb{E}[G]$")
    ax.legend()
    fig.savefig(outfilename, format=outfileformat)


def graph2(req_s, outfilename, outfileformat, opponent_file, p, q_list):
    args_planner = ("python", "planner.py")
    goals = []
    processes = []
    for q in q_list:
        args = (
            "python",
            "encoder.py",
            "--opponent",
            opponent_file,
            "--p",
            str(p),
            "--q",
            str(q),
        )
        enc = subprocess.Popen(args, stdout=subprocess.PIPE)
        processes.append(enc)
    for i in range(len(q_list)):
        proc = processes[i]
        vp = subprocess.check_output(args_planner, stdin=proc.stdout)
        proc.wait()
        ans = vp.decode().strip().split("\n")[req_s]
        ans = float(ans.split()[0])
        goals.append(ans)
        print(p, q_list[i], ans)

    fig, ax = plt.subplots()
    ax.plot(q_list, goals, label=("$p=%f$" % (p)))
    ax.set_title("$\mathbb{E}[G]$ vs $q$ for $p=%f$" % (p))
    ax.set_xlabel("$q$")
    ax.set_ylabel("$\mathbb{E}[G]$")
    ax.legend()
    fig.savefig(outfilename, format=outfileformat)


if __name__ == "__main__":
    plt.rc("font", family="serif")
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsfonts}")

    req_state = ((index_to_coords(5), index_to_coords(9)), index_to_coords(8), 0)
    req_s = state_to_index(req_state)

    graph1(
        req_s,
        "graph1.svg",
        "svg",
        "data/football/test-1.txt",
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        0.7,
    )
    graph2(
        req_s,
        "graph2.svg",
        "svg",
        "data/football/test-1.txt",
        0.3,
        [0.6, 0.7, 0.8, 0.9, 1.0],
    )
