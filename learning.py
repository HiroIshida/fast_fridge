import dill
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from pr2opt_common import *
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config
from task import ReachingTask
from task import OpeningTask
from task import Visualizer

from regexp import ExpansionAlgorithm
from regexp import RBF
from regexp import GridExpansionAlgorithm

np.random.seed(1)

#vis = Visualizer()
#np.random.seed(3)

robot_model = pr2_init()
joint_list = rarm_joint_list(robot_model)
av_start = get_robot_config(robot_model, joint_list, with_base=True)

trans = np.zeros(3)
rpy = np.zeros(3)
task3 = ReachingTask(robot_model, 12)
task3.load_sol_cache()
task3.reset_fridge_pose_from_handle_pose(trans, rpy)
pos_typical = task3.fridge.typical_object_position()
grid = task3.fridge.get_grid(N_grid=4)
n_points = grid.N_grid**3

logicals_filled = np.zeros(n_points, dtype=bool)
logicals_invalid = np.zeros(n_points, dtype=bool)
logicals_positive = np.zeros(n_points, dtype=bool)

def pick_point():
    idxes_negative = np.where(~logicals_filled)[0]
    i = np.random.randint(len(idxes_negative))
    idx = idxes_negative[i]
    return grid.pts[idx]

def generate_trajectory(pos_nominal):
    task3.setup(position=pos_nominal)

    dists = np.sum((grid.pts - np.atleast_2d(pos_nominal))**2, axis=1)
    idx_closest = np.argmin(dists)
    try:
        av_seq_sol = task3.solve()
    except:
        logicals_filled[idx_closest] = True
        logicals_invalid[idx_closest] = True
        return
    if av_seq_sol is None:
        logicals_filled[idx_closest] = True
        logicals_invalid[idx_closest] = True
        return

    gea = GridExpansionAlgorithm(grid, pos_nominal)

    def predicate(pos):
        is_inside = task3.fridge.is_inside(np.atleast_2d(pos))[0]
        if not is_inside:
            return False

        task3.setup(position=pos)
        result = task3.replanning(ignore_collision=False, bench_type="normal")
        if result is None:
            return False
        return result.nfev < 30
    gea.run(predicate, verbose=True)
    logicals_filled[gea.idxes_positive] = True
    logicals_positive[gea.idxes_positive] = True

while True:
    pt = pick_point()
    generate_trajectory(pt)
    if np.all(logicals_filled):
        break

"""
def find_region_using_gp():
    x_init = pos_typical
    kernel = RBF(0.1)
    ea = ExpansionAlgorithm(x_init, kernel=kernel, noise=0.1)

    def predicate(pos):
        is_inside = task3.fridge.is_inside(np.atleast_2d(pos))[0]
        if not is_inside:
            return False

        task3.setup(position=pos)
        result = task3.replanning(ignore_collision=False, bench_type="normal")
        if result is None:
            return False
        return result.nfev < 30
    ea.run(predicate, verbose=True)
    ea.plot3d()
    plt.show()

def construct_grid_data():
    pts = task3.fridge.grid_sample_from_inside(N=8)
    results = []
    for i in range(len(pts)):
        pos = pts[i]
        task3.setup(position=pos)
        opt_res = task3.replanning(ignore_collision=False, bench_type="normal")
        results.append(opt_res)

    with open("tmp.dill", "wb") as f:
        data = {"pts": pts, "results" :results}
        dill.dump(data, f)

    def is_properly_solved(result):
        if result is None:
            return False
        return result.nfev < 20
    logidx_valid = np.array([is_properly_solved(res) for res in results])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat3d = lambda X, c:  ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c)
    scat3d(pts[logidx_valid, :], "blue")
    scat3d(pts[~logidx_valid, :], "red")
    plt.show()
"""
