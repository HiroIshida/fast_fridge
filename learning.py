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

class TrajetorySampler(object):
    def __init__(self, N_grid=8):
        robot_model = pr2_init()
        joint_list = rarm_joint_list(robot_model)
        av_start = get_robot_config(robot_model, joint_list, with_base=True)

        trans = np.zeros(3)
        rpy = np.zeros(3)
        task = ReachingTask(robot_model, 12)
        task.load_sol_cache()
        task.reset_fridge_pose_from_handle_pose(trans, rpy)
        pos_typical = task.fridge.typical_object_position()
        grid = task.fridge.get_grid(N_grid=8)
        n_points = grid.N_grid**3

        self.task = task
        self.grid = grid
        self.logicals_filled = np.zeros(n_points, dtype=bool)

    def pick_next_point(self):
        idxes_negative = np.where(~self.logicals_filled)[0]
        i = np.random.randint(len(idxes_negative))
        idx = idxes_negative[i]
        return self.grid.pts[idx]

    def compute_feasible_region(self, pos_nominal):
        self.task.setup(position=pos_nominal)
        dists = np.sum((self.grid.pts - np.atleast_2d(pos_nominal))**2, axis=1)
        idx_closest = np.argmin(dists)
        try:
            av_seq_sol = self.task.solve()
        except:
            self.logicals_filled[idx_closest] = True
            return
        if av_seq_sol is None:
            self.logicals_filled[idx_closest] = True
            return

        gea = GridExpansionAlgorithm(self.grid, pos_nominal)

        def predicate(pos):
            is_inside = self.task.fridge.is_inside(np.atleast_2d(pos))[0]
            if not is_inside:
                return False

            self.task.setup(position=pos)
            result = self.task.replanning(ignore_collision=False, bench_type="normal")
            if result is None:
                return False
            return result.nfev < 30
        gea.run(predicate, verbose=True)
        self.logicals_filled[gea.idxes_positive] = True

    def run(self):
        while True:
            pt = self.pick_next_point()
            self.compute_feasible_region(pt)
            if np.all(self.logicals_filled):
                break

ts = TrajetorySampler(N_grid=5)
ts.run()
