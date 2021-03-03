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

from regexp import RegionPopulationAlgorithm
from regexp import InvalidSearchCenterPointException

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
        grid = task.fridge.get_grid(N_grid=N_grid)

        self.task = task
        self.rpa = RegionPopulationAlgorithm(grid)

    def predicate_generator(self, pos_nominal):
        self.task.setup(position=pos_nominal)
        try:
            av_seq_sol = self.task.solve()
        except:
            raise InvalidSearchCenterPointException
        if av_seq_sol is None:
            raise InvalidSearchCenterPointException

        def predicate(pos):
            is_inside = self.task.fridge.is_inside(np.atleast_2d(pos))[0]
            if not is_inside:
                return False

            self.task.setup(position=pos)
            result = self.task.replanning(ignore_collision=False, bench_type="normal")
            if result is None:
                return False
            return result.nfev < 30
        return predicate

    def run(self):
        pos_init = self.task.fridge.typical_object_position()
        self.rpa.update(pos_init, predicate_generator=self.predicate_generator)

        while True:
            if self.rpa.is_terminated():
                break
            pos_next = self.rpa.get_next_point()
            self.rpa.update(pos_next, predicate_generator=self.predicate_generator)

ts = TrajetorySampler(N_grid=5)
ts.run()
