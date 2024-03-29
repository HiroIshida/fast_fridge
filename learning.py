import argparse
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

from regexp import RegionPopulationAlgorithm2
from regexp import InvalidSearchCenterPointException

np.random.seed(2)

class RegionEquippedTrajectory(object):
    def __init__(self, av_seq, feasible_set, classifier):
        self.av_seq = av_seq
        self.feasible_set = feasible_set
        self.classifier = classifier

class TrajectoryLibrary(object):
    def __init__(self, trajectory_list, grid):
        self.trajectory_list = trajectory_list

    def dump_dill(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    def find_trajectory(self, pos):
        # pos must be relative to the fridge model
        scores = np.array([traj.classifier.predict(np.atleast_2d(pos))[0]
            for traj in self.trajectory_list])
        if np.all(scores < 1e-5):
            print("no feasible trajectory found")
            return None
        traj_most_feasible = self.trajectory_list[np.argmax(scores)]
        return traj_most_feasible

class TrajetorySampler(object):

    def __init__(self, N_grid=8):
        robot_model = pr2_init()
        joint_list = rarm_joint_list(robot_model)
        av_start = get_robot_config(robot_model, joint_list, with_base=True)

        trans = np.zeros(3)
        rpy = np.zeros(3)
        task = ReachingTask(robot_model, 12)
        task.load_sol_cache()
        task.reset_fridge_pose(trans, rpy)
        grid = task.fridge.get_grid(N_grid=N_grid)

        self.task = task
        self.grid = grid
        self.rpa = RegionPopulationAlgorithm2(grid, 0.1)
        self.traj_list = []

        # set to None before update
        self.nominal_trajectory_cache = None

    @classmethod
    def default_cache_file_name(cls, N_grid):
        return "traj_sampler_cache{0}.dill".format(N_grid)

    @classmethod
    def from_dilled_data(cls, N_grid):
        """
        because object by just loading .dill file cannot use
        newly added methods
        """
        with open(cls.default_cache_file_name(N_grid), "rb") as f:
            data = dill.load(f)
        N_grid = data.grid.N_grid
        obj = cls(N_grid=N_grid)
        obj.task = data.task
        obj.grid = data.grid
        obj.rpa = data.rpa
        obj.traj_list = data.traj_list
        return obj

    def predicate_generator(self, pos_nominal):
        self.task.load_sol_cache()
        self.task.setup(position=pos_nominal)
        try:
            print("solving nominal trajectory...")
            av_seq_sol = self.task.solve()
            print("solved")
        except:
            raise InvalidSearchCenterPointException
        if av_seq_sol is None:
            raise InvalidSearchCenterPointException
        if not self.task.check_trajectory(n_mid=20):
            print("check trajectory failed")
            raise InvalidSearchCenterPointException

        self.nominal_trajectory_cache = av_seq_sol

        def predicate(pos):
            is_inside = self.task.fridge.is_inside(np.atleast_2d(pos))[0]
            if not is_inside:
                return False

            self.task.setup(position=pos)
            result = self.task.replanning(ignore_collision=False, bench_type="normal")
            if result is None:
                return False
            if not self.task.check_trajectory(n_mid=20):
                print("check trajectory failed")
                return False
            return result.nfev < 40
        return predicate

    def run(self):
        pos_init = self.task.fridge.typical_object_position()
        gea, cea = self.rpa.update(pos_init, predicate_generator=self.predicate_generator)

        model = cea.model
        pts_feasible = self.grid.pts[gea.idxes_positive] - np.atleast_2d(self.task.fridge.worldpos())
        traj = RegionEquippedTrajectory(self.nominal_trajectory_cache, pts_feasible, model)
        self.traj_list.append(traj)

        while True:
            with open(self.default_cache_file_name(self.grid.N_grid), "wb") as f:
                dill.dump(self, f)

            self.nominal_trajectory_cache = None
            if self.rpa.is_terminated():
                #self.rpa.show(showtype="strange")
                #plt.show()
                break
            pos_next = self.rpa.get_next_point()
            gea, cea = self.rpa.update(pos_next, predicate_generator=self.predicate_generator)

            if self.nominal_trajectory_cache is not None:
                pts_feasible = self.grid.pts[gea.idxes_positive] - np.atleast_2d(self.task.fridge.worldpos())
                model = cea.model
                traj = RegionEquippedTrajectory(self.nominal_trajectory_cache, pts_feasible, model)
                self.traj_list.append(traj)

    def dump_trajectory_library(self):
        return TrajectoryLibrary(self.traj_list, self.grid)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-n', type=int, default=4)
    args = parser.parse_args()

    N_grid = args.n
    ts = TrajetorySampler(N_grid=N_grid)
    ts.run()
    ts = TrajetorySampler.from_dilled_data(N_grid)

    traj_lib = ts.dump_trajectory_library()
    lib_file_name = "traj_lib{0}.dill".format(N_grid)
    traj_lib.dump_dill(lib_file_name)
    with open(lib_file_name, 'rb') as f:
        traj_lib_loaded = dill.load(f)
