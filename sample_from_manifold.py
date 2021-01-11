import numpy as np
from scipy.linalg import null_space

class ManifoldSampler(object):
    def __init__(self, x_init, func, b_min, b_max, feasible_predicate=None):
        self.itr = 0
        self.func = func
        self.b_min = b_min
        self.b_max = b_max
        self.n_dim = len(b_min)
        self.feasible_predicate = feasible_predicate

        self.eps = 0.2

        self.X = np.zeros((100000, self.n_dim))
        self.X[0] = x_init

    def _sample_from_box(self):
        w = self.b_max - self.b_min
        x = np.random.rand(self.n_dim) * w + self.b_min
        return x

    def _project_to_manifold(self, x):
        th = 1e-4
        sr_weight = 1.0
        while True:
            f_val, jac = self.func(x)
            if np.all(np.abs(f_val) < th):
                break
            J_sharp = jac.T.dot(np.linalg.inv(jac.dot(jac.T) + sr_weight))
            x += (- f_val).dot(J_sharp.T)
        return x

    def sample(self):
        # step1 sample from box
        x_rand = self._sample_from_box()

        # step2 find nearest neighbore sample
        diff_sqrt = np.sum((self.X[:self.itr+1] - x_rand[None, :])**2, axis=1)
        idx_nearest = np.argmin(diff_sqrt)
        x_nearest = self.X[idx_nearest]

        # step3 project difference_vector onto the nullspace
        diff_vec = x_rand - x_nearest
        _, jac = self.func(x_nearest)
        N = null_space(jac)
        diff_vec_nspace = N.dot(N.T).dot(diff_vec)
        diff_norm = np.linalg.norm(diff_vec_nspace)
        if diff_norm < self.eps:
            x_new_nspace = x_nearest + diff_vec_nspace
        else:
            x_new_nspace = x_nearest + diff_vec_nspace * self.eps / diff_norm

        # step4 project onto the manifold
        x_new = self._project_to_manifold(x_new_nspace)
        if self.feasible_predicate is None:
            isvalid = True
        else:
            isvalid = self.feasible_predicate(x_new)

        # post process
        if isvalid:
            self.X[self.itr+1] = x_new
            self.itr += 1

if __name__=='__main__':
    def func(x):
        f_val = np.array([sum(x**2) - 1])
        jac = (2 * x).reshape(1, 3)
        return f_val, jac

    def predicate(x):
        return x[0] > 0

    b_min = np.ones(3) * -2
    b_max = np.ones(3) * 2
    x_init = np.array([1, 0, 0])
    m = ManifoldSampler(x_init, func, b_min=b_min, b_max=b_max, feasible_predicate=predicate)
    for i in range(10000):
        m.sample()

    from mpl_toolkits.mplot3d import Axes3D 
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(m.X[:, 0], m.X[:, 1], m.X[:, 2])
    plt.show()

