import dill
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open("traj_sampler_cache20.dill", 'rb') as f:
    model = dill.load(f)
    traj_list = model.traj_list
    grid = model.grid

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N = grid.N_grid
spacing = (grid.b_max - grid.b_min)/(N-1)
bitvec_inside_fridge = grid.logicals_inside
bitvec_whole_positive = np.zeros(N**3, dtype=bool)
scat3d = lambda X, c:  ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c)

pts_positive = grid.pts[bitvec_whole_positive]
#ax.scatter(pts_positive[:, 0], pts_positive[:, 1], pts_positive[:, 2])

vf_tuples = []
for traj in traj_list:
    clf = traj.classifier
    F, std = clf.predict(grid.pts, return_std=True)
    F[std > 0.8] = -100.0 # eliminate negative region where just too far from the samples
    bitvec_whole_positive = np.logical_or(bitvec_whole_positive, F > 0.0)
    F = F.reshape(N, N, N)
    F = np.swapaxes(F, 0, 1)
    try:
        verts, faces, _, _ = measure.marching_cubes_lewiner(F, 0, spacing=spacing)
        verts = verts + np.atleast_2d(grid.b_min)
        vf_tuples.append((verts, faces))
        #ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], color="yellow", alpha=0.3)
    except:
        pass

pts_positive = grid.pts[bitvec_whole_positive]
def myscat(P, **kwargs):
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], **kwargs)


bitvec_negative = np.logical_and(bitvec_inside_fridge, ~bitvec_whole_positive)
myscat(grid.pts[bitvec_whole_positive], c="blue", s=2)
myscat(grid.pts[bitvec_negative], c="red", s=1)

print(len(vf_tuples))
with_surf = False
if with_surf:
    for vf_tuple in vf_tuples:
        verts, faces =vf_tuple
        ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], color="yellow", alpha=0.7)

xlim, ylim, zlim = zip(grid.b_min, grid.b_max)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.set_size_inches(9.5, 7.2)
ax.view_init(elev=10., azim=-250)
filename = "feasible_set{0}.png".format("_withsurf" if with_surf else "")
plt.savefig(filename, format="png", dpi=300)
plt.show()
