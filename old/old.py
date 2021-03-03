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
