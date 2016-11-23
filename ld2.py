import pyopencl as cl
import pyopencl.array as clarr
import pyopencl.clrandom as clrand
import numpy as np
from tqdm import tqdm
from mako.template import Template
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, ctx=None, moore=True, t=10, r=7, s=0, p=0):
        # Maximum payoff difference
        dpmax = max(t, r, s, p) - min(t, r, s, p)
        if ctx is None:
            ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(ctx)
        tpl = Template(open("ex1.cl").read())
        src = tpl.render(moore=moore, T=t, R=r, S=s, P=p, DPMax=dpmax)
        self.prog = cl.Program(ctx, str(src)).build()
        self.rand = clrand.PhiloxGenerator(ctx).uniform

    def _count_worlds_sums(self, before, coop, iterno, wait=None):
        """
        Count the number of cooperating cells in each world in parallel
        """
        wait_list = []
        if wait is not None:
            wait_list.append(wait)

        worlds, w, h = before.shape
        ls = 1 if worlds % 4 else 4

        return self.prog.sum_world(self.queue, (worlds,), (ls,),
                                   before.data, coop.data,
                                   np.int32(iterno * before.shape[0]),
                                   np.int32(w), np.int32(h),
                                   wait_for=wait_list)

    def _run_1_pass(self, before, after, rewards, coop, iterno, wait=None):
        """
        Run 1 pass of the game (each player plays against its lattice neighbors),
        for all worlds in parallel
        """
        assert before.shape == after.shape
        assert before.shape == rewards.shape

        ls = 1 if before.shape[0] % 2 else 2
        localsize = (ls,) * len(before.shape)
        counted = self._count_worlds_sums(before, coop, iterno, wait)
        evt = self.prog.play(self.queue, before.shape, localsize,
                             before.data, rewards.data,
                             wait_for=[counted])
        return self.prog.choose_best(self.queue, before.shape, localsize,
                                     before.data, rewards.data, after.data,
                                     wait_for=[evt])

    def _run_1_pass_r(self, before, after, rewards, coop, iterno, wait=None):
        """
        Run 1 pass of the game (each player plays against its lattice neighbors),
        for all worlds in parallel
        """
        assert before.shape == after.shape
        assert before.shape == rewards.shape

        localsize = (2,) * len(before.shape)
        counted = self._count_worlds_sums(before, coop, iterno, wait)
        evt = self.prog.play(self.queue, before.shape, localsize,
                             before.data, rewards.data,
                             wait_for=[counted])

        # Probability of mutation
        mut_p = self.rand(self.queue, before.shape, dtype=np.float32)
        # Neighbor choice
        choice_p = self.rand(self.queue, before.shape, dtype=np.float32)
        return self.prog.choose_replicator(self.queue, before.shape, localsize,
                                           mut_p.data, choice_p.data,
                                           before.data, rewards.data, after.data,
                                           wait_for=[evt])

    def run(self, n=50, iterations=100, worlds=100, p=0.5, replicator=False):
        """
        Perform all simulations in parallel
        """
        # Initial state: multiple worlds of size n,n
        players = (np.random.rand(worlds, n, n) < p).astype(np.int32)

        # Allocate/copy work buffers onto GPU
        before = clarr.to_device(self.queue, players)
        after = clarr.zeros_like(before)
        rewards = clarr.zeros(self.queue, players.shape, dtype=np.float32)
        coop = clarr.zeros(self.queue, (iterations, worlds), dtype=np.int32)

        w = None
        for i in tqdm(range(iterations)):
            if replicator:
                w = self._run_1_pass_r(before, after, rewards, coop, i, wait=w)
            else:
                w = self._run_1_pass(before, after, rewards, coop, i, wait=w)
            before, after = after, before
        if w:
            w.wait()
        return coop.get().T/float(n*n), before.get()


if __name__ == "__main__":
    # np.random.seed(42)

    for color, m in [('b', True), ('r', False)]:
        r = Runner(moore=m, t=10, r=7, s=3)
        coop, final = r.run(n=50, iterations=200, worlds=100, p=0.5, replicator=True)

        plt.plot(coop.T, c=color, alpha=0.1)
        plt.plot(coop.mean(axis=0), c=color, lw=2, label='Moore' if m else 'Von N.')
        plt.plot(coop.max(axis=0), c=color)
        plt.plot(coop.min(axis=0), c=color)

    plt.legend()
    plt.grid()
    plt.show()
