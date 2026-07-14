from Metropolis import MetropolisSolver
from Gibbs import GibbsSolver
from Lifted import LiftedSolver


class AnnealingMixin:
    def __init__(self, *args, beta_max=10.0, cooling_rate=1.0005, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_0 = self.beta
        self.beta_max = beta_max
        self.cooling_rate = cooling_rate
        self.name = "annealed_" + self.name

    def _cool(self):
        if self.beta < self.beta_max:
            self.beta = min(self.beta * self.cooling_rate, self.beta_max)

    def solve_single(self):
        delta = super().solve_single()
        self._cool()
        return delta


class AnnealedMetropolis(AnnealingMixin, MetropolisSolver):
    pass


class AnnealedGibbs(AnnealingMixin, GibbsSolver):
    pass


class AnnealedLifted(AnnealingMixin, LiftedSolver):
    pass
