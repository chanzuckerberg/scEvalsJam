from pyro.distributions import RelaxedBernoulliStraightThrough
from torch.distributions import Bernoulli


class GumbelSoftmaxBernoulliStraightThrough(RelaxedBernoulliStraightThrough):
    # Distribution defined on {0, 1}, where p(x=1) = p and p(x=0) = 1-p,
    # and samples are generated using Gumbel Softmax to allow for
    # differentiable samples
    # TODO: redefine sample method to give hard samples? rsample already does,
    # which is the only method we use in guides
    def log_prob(self, value):
        return Bernoulli(probs=self.probs).log_prob(value)

    @property
    def mode(self):
        mode = (self.probs > 0.5).to(self.probs)
        return mode
