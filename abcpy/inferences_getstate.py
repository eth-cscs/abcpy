import numpy as np
from abcpy.output import Journal
from scipy import optimize

class RejectionABC:
    def __init__(self, observations, model, distance, backend, epsilon, n_samples, n_samples_per_param, seed=None):
        self.model = model
        self.dist_calc = distance
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.epsilon = epsilon
        self.observations_bds = backend.broadcast(observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['backend']
        return state

    def sample(self, observations, n_samples, n_samples_per_param, epsilon, full_output=0):
        journal = Journal(full_output)
        journal.configuration["n_samples"] = n_samples
        journal.configuration["n_samples_per_param"] = n_samples_per_param
        journal.configuration["epsilon"] = epsilon

        accepted_parameters = None

        # Initialize variables that need to be available remotely

        # main Rejection ABC algorithm
        seed_arr = self.rng.randint(1, n_samples * n_samples, size=n_samples, dtype=np.int32)
        seed_pds = self.backend.parallelize(seed_arr)

        accepted_parameters_pds = self.backend.map(self._sample_parameter, seed_pds)
        accepted_parameters = self.backend.collect(accepted_parameters_pds)
        accepted_parameters = np.array(accepted_parameters)

        journal.add_parameters(accepted_parameters)
        journal.add_weights(np.ones((n_samples, 1)))

        return journal

    def _sample_parameter(self, seed):
        distance = self.dist_calc.dist_max()
        self.model.prior.reseed(seed)

        while distance > self.epsilon:
            # Accept new parameter value if the distance is less than epsilon
            self.model.sample_from_prior()
            y_sim = self.model.simulate(self.n_samples_per_param)
            distance = self.dist_calc.distance(self.observations_bds.value(), y_sim)

        return self.model.get_parameters()

