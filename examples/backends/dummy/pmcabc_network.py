
# The prior information specifiying that depending on the school location, the score for class size and social background should be different
from abcpy.continuousmodels import Uniform, Normal
school_location_score = Uniform([[0.2,0.2],[0.3,0.3]])

# The school location affects both the class size and the social background
class_size_score = Normal([[school_location_score[0]], [0.1]])

background_score = Normal([[school_location_score[1]], [0.1]])

# Mean grade
grade = Normal([[4.5],[0.25]])

# The final model of the grade
model_1 = grade-class_size_score-background_score

#Probability that somebody gets scholarship
scholarship_score = Normal([[2],[0.5]])

# The final score, including the social background
model_2 = scholarship_score + 3*background_score

from abcpy.statistics import Identity
statistics_calculator = Identity(degree = 2, cross = False)

from abcpy.distances import DefaultJointDistance
distance_calculator = DefaultJointDistance(statistics_calculator)

from abcpy.backends import BackendDummy as Backend
backend = Backend()

from abcpy.perturbationkernels import MultivariateNormalKernel, MultiStudentTKernel
kernel_1 = MultivariateNormalKernel([school_location_score, scholarship_score])
kernel_2 = MultiStudentTKernel([class_size_score, background_score, grade])

from abcpy.perturbationkernels import JointPerturbationKernel
kernel = JointPerturbationKernel([kernel_1, kernel_2])

T, n_sample, n_samples_per_param = 3, 250, 10
eps_arr = np.array([.75])
epsilon_percentile = 10

from abcpy.inferences import PMCABC
sampler = PMCABC([model_1, model_2], distance_calculator, backend, kernel)
journal = sampler.sample([y_obs1, y_obs2], T, eps_arr, n_sample, n_samples_per_param, epsilon_percentile)

