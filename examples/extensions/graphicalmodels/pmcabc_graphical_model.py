import numpy as np

"""An example showing how to implement a bayesian network in ABCpy"""
def infer_parameters():
    # The data corresponding to model_1 defined below
    grades_obs = [3.396010144702873, 4.028241323112599, 3.62536728374886, 4.480429613325388, 3.787787804731597, 3.5899690679086107, 4.1736660809715405, 4.1593985002639275, 4.311185599797756, 4.049607307458736, 4.154878620607175, 4.429684988710977, 3.7533184335061813, 4.127065260139277, 4.521082883155887, 4.294274689695604, 3.94464672887112, 4.157832218067926, 4.2411935748991, 4.178945229216105, 3.785717756612455, 4.900210170980145, 3.8400182404620633, 4.38150456491072, 4.177538374339252, 4.015592937396663, 4.140537380463723, 3.852151227951589, 4.213718852494517, 3.9974576730715636, 4.119116004215904, 3.889664747339427, 4.27617521495367, 3.705033455404327, 3.8546905507341243, 3.61034397123066, 3.951050063304417, 4.109516892585332, 4.251958297243783, 4.377282299313978, 4.182230067895948, 3.7016957956246976, 3.9867272122758415, 4.278252460995561, 3.6988271248197466, 3.946398833830572, 4.1906415617863235, 4.004123105115532, 4.514888610733722, 4.066627046665651, 4.070266390101292, 4.238018975996794, 3.8252827974408246, 4.112394933441697, 4.093704899981078, 3.565044873345844, 4.0258611357597385, 4.363343385761771, 4.08570600900867, 4.077273623002418, 3.8350649704413153, 4.031198987911091, 3.8605569436588647, 4.337029660312197, 3.9892864794157643, 3.943365738564718, 4.867738846698944]

    # The prior information changing the class size and social background, depending on school location
    from abcpy.continuousmodels import Uniform, Normal
    school_location = Uniform([[0.2], [0.3]], )

    # The average class size of a certain school
    class_size = Normal([[school_location], [0.1]], )

    # The social background from which a student originates
    background = Normal([[school_location], [0.1]], )

    # The grade a student would receive without any bias
    grade_without_additional_effects = Normal([[4.5], [0.25]], )

    # The grade a student of a certain school receives
    final_grade = grade_without_additional_effects - class_size - background

    # Define a summary statistics
    from abcpy.statistics import Identity
    statistics_calculator = Identity(degree = 2, cross = False)

    # Define a distance measure
    from abcpy.distances import LogReg
    distance_calculator = LogReg(statistics_calculator)

    # Define a backend
    from abcpy.backends import BackendDummy as Backend
    backend = Backend()

    # Defien a perturbation kernel
    from abcpy.perturbationkernel import DefaultKernel
    kernel = DefaultKernel([school_location, class_size, grade_without_additional_effects, background])

    # Define sampling parameters
    T, n_sample, n_samples_per_param = 3, 250, 10
    eps_arr = np.array([.75])
    epsilon_percentile = 10

    # Define sampler
    from abcpy.inferences import PMCABC
    sampler = PMCABC([final_grade], [distance_calculator], backend, kernel)

    # Sample
    journal = sampler.sample([grades_obs], T, eps_arr, n_sample, n_samples_per_param, epsilon_percentile)

    return journal


def analyse_journal(journal):
    # output parameters and weights
    print(journal.get_stored_output_values())
    print(journal.weights)

    # do post analysis
    print(journal.posterior_mean())
    print(journal.posterior_cov())
    print(journal.posterior_histogram())

    # print configuration
    print(journal.configuration)

    # save and load journal
    journal.save("experiments.jnl")

    from abcpy.output import Journal
    new_journal = Journal.fromFile('experiments.jnl')

if __name__  == "__main__":
    journal = infer_parameters()
    analyse_journal(journal)
