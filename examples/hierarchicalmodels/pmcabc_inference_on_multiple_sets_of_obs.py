import numpy as np

"""An example showing how to implement a bayesian network in ABCpy"""
def infer_parameters():
    # The data corresponding to model_1 defined below
    grades_obs = [3.872486707973337, 4.6735380808674405, 3.9703538990858376, 4.11021272048805, 4.211048655421368, 4.154817956586653, 4.0046893064392695, 4.01891381384729, 4.123804757702919, 4.014941267301294, 3.888174595940634, 4.185275142948246, 4.55148774469135, 3.8954427675259016, 4.229264035335705, 3.839949451328312, 4.039402553532825, 4.128077814241238, 4.361488645531874, 4.086279074446419, 4.370801602256129, 3.7431697332475466, 4.459454162392378, 3.8873973643008255, 4.302566721487124, 4.05556051626865, 4.128817316703757, 3.8673704442215984, 4.2174459453805015, 4.202280254493361, 4.072851400451234, 3.795173229398952, 4.310702877332585, 4.376886328810306, 4.183704734748868, 4.332192463368128, 3.9071312388426587, 4.311681374107893, 3.55187913252144, 3.318878360783221, 4.187850500877817, 4.207923106081567, 4.190462065625179, 4.2341474252986036, 4.110228694304768, 4.1589891480847765, 4.0345604687633045, 4.090635481715123, 3.1384654393449294, 4.20375641386518, 4.150452690356067, 4.015304457401275, 3.9635442007388195, 4.075915739179875, 3.5702080541929284, 4.722333310410388, 3.9087618197155227, 4.3990088006390735, 3.968501165774181, 4.047603645360087, 4.109184340976979, 4.132424805281853, 4.444358334346812, 4.097211737683927, 4.288553086265748, 3.8668863066511303, 3.8837108501541007]

    # The prior information changing the class size and the teacher student ratio, depending on the yearly budget of the school 
    from abcpy.continuousmodels import Uniform, Normal
    school_budget = Uniform([[1], [10]], name = 'school_budget')

    # The average class size of a certain school
    class_size = Normal([[800*school_budget], [1]], name = 'class_size')

    # The number of teachers in the school
    no_teacher = Normal([[20*school_budget], [1]], name = 'no_teacher')

    # The grade a student would receive without any bias
    grade_without_additional_effects = Normal([[4.5], [0.25]], name = 'grade_without_additional_effects')

    # The grade a student of a certain school receives
    final_grade = grade_without_additional_effects - .001 * class_size + .02 * no_teacher

    # The data corresponding to model_2 defined below
    scholarship_obs = [2.7179657436207805, 2.124647285937229, 3.07193407853297, 2.335024761813643, 2.871893855192, 3.4332002458233837, 3.649996835818173, 3.50292335102711, 2.815638168018455, 2.3581613289315992, 2.2794821846395568, 2.8725835459926503, 3.5588573782815685, 2.26053126526137, 1.8998143530749971, 2.101110815311782, 2.3482974964831573, 2.2707679029919206, 2.4624550491079225, 2.867017757972507, 3.204249152084959, 2.4489542437714213, 1.875415915801106, 2.5604889644872433, 3.891985093269989, 2.7233633223405205, 2.2861070389383533, 2.9758813233490082, 3.1183403287267755, 2.911814060853062, 2.60896794303205, 3.5717098647480316, 3.3355752461779824, 1.99172284546858, 2.339937680892163, 2.9835630207301636, 2.1684912355975774, 3.014847335983034, 2.7844122961916202, 2.752119871525148, 2.1567428931391635, 2.5803629307680644, 2.7326646074552103, 2.559237193255186, 3.13478196958166, 2.388760269933492, 3.2822443541491815, 2.0114405441787437, 3.0380056368041073, 2.4889680313769724, 2.821660164621084, 3.343985964873723, 3.1866861970287808, 4.4535037154856045, 3.0026333138006027, 2.0675706089352612, 2.3835301730913185, 2.584208398359566, 3.288077633446465, 2.6955853384148183, 2.918315169739928, 3.2464814419322985, 2.1601516779909433, 3.231003347780546, 1.0893224045062178, 0.8032302688764734, 2.868438615047827]

    # A quantity that determines whether a student will receive a scholarship
    scholarship_without_additional_effects = Normal([[2], [0.5]], name = 'schol_without_additional_effects')

    # A quantity determining whether a student receives a scholarship, including his social teacher_student_ratio
    final_scholarship = scholarship_without_additional_effects + .03 * no_teacher

    # Define a summary statistics for final grade and final scholarship
    from abcpy.statistics import Identity
    statistics_calculator_final_grade = Identity(degree = 2, cross = False)
    statistics_calculator_final_scholarship = Identity(degree = 3, cross = False)

    # Define a distance measure for final grade and final scholarship
    from abcpy.distances import Euclidean
    distance_calculator_final_grade = Euclidean(statistics_calculator_final_grade)
    distance_calculator_final_scholarship = Euclidean(statistics_calculator_final_scholarship)

    # Define a backend
    from abcpy.backends import BackendDummy as Backend
    backend = Backend()

    # Define a perturbation kernel
    from abcpy.perturbationkernel import DefaultKernel
    kernel = DefaultKernel([school_budget, class_size, grade_without_additional_effects, \
                            no_teacher, scholarship_without_additional_effects])

    # Define sampling parameters
    T, n_sample, n_samples_per_param = 3, 250, 10
    eps_arr = np.array([.75])
    epsilon_percentile = 10

    # Define sampler
    from abcpy.inferences import PMCABC
    sampler = PMCABC([final_grade, final_scholarship], \
                     [distance_calculator_final_grade, distance_calculator_final_scholarship], backend, kernel)

    # Sample
    journal = sampler.sample([grades_obs, scholarship_obs], \
                             T, eps_arr, n_sample, n_samples_per_param, epsilon_percentile)


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
