"""Tests here the examples which do not require parallelization."""

import unittest


class ExampleApproxLhdTest(unittest.TestCase):
    def test_pmc(self):
        from examples.approx_lhd.pmc_hierarchical_models import infer_parameters
        journal = infer_parameters(steps=1, n_sample=50)
        test_result = journal.posterior_mean()["school_location"]
        expected_result = 0.2566394909510058
        self.assertAlmostEqual(test_result, expected_result)


class ExampleBackendsTest(unittest.TestCase):
    def test_dummy(self):
        from examples.backends.dummy.pmcabc_gaussian import infer_parameters
        journal = infer_parameters(steps=1, n_sample=50)
        test_result = journal.posterior_mean()["mu"]
        expected_result = 175.00683044068612
        self.assertAlmostEqual(test_result, expected_result)


class ExampleExtensionsModelsTest(unittest.TestCase):
    def test_cpp(self):
        from examples.extensions.models.gaussian_cpp.pmcabc_gaussian_model_simple import infer_parameters
        journal = infer_parameters(steps=1, n_sample=50)
        test_result = journal.posterior_mean()["mu"]
        expected_result = 173.74453347475725
        self.assertAlmostEqual(test_result, expected_result)

    def test_f90(self):
        from examples.extensions.models.gaussian_f90.pmcabc_gaussian_model_simple import infer_parameters
        journal = infer_parameters(steps=1, n_sample=50)
        test_result = journal.posterior_mean()["mu"]
        # note that the f90 example does not always yield the same result on some machines, even if it uses random seed
        expected_result = 173.84265330966315
        self.assertAlmostEqual(test_result, expected_result, delta=3)

    def test_python(self):
        from examples.extensions.models.gaussian_python.pmcabc_gaussian_model_simple import infer_parameters
        journal = infer_parameters(steps=1, n_sample=50)
        test_result = journal.posterior_mean()["mu"]
        expected_result = 175.00683044068612
        self.assertAlmostEqual(test_result, expected_result)

    def test_R(self):
        import os
        print(os.getcwd())
        from examples.extensions.models.gaussian_R.pmcabc_gaussian_model_simple import infer_parameters
        journal = infer_parameters(steps=1, n_sample=50)
        test_result = journal.posterior_mean()["mu"]
        expected_result = 173.4192372459506
        self.assertAlmostEqual(test_result, expected_result)


class ExampleExtensionsPerturbationKernelsTest(unittest.TestCase):
    def test_pmcabc_perturbation_kernel(self):
        from examples.extensions.perturbationkernels.pmcabc_perturbation_kernels import infer_parameters
        journal = infer_parameters(steps=1, n_sample=50)
        test_result = journal.posterior_mean()["schol_without_additional_effects"]
        expected_result = 1.9492397683665226
        self.assertAlmostEqual(test_result, expected_result)


class ExampleHierarchicalModelsTest(unittest.TestCase):
    def test_pmcabc(self):
        from examples.hierarchicalmodels.pmcabc_inference_on_multiple_sets_of_obs import infer_parameters
        journal = infer_parameters(steps=1, n_sample=50)
        test_result = journal.posterior_mean()["schol_without_additional_effects"]
        expected_result = 1.9492397683665226
        self.assertAlmostEqual(test_result, expected_result)


class ExampleModelSelectionTest(unittest.TestCase):
    def test_random_forest(self):
        from examples.modelselection.randomforest_modelselections import infer_model
        model, model_prob = infer_model()
        expected_result = 0.8704000000000001
        # this is not fully reproducible, there are some fluctuations in the estimated value
        self.assertAlmostEqual(model_prob[0], expected_result, delta=0.05)


class ExampleStatisticsLearningTest(unittest.TestCase):
    def test_pmcabc(self):
        from examples.statisticslearning.pmcabc_gaussian_statistics_learning import infer_parameters
        journal = infer_parameters(steps=1, n_sample=50)
        test_result = journal.posterior_mean()["mu"]
        expected_result = 172.52136853079725
        self.assertAlmostEqual(test_result, expected_result)


if __name__ == '__main__':
    unittest.main()
