# Code for these tests was suggested by ChatGPT and Copilot, and modified by PV.
# The code in GaussianMixtureModel is my own.

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from DL_modules.custom_layers import GaussianMixtureModel


def test_prior_predictive():
    # Parameters for the test
    locations = [0.0, 5.0]
    variances = [1.0, 2.0]
    weights = [0.6, 0.4]
    test_point = 1.0
    measurement_variance = 0.3
    variances_extra = 0.5

    # Expected value computed manually
    expected_probability = weights[0] * norm.pdf(
        test_point, loc=locations[0], scale=np.sqrt(variances[0] + measurement_variance + variances_extra)
    ) + weights[1] * norm.pdf(
        test_point, loc=locations[1], scale=np.sqrt(variances[1] + measurement_variance + variances_extra)
    )

    # Assuming your class is named GaussianMixtureModel and has a .probability(x) method
    gmm = GaussianMixtureModel(
        locations=torch.tensor([[0.0], [5.0]]), variances=torch.tensor([[1.0], [2.0]]), weights=torch.tensor(weights)
    )
    log_prior_predictive = gmm.compute_log_prior_predictive(
        measurements=torch.tensor([[[test_point]]]),
        measurement_variances=torch.tensor([measurement_variance]),
        variances_extra=torch.tensor([variances_extra]),
    )

    assert np.isclose(torch.pow(10, log_prior_predictive), expected_probability, rtol=1e-7), (
        "Mismatch in GMM prior predictive probability"
    )


def test_posterior_predictive():
    # Define 2-component 1D GMM
    means = [0.0, 3.0]
    variances = [1.0, 1.0]
    weights = [0.5, 0.5]
    gmm = GaussianMixtureModel(
        locations=torch.tensor([[0.0], [3.0]]), variances=torch.tensor([1.0]), weights=torch.tensor(weights)
    )

    meas_var = 0.2  # measurement variance
    measvar_extra = 1

    # Simulated first measurement
    x_true = 2.7
    y1 = np.random.normal(loc=x_true, scale=np.sqrt(meas_var))

    # Compute responsibilities and component conditionals
    responsibilities = []
    component_dists = []

    for mu, var, w in zip(means, variances, weights):
        post_var = 1 / (1 / var + 1 / meas_var)
        post_mean = post_var * (mu / var + y1 / meas_var)

        marginal = norm.pdf(y1, loc=mu, scale=np.sqrt(var + meas_var))
        responsibilities.append(w * marginal)
        component_dists.append((post_mean, post_var))

    responsibilities = np.array(responsibilities)
    responsibilities /= responsibilities.sum()

    # Mixture predictive distribution: each component has predictive variance = postvar + measvar
    def logpy2_given_y1(y2):
        terms = []
        for w, (m, v) in zip(responsibilities, component_dists):
            pred_std = np.sqrt(v + meas_var + measvar_extra)
            terms.append(np.log(w) + norm.logpdf(y2, loc=m, scale=pred_std))
        return np.logaddexp.reduce(terms)

    # Simulate a second measurement
    y2 = np.random.normal(loc=x_true, scale=np.sqrt(meas_var))
    log_prob = logpy2_given_y1(y2)

    log10_post_predictive = gmm.compute_log_posterior_predictive(
        measurements=torch.tensor([[[y2]]]),
        given=torch.tensor([[[y1]]]),
        within_variances=torch.tensor([meas_var]),
        measurement_variances_extra=torch.tensor([measvar_extra]),
    )
    expected = np.exp(log_prob)
    assert np.isclose(torch.pow(10, log10_post_predictive), expected, rtol=1e-5), (
        "Mismatch in GMM posterior predictive probability"
    )

    gmm_2d = GaussianMixtureModel(
        locations=torch.tensor([[0.0, 0.0], [3.0, 3.0]]),
        variances=torch.tensor([1.0, 1.0]),
        weights=torch.tensor(weights),
    )
    log10_post_predictive = gmm_2d.compute_log_posterior_predictive(
        measurements=torch.tensor([[[y2], [y2]]]),
        given=torch.tensor([[[y1], [y1]]]),
        within_variances=torch.tensor([meas_var, meas_var]),
        measurement_variances_extra=torch.tensor([measvar_extra, measvar_extra]),
    )
    result = torch.pow(10, log10_post_predictive)
    assert np.isclose(result, np.array([expected, expected]), rtol=1e-6).all(), (
        "Mismatch in GMM posterior predictive probability"
    )


def test_posterior_predictive_batch():
    np.random.seed(42)
    n_locations = 2
    n_features = 3

    locs = np.random.randn(n_locations, n_features)
    variances = np.abs(np.random.rand(n_locations, n_features)) + 0.1
    weights = np.abs(np.random.rand(n_locations)) + 0.1
    weights /= weights.sum()

    rep1 = 4
    rep2 = 5
    within_var = 0.5 * np.abs(np.random.rand(n_features))
    measvar_extra = 0.3 * np.abs(np.random.rand(n_features))

    def samplefromgmm():
        k = np.random.choice(n_locations, p=weights)
        return np.random.normal(locs[k], np.sqrt(variances[k]))

    measurement1 = []
    measurement2 = []
    mean1 = []
    mean2 = []

    for _ in range(2):  # batch size = 2
        mu = samplefromgmm()

        meas1 = np.random.normal(mu, np.sqrt(within_var + measvar_extra), size=(rep1, n_features))
        meas2 = np.random.normal(mu, np.sqrt(within_var), size=(rep2, n_features))

        measurement1.append(meas1.transpose())
        measurement2.append(meas2.transpose())

        mean1.append(meas1.mean(axis=0))
        mean2.append(meas2.mean(axis=0))

    mean1 = np.stack(mean1)
    mean2 = np.stack(mean2)

    gmm = GaussianMixtureModel(
        locations=torch.tensor(locs), variances=torch.tensor(variances), weights=torch.tensor(weights)
    )
    within_var_for_gmm = torch.tensor(within_var)
    measvar_extra_for_gmm = torch.tensor(measvar_extra)
    log_probs = gmm.compute_log_posterior_predictive(
        torch.tensor(measurement1), torch.tensor(measurement2), within_var_for_gmm, measvar_extra_for_gmm
    )

    def reference_logprob_conditional(mean1, mean2):
        post_means = []
        post_variances = []

        for k in range(n_locations):
            postmeank = []
            postvark = []

            for d in range(n_features):
                prior_mean = locs[k, d]
                prior_var = variances[k, d]
                meas_var = within_var[d] / rep2

                # Posterior of μ given mean2[d]
                precisionpost = 1 / prior_var + 1 / meas_var
                var_post = 1 / precisionpost
                mean_post = (prior_mean / prior_var + mean2[d] / meas_var) * var_post

                postmeank.append(mean_post)
                postvark.append(var_post)

            post_means.append(np.array(postmeank))
            post_variances.append(np.array(postvark))

        # Initialize list to hold log-likelihoods per feature
        log_liks_per_feature = [[] for _ in range(n_features)]

        # Compute log-likelihood of mean1 under posterior mixture component-wise per feature
        for d in range(n_features):
            post_logweights = []

            for k in range(n_locations):
                prior_mean = locs[k, d]
                prior_var = variances[k, d]
                var = post_variances[k][d] + within_var[d] / rep1 + measvar_extra[d]

                # Calculate log likelihood of mean2 under prior per feature for each component
                loglik = np.log(weights[k]) + norm.logpdf(
                    mean2[d], loc=prior_mean, scale=np.sqrt(prior_var + within_var[d] / rep2)
                )
                post_logweights.append(loglik)

                # Log probability component for individual feature
                logp_feature = loglik + norm.logpdf(mean1[d], loc=post_means[k][d], scale=np.sqrt(var))
                log_liks_per_feature[d].append(logp_feature)

            # Normalize posterior weights for the feature
            logweightspost_sum = logsumexp(post_logweights)

            # Adjust log probabilities with normalized weights
            log_liks_per_feature[d] = log_liks_per_feature[d] - logweightspost_sum

        # Compute logsumexp for each feature to get final log likelihoods
        log_liks = [logsumexp(log_l) for log_l in log_liks_per_feature]

        return log_liks

    for i in range(2):
        ref = reference_logprob_conditional(mean1[i], mean2[i])
        ref = np.exp(ref)
        assert np.isclose(torch.pow(10, log_probs[i]), ref, atol=1e-6).all()


def test_posterior_predictive_specific_source():
    def batchconditionalprob(batch_data1, batch_data2, batch_var1, batch_var2, measurement_variances_extra):
        """Compute conditional probability densities for each pair of replicated datasets in batch.

        Parameters:
        - batch_data1: list of np.arrays, each shape (n1, d)
        - batch_data2: list of np.arrays, each shape (n2, d)
        - batch_var1: list of np.arrays, each shape (d,)
        - batch_var2: list of np.arrays, each shape (d,)

        Returns:
        - results: list of dictionaries with mean1, mean2, and conditional probabilities
        """
        results = []
        var1 = batch_var1
        var2 = batch_var2
        e = measurement_variances_extra

        for i in range(len(batch_data1)):
            data1 = batch_data1[i]
            data2 = batch_data2[i]

            n1, n2 = data1.shape[0], data2.shape[0]
            mean1 = np.mean(data1, axis=0)
            mean2 = np.mean(data2, axis=0)

            predictive_mean = mean2
            predictive_var = var1 / n1 + var2 / n2 + e
            predictive_std = np.sqrt(predictive_var)

            log_probdensity = np.log10(norm(loc=predictive_mean, scale=predictive_std).pdf(mean1))

            results.append(log_probdensity)

        return results

    # Example: Batch of two variable sets(each with 3 variables)
    batch_data1 = [
        np.array([[1.0, 2.1, 3.2], [1.2, 2.0, 3.1], [0.9, 1.9, 3.3], [1.1, 2.2, 3.0]]),
        np.array([[0.5, 1.0, 2.0], [0.6, 1.1, 2.2], [0.4, 0.9, 1.9], [0.55, 1.05, 2.1]]),
    ]

    batch_data2 = [
        np.array([[0.8, 2.0, 2.9], [1.0, 1.8, 3.0], [1.1, 2.2, 3.2], [1.2, 2.1, 3.1], [0.9, 2.0, 3.0]]),
        np.array([[0.45, 1.1, 1.9], [0.5, 1.0, 2.0], [0.6, 1.2, 2.1], [0.55, 1.0, 2.2], [0.4, 0.95, 1.95]]),
    ]

    batch_var = np.array([0.01, 0.02, 0.015])

    measurement_variances_extra = np.array([0.5, 0.5, 0.5])

    expected = batchconditionalprob(batch_data1, batch_data2, batch_var, batch_var, measurement_variances_extra)
    gmm = GaussianMixtureModel(torch.tensor([[0]]), torch.tensor([0]), torch.tensor([1]))
    batch_data_1_torch = torch.tensor(batch_data1).permute(0, 2, 1)
    batch_data_2_torch = torch.tensor(batch_data2).permute(0, 2, 1)
    batch_var_torch = torch.tensor(batch_var)
    measurement_variances_extra_torch = torch.tensor(measurement_variances_extra)
    test_results = gmm.compute_log_posterior_predictive_specific_source(
        batch_data_1_torch, batch_data_2_torch, batch_var_torch, measurement_variances_extra_torch
    )
    assert np.isclose(test_results, expected, atol=1e-6).all()


def test_compute_log_probability_given_known_locations():
    # Known,population means and variances for the 3 variables
    population_means = np.array([[10.0, 20.0, 30.0], [5, 3, 4]])
    measurement_variances = np.array([1.0, 4.0, 9.0])

    # Simulate 5 replicates for each variable in 2 batches
    np.random.seed(0)  # for reproducibility
    batches = []
    for i in range(2):  # 2 batches
        replicates = np.array(
            [
                np.random.normal(loc=mu, scale=np.sqrt(var), size=5)
                for mu, var in zip(population_means[i], measurement_variances)
            ]
        )
        batches.append(replicates)

    # Compute sample means for each variable in each batch
    sample_means = [np.mean(batch, axis=1) for batch in batches]

    # Compute likelihoods of observing these sample means
    llikelihoods = []
    for i, mean_vector in enumerate(sample_means):
        llikelihood = norm.logpdf(mean_vector, loc=population_means[i], scale=np.sqrt(measurement_variances / 5))
        llikelihoods.append(llikelihood)
    expected = llikelihoods / np.log(10)
    gmm = GaussianMixtureModel(torch.tensor([[0]]), torch.tensor([0]), torch.tensor([1]))
    measurements_torch = torch.tensor(batches)
    measurement_variances_torch = torch.tensor(measurement_variances)
    locations_torch = torch.tensor(population_means)
    test_result = gmm.compute_log_probabilities_given_known_locations(
        measurements_torch,
        locations_torch,
        measurement_variances_torch,
    )
    assert np.isclose(test_result, expected, atol=1e-6).all()


def test_all_public_methods_no_errors():
    n_samples = 1000
    n_replicates_measurements = 100
    n_replicates_given = 200

    locations = torch.tensor([[0.0], [1.0], [2.0]])
    variances = torch.tensor([0.3])
    measurement_variances = torch.tensor([[0.5]])
    extra_variances = torch.tensor([[1.0]])
    weights = torch.tensor([90.0, 9.0, 1.0])
    gmm = GaussianMixtureModel(locations, variances, weights)
    indices_locations = gmm.sample_location_indices(n_samples)
    locations_sampled = gmm.get_locations(indices_locations)
    samples_from_gmm = gmm.sample_from_gmm_given_location_indices(indices_locations)
    measurements = gmm.draw_measurements(
        locations_sampled, measurement_variances.repeat(n_samples, 1), n_replicates=n_replicates_measurements
    )
    given = gmm.draw_measurements(
        locations_sampled, measurement_variances.repeat(n_samples, 1), n_replicates=n_replicates_given
    )
    log_probability_given_known_locations = gmm.compute_log_probabilities_given_known_locations(
        measurements, locations_sampled, measurement_variances
    )
    log_prior_predictive = gmm.compute_log_prior_predictive(measurements, measurement_variances, extra_variances)
    log_posterior_predictive = gmm.compute_log_posterior_predictive(
        measurements, given, measurement_variances, extra_variances
    )
    log_posterior_predictive_specific_source = gmm.compute_log_posterior_predictive_specific_source(
        measurements, given, measurement_variances, extra_variances
    )
    sampled_locations = gmm.get_locations(indices_locations)
    assert indices_locations.shape == torch.Size([n_samples])
    # 1 is the number of features
    assert locations_sampled.shape == torch.Size([n_samples, 1])
    assert samples_from_gmm.shape == torch.Size([n_samples, 1])
    assert measurements.shape == torch.Size([n_samples, 1, n_replicates_measurements])
    assert log_probability_given_known_locations.shape == torch.Size([n_samples, 1])
    assert log_prior_predictive.shape == torch.Size([n_samples, 1])
    assert log_posterior_predictive.shape == torch.Size([n_samples, 1])
    assert log_posterior_predictive_specific_source.shape == torch.Size([n_samples, 1])
    assert sampled_locations.shape == torch.Size([n_samples, 1])
