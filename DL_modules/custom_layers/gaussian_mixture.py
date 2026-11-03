import torch
import torch.distributions as dist


class GaussianMixtureModel:
    def __init__(self, locations: torch.Tensor, variances: torch.Tensor, weights: torch.Tensor):
        self.locations = self._set_locations(locations)  # shape = (1, n_locations, n_features)
        self.variances = self._set_variances(variances)  # shape = (1, n_locations, n_features)
        self.log_weights_normalized = self._set_log_weights(weights)  # shape = (1, n_locations, n_features)

        """
        Gaussian Mixture Model. All features are treated as independent.

        :param: locations: torch.Tensor with elements of type Int or Float with shape (n, m): n locations, m features
        :param: variances: torch.Tensor with elements of type Int or Float with shape (n, m): each location element
            has its own variance, or Tensor with shape (m): each feature has its own variance.
            All locations given feature have same variance.
        :param: weights: torch.Tensor with elements of type Int or Float with shape (n): weights of the location
            vectors, these do not have to be normalized
        """

    def sample_location_indices(self, n_samples: int) -> torch.Tensor:
        """
        Draw a random sample of locations of length n according to the weights.

        """
        weights_column = self._check_and_select_column_of_log_weights_normalized()

        # Use torch.multinomial to sample indices based on the weights
        indices = torch.multinomial(torch.exp(weights_column), n_samples, replacement=True)
        return indices

    def sample_from_gmm_given_location_indices(self, location_indices: torch.Tensor) -> torch.Tensor:
        locations = self.locations[:, location_indices].squeeze(0)
        variances = self.variances[:, location_indices].squeeze(0)

        # Draw samples from a normal distribution for each mean/variance pair
        sampled_coordinates = torch.normal(locations, torch.sqrt(variances))

        return sampled_coordinates

    def get_locations(self, indices: torch.Tensor) -> torch.Tensor:
        locations = self.locations[:, indices]
        return locations.squeeze(0)

    def compute_log_prior_predictive(
        self, measurements: torch.Tensor, measurement_variances: torch.Tensor, variances_extra: torch.Tensor
    ) -> torch.Tensor:
        """Compute the log_prior_predictive for each measurement (each row and repetition separately).

        :param: measurements: n, f, r torch.Tensor. Measurements with r repetitions of f features, n rows
        :param: measurement_variances: (f,) torch.Tensor. Measurement variances for f features.
            Replicates are assumed to have the same measurement_variances
        :param variances_extra: (f, ) torch.Tensor. Extra variances for f features.
            These are assumed constant and are added to the total variance without Bayesian updating.
        :return: n, f torch.Tensor. log_prior_predictive probabilities of means.
        """
        if measurements.shape[1] != len(measurement_variances):
            raise ValueError("n_features in Measurements and measurement_variances must have the same length")

        if measurements.shape[1] != len(variances_extra):
            raise ValueError("n_features in Measurements and variances_extra must have the same length")

        # Initialize tensor to hold log probabilities
        log_probs = self._compute_log_predictive(
            measurements,
            measurement_variances,
            variances_extra,
            self.locations,
            self.variances,
            self.log_weights_normalized,
        )

        # return tensor of shape (n_measurements (n), n_features (f))
        return log_probs

    def compute_log_posterior_predictive(
        self,
        measurements: torch.Tensor,
        given: torch.Tensor,
        within_variances: torch.Tensor,
        measurement_variances_extra: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param measurements: n, f, r torch.Tensor. Measurements with r repetitions of f features, n rows
        :param given: n, f, s torch.Tensor. The 'given' measurements. The GMM is updated by these. n rows are treated
            as pairs with the corresponding rows of measurements. For the s replicates the mean is used.
        :param within_variances: (f,) torch.Tensor. Measurement variances for f features.
            Used in Bayesian updating of posterior mean.
        :param measurement_variances_extra: (f,) torch.Tensor. Extra variances for f features.
            These are assumed constant and are added to the total variance and not used in Bayesian updating of mean.
        :return: n, f torch.Tensor of log_posterior_predictive probabilities of measurements row 'given' the same row.
            Each feature is treated independently.
        """
        n, f, _ = measurements.shape
        m, g, _ = given.shape
        if n != m:
            raise ValueError("The number of rows in measurements and the number of rows in 'given' must be equal")
        if f != g:
            raise ValueError("the number of features in measurements and 'given' must be equal")
        if measurements.shape[1] != len(within_variances):
            raise ValueError("n_features (dim 1) in Measurements and length of measurement_variances must be the same")
        if measurements.shape[1] != len(measurement_variances_extra):
            raise ValueError(
                "n_features (dim 1) of Measurements and length of measurement_variances_extra must eb the same"
            )
        if given.shape[0:1] != measurements.shape[0:1]:
            raise ValueError("batchsize or n_features of measurements and given is not the same")
        given_variances = within_variances.repeat(n, 1)
        log_weights_normalized_updated = self._update_weights_components(given, given_variances)
        updated_locations, updated_variances = self._update_locations_variances_gmm(given, given_variances)
        log_probs = self._compute_log_predictive(
            measurements,
            within_variances.repeat(n, 1),
            measurement_variances_extra.repeat(n, 1),
            updated_locations,
            updated_variances,
            log_weights_normalized_updated,
        )

        return log_probs

    @staticmethod
    def compute_log_posterior_predictive_specific_source(
        measurements: torch.Tensor,
        given: torch.Tensor,
        within_variances: torch.Tensor,
        measurement_variances_extra: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param measurements: n, f, r torch.Tensor. Measurements with r repetitions of f features, n rows
        :param measurement_variances: n, f, torch.Tensor. Measurement variances for f features, n rows.
        :param given: n, f, s torch.Tensor. The 'given' measurements. The GMM is updated by these. n rows are treated
            as pairs with the corresponding rows of measurements. For the s replicates the mean is used.
        :param given_variances: n, f, torch.Tensor. Measurement variances for f features, n rows.
            Replicates are assumed to have the same variance.
        :return: n, f torch.Tensor of log_posterior_predictive probabilities of measurements row 'given' the same row.
            Each feature is treated independently. We assume a improper uninformative prior for the mean.
        """
        n, f, _ = measurements.shape
        m, g, _ = given.shape
        if n != m:
            raise ValueError("The number of rows in measurements and the number of rows in 'given' must be equal")
        if f != g:
            raise ValueError("the number of features in measurements and 'given' must be equal")
        if measurements.shape[1] != len(within_variances):
            raise ValueError("n_features Measurements and length(measurement_variances) must be equal")
        if measurements.shape[1] != len(measurement_variances_extra):
            raise ValueError("n_features given and measurement_variances must have the same shape")

        if measurements.shape[0:2] != given.shape[0:2]:
            raise ValueError("Measurements and given must have same batch_size (dim 0) and n_features (dim 1)")

        n_rep_measurements = measurements.shape[2]
        n_rep_given = given.shape[2]
        measurement_means = measurements.mean(dim=2)
        measurement_means = measurement_means.unsqueeze(2)
        given_means = given.mean(dim=2)
        given_means = given_means.unsqueeze(2)
        updated_variances = (1 / n_rep_measurements + 1 / n_rep_given) * within_variances + measurement_variances_extra
        updated_variances = updated_variances.repeat(n, 1).unsqueeze(2)

        # Compute the log probability
        normal = dist.Normal(loc=given_means, scale=torch.sqrt(updated_variances))
        log_probs = normal.log_prob(measurement_means)

        # return tensor of shape (n_measurements, n_features)
        return log_probs.squeeze(-1) / torch.log(torch.tensor(10.0))

    @staticmethod
    def draw_measurements(means: torch.Tensor, measurement_variances: torch.Tensor, n_replicates: int) -> torch.Tensor:
        means = means.unsqueeze(-1)
        measurement_variances = measurement_variances.unsqueeze(-1)

        # Draw samples from a normal distribution for each mean/variance pair
        sampled_measurements = torch.normal(
            means.repeat(1, 1, n_replicates), torch.sqrt(measurement_variances.repeat(1, 1, n_replicates))
        )
        return sampled_measurements

    @staticmethod
    def compute_log_probabilities_given_known_locations(
        measurements: torch.Tensor, known_locations: torch.Tensor, measurement_variances: torch.Tensor
    ) -> torch.Tensor:
        """

        :param measurements: n, f, r torch.Tensor. Measurements with r repetitions of f features, n rows.
            Each of the n * r measurements is treated separately.
        :param known_locations: n, f torch Tensor of locations
         :param measurement_variances: (f,) torch.Tensor. Measurement variances for f features
        :return: log_probs of measurements given true locations and measurement_variances, shape (n, f).
        """
        measurement_means = measurements.mean(dim=2)
        n_batch, _, n_rep = measurements.shape
        measurement_variances = measurement_variances.repeat(n_batch, 1)

        # Compute the log probability
        normal = dist.Normal(loc=known_locations, scale=torch.sqrt(measurement_variances / n_rep))
        log_probs = normal.log_prob(measurement_means)

        return log_probs / torch.log(torch.tensor(10.0))

    @staticmethod
    def _compute_log_predictive(
        measurements: torch.Tensor,
        measurement_variances: torch.Tensor,
        variances_extra: torch.Tensor,
        locations_gmm: torch.Tensor,
        variances_gmm: torch.Tensor,
        log_weights_normalized_gmm: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param measurements: (l, m, k) torch.Tensor. l measurements with k repetitions of m features
        :param measurement_variances: (m,) torch tensor. Variances per measurement of m features.
            Replicates are assumed to have the same measurement_variance
        :param variances_extra: (m,) torch tensor. Variances per measurement of m features.
        :return: (l, m, 1) torch.Tensor. log_prior_predictive probabilities.
            All rows in measurements are treated separately, the replicates are treated as means
        """
        log_probability_components = compute_log_predictive_components(
            measurements,
            measurement_variances,
            variances_extra,
            locations_gmm,
            variances_gmm,
            log_weights_normalized_gmm,
        )

        # Sum log probabilities across locations for each measurement
        # Use log-sum-exp for numerical stability
        log_probs_sum = torch.logsumexp(log_probability_components, dim=-1)

        return log_probs_sum / torch.log(torch.tensor(10.0))

    @staticmethod
    def _set_locations(locations: torch.Tensor) -> torch.Tensor:
        """Init class by setting locations."""
        if locations.dim() != 2:
            raise ValueError("locations must have 2 dimensions")
        return locations.float().unsqueeze(0)

    def _set_log_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Init class by setting log weights."""
        if weights.dim() != 1:
            raise ValueError("weights must have 1 dimension")
        if weights.size(0) != self.locations.size(1):
            raise ValueError("the number of weights and the number of locations must be equal")
        weights_normalized = weights / torch.sum(weights)
        weights_normalized = torch.unsqueeze(weights_normalized.float(), 1)
        return torch.log(weights_normalized.repeat(1, self.locations.size(2))).unsqueeze(0)

    def _set_variances(self, variances: torch.Tensor) -> torch.Tensor:
        """Init class by setting variances."""
        if variances.dim() == 2:
            # then each location and feature has its own variance.
            if variances.size() != self.locations.size()[1:]:
                raise ValueError("Variances and locations do not have the same size.")
            return variances.float().unsqueeze(0)
        elif variances.dim() == 1:
            # Then all kde_locations for a feature are assumed to have same variance.
            # Repeat this variance across locations...
            if variances.size(0) != self.locations.size(2):  # locations.size(2) is number of features.
                raise ValueError(
                    "Variances and features do not have the same length, "
                    "each feature should have one variance associated with it."
                )
            return variances.float().repeat(self.locations.size(1), 1).unsqueeze(0)
        else:
            raise ValueError("Variances must have dimension 1 or 2.")

    def _check_and_select_column_of_log_weights_normalized(self) -> torch.Tensor:
        """Check if all columns are identical by comparing all columns to the first one.
        If so, return one of the columns.
        """
        lwn = self.log_weights_normalized.squeeze(0)
        first_column = lwn[:, 0]
        all_identical = (lwn == first_column.unsqueeze(1)).all(dim=0)

        if all_identical.all():
            # If all columns are identical, return just the first column
            return first_column  # The resulting tensor will now be n
        else:
            raise ValueError(
                "Weights for al features must be identical, "
                "so that random draw of component by multinomial procedure does not depend on which "
                "feature."
            )

    def _update_weights_components(self, given: torch.Tensor, given_variances: torch.Tensor) -> torch.Tensor:
        """Compute normalized posterior weights given measurements and measurement_variances."""
        _, _, n_rep = given.shape

        given_variances_of_means = given_variances / n_rep
        given_means = torch.mean(given, dim=2, keepdim=True)
        log_updated_weights = compute_log_predictive_components(
            given_means,
            given_variances_of_means,
            torch.zeros(given_variances_of_means.shape),
            self.locations,
            self.variances,
            self.log_weights_normalized,
        )

        log_updated_weights_normalized = log_updated_weights - torch.logsumexp(
            log_updated_weights, dim=-1, keepdim=True
        )

        # return torch tensor of shape (n_measurements, n_locations, n_features)
        return log_updated_weights_normalized.permute(0, 2, 1)

    def _update_locations_variances_gmm(
        self, given: torch.Tensor, given_variances: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior location variances of gmmm given measurements and measurement_variances."""
        n_measurements, _, n_rep = given.shape

        given_variances_of_means = given_variances / n_rep

        given_means = torch.mean(given, dim=2)

        # self.variances.shape = (n_locations, n_features).
        # given_variances_of_means.shape = given_means.shape = (1, n_features).
        # Below, we assume that measurement (given) variance is constant. So we use broadcasting.

        # Precalculate quantities where the self.variances aren't zero.
        # If they are zero, there is no uncertainty about the locations, and the posterior location equals the prior
        # location and the variance about the location is still zero.
        nonzero_precision = torch.where(self.variances != 0, 1 / self.variances, torch.zeros_like(self.variances))

        # Prepare for broadcasting
        given_variances_of_means = given_variances_of_means.unsqueeze(1)
        given_means = given_means.unsqueeze(1)

        summed_precision = nonzero_precision + 1 / given_variances_of_means

        # Calculate posterior locations and variances
        locations_posterior = (
            self.locations / self.variances + given_means / given_variances_of_means
        ) / summed_precision
        locations_posterior = torch.where(
            self.variances.repeat(n_measurements, 1, 1) == 0,
            self.locations.repeat(n_measurements, 1, 1),
            locations_posterior,
        )

        variances_posterior = 1 / summed_precision
        variances_posterior = torch.where(
            self.variances.repeat(n_measurements, 1, 1) == 0, torch.zeros_like(variances_posterior), variances_posterior
        )

        # return two tensors of shape (n_measurements, n_locations, n_features)
        return locations_posterior, variances_posterior


def compute_log_predictive_components(
    measurements: torch.Tensor,
    measurement_variances: torch.Tensor,
    variances_extra: torch.Tensor,
    locations_gmm: torch.Tensor,
    variances_gmm: torch.Tensor,
    log_weights_normalized_gmm: torch.Tensor,
) -> torch.Tensor:
    """

    :param measurements: n, f, r torch.Tensor. l measurements with r repetitions of f features
    :param measurement_variances: n, f torch tensor. Variances per measurement of f features.
        Replicates are assumed to have the same measurement_variance
    :param variances_extra: f torch tensor. Variances per measurement of f features.
    :param locations_gmm: m, f torch tensor. m locations of f features
    :param variances_gmm: m, f torch tensor. m * f variances of locations_gmm elements
    :param log_weights_normalized_gmm: m * f log weights of f features
    :return: n, f, m torch.Tensor. log_prior_predictive probability components.
        All measurements are treated as single measurements, so each row and replicate is treated separately.
    """
    n, f, r = measurements.shape
    measurement_means = measurements.mean(dim=2)
    v = measurement_variances.shape[-1]
    if len(measurement_variances.shape) == 1:
        measurement_variances = measurement_variances.repeat(n, 1)
    if len(variances_extra.shape) == 1:
        variances_extra = variances_extra.repeat(n, 1)

    if f != v:
        raise ValueError(
            "Measurements must have the same number of columns (features) as measurement variance components."
        )

    # prepare tensors for broadcasting to n_measurements, n_features, n_locations
    measurement_means = measurement_means.unsqueeze(2)
    locations_gmm = locations_gmm.permute(0, 2, 1)
    log_weights_normalized_gmm = log_weights_normalized_gmm.permute(0, 2, 1)
    variances_gmm = variances_gmm.permute(0, 2, 1)
    measurement_variances = measurement_variances.unsqueeze(2)
    variances_extra = variances_extra.unsqueeze(2)

    # Calculate log probabilities for each Gaussian component
    # std has shape (n_measurements, n_features, n_locations)
    std = torch.sqrt(variances_gmm + measurement_variances / r + variances_extra)
    normal = dist.Normal(loc=locations_gmm, scale=std)
    log_prob_components = normal.log_prob(measurement_means)

    log_prob_components = log_prob_components + log_weights_normalized_gmm

    # return tensor of shape (n_measurements (n), n_features (f), n_locations)
    return log_prob_components


class KDE(GaussianMixtureModel):
    def __init__(self, locations: torch.Tensor):
        variances = self.get_silverman_bandwidths(locations)
        n_locations = locations.shape[0]
        super().__init__(locations, variances, torch.full((n_locations,), 1 / n_locations))

    # identical copy from get_kde_bandwidth in KL_div_to_normal, but adjusted for multiple features
    @staticmethod
    def get_silverman_bandwidths(data: torch.Tensor) -> torch.Tensor:
        sd = torch.std(data, dim=0, unbiased=True)
        q1 = torch.quantile(data, 0.25, dim=0)
        q3 = torch.quantile(data, 0.75, dim=0)
        iqr = q3 - q1
        if sd == 0.0:
            raise ValueError('KDE: Silverman bandwidth cannot be calculated if standard deviation is 0')

        h = 0.9 * torch.pow(torch.tensor(len(data)), -0.2) * torch.min(sd, iqr / 1.34)
        return h
