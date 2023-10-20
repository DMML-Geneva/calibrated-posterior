import hypothesis as h
import torch
from torch.nn import functional as F
import torchsort
from functorch import vmap

from hypothesis.nn.ratio_estimation import BaseCriterion


class RegularizedCriterion(BaseCriterion):
    def __init__(
        self,
        estimator,
        batch_size=h.default.batch_size,
        gamma=100.0,
        logits=False,
        **kwargs,
    ):
        super(RegularizedCriterion, self).__init__(
            estimator=estimator, batch_size=batch_size, logits=logits, **kwargs
        )
        self._gamma = gamma

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        assert value >= 0
        self._gamma = value


def get_logq_for_ranks(model, x, y, prior, n_samples):
    if x.shape[-1] == 1:
        sample_shape = x.shape
    else:
        sample_shape = x.shape[:1]
    log_prior_probabilities = [prior.log_prob(x).sum(-1)]
    log_ratios = [
        model.log_ratio(
            inputs=x,
            outputs=y,
        )
    ]
    for _ in range(n_samples):
        log_prior_probabilities.append(
            prior.log_prob((samples := prior.sample(sample_shape))).sum(-1)
        )
        log_ratios.append(
            model.log_ratio(
                inputs=samples,
                outputs=y,
            )
        )
    return (
        torch.stack(log_prior_probabilities, dim=1)
        + torch.stack(log_ratios, dim=1).squeeze()
    )


def batched_get_logq_for_ranks(model, x, y, prior, n_samples):
    if x.shape[-1] == 1:
        inputs = torch.cat(
            [
                x.unsqueeze(1),
                prior.sample(x.shape + (n_samples,)).movedim(-1, 1),
            ],
            dim=1,
        )
    else:
        inputs = torch.cat(
            [
                x.unsqueeze(1),
                prior.sample(x.shape[:1] + (n_samples,)),
            ],
            dim=1,
        )
    log_prior_probabilities = prior.log_prob(inputs).sum(-1)
    log_ratios = vmap(model.log_ratio, in_dims=(1, None), out_dims=1)(
        inputs, y
    ).squeeze()
    return log_prior_probabilities + log_ratios


class STEFunctionRanksq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class STEFunctionRankslogq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.where(input > 0, 0, float("-inf"))

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


def get_ranks(model, x, y, prior, n_samples, logits=False, batched=True):
    if batched:
        logq = batched_get_logq_for_ranks(model, x, y, prior, n_samples)
    else:
        logq = get_logq_for_ranks(model, x, y, prior, n_samples)
    if logits:
        return (
            (
                logq[:, 1:]
                + STEFunctionRankslogq.apply(
                    logq[:, 0].unsqueeze(1) - logq[:, 1:]
                )
            ).logsumexp(dim=1)
            - logq[:, 1:].logsumexp(dim=1)
        ).exp()
    else:
        q = logq.exp()
        return (
            q[:, 1:] * STEFunctionRanksq.apply(q[:, 0].unsqueeze(1) - q[:, 1:])
        ).sum(dim=1) / q[:, 1:].sum(dim=1)


def get_coverage(ranks, device):
    # Source: https://github.com/montefiore-ai/balanced-nre/blob/main/demo.ipynb
    # As a sample at a given rank belongs to the credible regions at levels 1-rank and below,
    # the coverage at level 1-alpha is the proportion of samples with ranks alpha and above.
    ranks = ranks[~ranks.isnan()]
    alpha = torchsort.soft_sort(ranks.unsqueeze(0)).squeeze()
    return (
        torch.linspace(0.0, 1.0, len(alpha) + 2, device=device)[1:-1],
        1 - torch.flip(alpha, dims=(0,)),
    )


def get_calibration_error(
    model,
    x,
    y,
    device,
    prior,
    n_samples,
    calibration=0,
    logits=False,
    instances_subsample=1,
    batched=True,
):
    if instances_subsample < 1:
        idxs = torch.randperm(x.shape[0], device=x.device)
        x = x[idxs[: int(x.shape[0] * instances_subsample)]]
        y = y[idxs[: int(y.shape[0] * instances_subsample)]]
    ranks = get_ranks(
        model,
        x,
        y,
        prior,
        logits=logits,
        n_samples=n_samples,
        batched=batched,
    )
    coverage, expected = get_coverage(ranks, device)
    if calibration == 0:
        return torch.nn.functional.relu(expected - coverage).pow(2).mean()
        # default in BNRE paper, conservatiness regularizer
    elif calibration == 1:
        return (coverage - expected).pow(2).mean()  # calibration regularizer
    else:
        return (
            (
                (1 - calibration)
                * torch.nn.functional.relu(expected - coverage)
                + calibration * (coverage - expected)
            )
            .pow(2)
            .mean()
        )  # mixture of conservatiness and calibration regularizers


class CalibratedCriterion(RegularizedCriterion):
    def __init__(
        self,
        estimator,
        prior,
        n_samples,
        instances_subsample=1,
        calibration=0,
        batch_size=h.default.batch_size,
        gamma=100.0,
        logits=False,
        batched=True,
        device=None,
        **kwargs,
    ):
        super(CalibratedCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            gamma=gamma,
            logits=logits,
            **kwargs,
        )
        self.prior = prior.to(device if device is not None else h.accelerator)
        self.n_samples = n_samples
        self.instances_subsample = instances_subsample
        self.calibration = calibration
        self.batched = batched

    def _forward_without_logits(self, **kwargs):
        effective_batch_size = len(
            kwargs[self._independent_random_variables[0][0]]
        )
        regularizer = get_calibration_error(
            model=self._estimator,
            x=kwargs["inputs"],
            y=kwargs["outputs"],
            device=kwargs["inputs"].device,
            prior=self.prior,
            n_samples=self.n_samples,
            logits=False,
            instances_subsample=self.instances_subsample,
            calibration=self.calibration,
            batched=self.batched,
        )
        y_dependent, _ = self._estimator(**kwargs)
        for group in self._independent_random_variables:
            random_indices = torch.randperm(effective_batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][
                    random_indices
                ]  # Make group independent.
        y_independent, log_r_independent = self._estimator(**kwargs)
        loss = self._criterion(
            y_dependent, self._ones[:effective_batch_size]
        ) + self._criterion(y_independent, self._zeros[:effective_batch_size])
        loss = loss + self._gamma * regularizer
        return loss

    def _forward_with_logits(self, **kwargs):
        effective_batch_size = len(
            kwargs[self._independent_random_variables[0][0]]
        )
        regularizer = get_calibration_error(
            model=self._estimator,
            x=kwargs["inputs"],
            y=kwargs["outputs"],
            device=kwargs["inputs"].device,
            prior=self.prior,
            n_samples=self.n_samples,
            logits=True,
            instances_subsample=self.instances_subsample,
            calibration=self.calibration,
            batched=self.batched,
        )
        y_dependent, log_r_dependent = self._estimator(**kwargs)
        for group in self._independent_random_variables:
            random_indices = torch.randperm(effective_batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][
                    random_indices
                ]  # Make group independent.
        y_independent, log_r_independent = self._estimator(**kwargs)
        loss = self._criterion(
            log_r_dependent, self._ones[:effective_batch_size]
        ) + self._criterion(
            log_r_independent, self._zeros[:effective_batch_size]
        )
        loss = loss + self._gamma * regularizer

        return loss


# class ConservativeEqualityCriterion(RegularizedCriterion):
#     def __init__(
#         self,
#         estimator,
#         batch_size=h.default.batch_size,
#         gamma=1.0,
#         logits=False,
#         **kwargs,
#     ):
#         super(ConservativeEqualityCriterion, self).__init__(
#             estimator=estimator,
#             batch_size=batch_size,
#             gamma=gamma,
#             logits=logits,
#             **kwargs,
#         )
#
#     def _forward_without_logits(self, **kwargs):
#         # Forward passes
#         y_joint, _ = self._estimator(**kwargs)
#         ## Shuffle to make necessary variables independent.
#         for group in self._independent_random_variables:
#             random_indices = torch.randperm(self._batch_size)
#             for variable in group:
#                 kwargs[variable] = kwargs[variable][
#                     random_indices
#                 ]  # Make group independent.
#         y_marginals, log_r_marginals = self._estimator(**kwargs)
#         # Compute losses
#         loss_joint_1 = self._criterion(y_joint, self._ones)
#         loss_marginals_0 = self._criterion(y_marginals, self._zeros)
#         # Learn mixture of the joint vs. marginals
#         loss = loss_joint_1 + loss_marginals_0
#         regularizer = (1 - log_r_marginals.exp().mean()).pow(2)
#         loss = loss + self._gamma * regularizer
#
#         return loss
#
#     def _forward_with_logits(self, **kwargs):
#         # Forward passes
#         _, log_r_joint = self._estimator(**kwargs)
#         ## Shuffle to make necessary variables independent.
#         for group in self._independent_random_variables:
#             random_indices = torch.randperm(self._batch_size)
#             for variable in group:
#                 kwargs[variable] = kwargs[variable][
#                     random_indices
#                 ]  # Make group independent.
#         _, log_r_marginals = self._estimator(**kwargs)
#         # Compute losses
#         loss_joint_1 = self._criterion(log_r_joint, self._ones)
#         loss_marginals_0 = self._criterion(log_r_marginals, self._zeros)
#         # Learn mixture of the joint vs. marginals
#         loss = loss_joint_1 + loss_marginals_0
#         regularizer = (1 - log_r_marginals.exp().mean()).pow(2)
#         loss = loss + self._gamma * regularizer
#
#         return loss
#
#
# class VariationalInferenceCriterion(BaseCriterion):
#     def __init__(
#         self,
#         estimator,
#         batch_size=h.default.batch_size,
#         logits=False,
#         dataset_train_size=None,
#         **kwargs,
#     ):
#         super(VariationalInferenceCriterion, self).__init__(
#             estimator=estimator, batch_size=batch_size, logits=logits, **kwargs
#         )
#
#         self._dataset_train_size = dataset_train_size
#
#     def _forward_without_logits(self, **kwargs):
#         # Forward passes
#         y_joint, log_r_joint = self._estimator(**kwargs)
#         effective_batch_size = len(
#             kwargs[self._independent_random_variables[0][0]]
#         )
#         ## Shuffle to make necessary variables independent.
#         for group in self._independent_random_variables:
#             random_indices = torch.randperm(effective_batch_size)
#             for variable in group:
#                 kwargs[variable] = kwargs[variable][
#                     random_indices
#                 ]  # Make group independent.
#         y_marginals, log_r_marginals = self._estimator(**kwargs)
#
#         data_log_likelihood = (
#             torch.log(y_joint).mean() + torch.log(1 - y_marginals).mean()
#         ) * self._dataset_train_size
#         kl_weight_prior = self._estimator.kl_loss()
#         loss = kl_weight_prior - data_log_likelihood
#
#         return loss
#
#     def _forward_with_logits(self, **kwargs):
#         raise NotImplementedError()
#
#
# class VariationalInferenceCriterionNoKL(BaseCriterion):
#     def __init__(
#         self,
#         estimator,
#         batch_size=h.default.batch_size,
#         logits=False,
#         dataset_train_size=None,
#         **kwargs,
#     ):
#         super(VariationalInferenceCriterion, self).__init__(
#             estimator=estimator, batch_size=batch_size, logits=logits, **kwargs
#         )
#
#         self._dataset_train_size = dataset_train_size
#
#     def _forward_without_logits(self, **kwargs):
#         # Forward passes
#         y_joint, log_r_joint = self._estimator(**kwargs)
#         effective_batch_size = len(
#             kwargs[self._independent_random_variables[0][0]]
#         )
#         ## Shuffle to make necessary variables independent.
#         for group in self._independent_random_variables:
#             random_indices = torch.randperm(effective_batch_size)
#             for variable in group:
#                 kwargs[variable] = kwargs[variable][
#                     random_indices
#                 ]  # Make group independent.
#         y_marginals, log_r_marginals = self._estimator(**kwargs)
#
#         data_log_likelihood = (
#             torch.log(y_joint).mean() + torch.log(1 - y_marginals).mean()
#         ) * self._dataset_train_size
#         loss = -data_log_likelihood
#
#         return loss
#
#     def _forward_with_logits(self, **kwargs):
#         raise NotImplementedError()
#
#
# class KLCriterion(BaseCriterion):
#     def __init__(
#         self, estimator, batch_size=h.default.batch_size, logits=False, **kwargs
#     ):
#         super(KLCriterion, self).__init__(
#             estimator=estimator, batch_size=batch_size, logits=logits, **kwargs
#         )
#
#     def _forward_without_logits(self, **kwargs):
#         log_posterior_joint = self._estimator.log_posterior(**kwargs)
#         loss = -log_posterior_joint.mean()
#
#         return loss
#
#
# class KLBalancedCriterion(BaseCriterion):
#     def __init__(
#         self,
#         estimator,
#         batch_size=h.default.batch_size,
#         gamma=100.0,
#         logits=False,
#         **kwargs,
#     ):
#         super(KLBalancedCriterion, self).__init__(
#             estimator=estimator, batch_size=batch_size, logits=logits, **kwargs
#         )
#         self._gamma = gamma
#
#     def _forward_without_logits(self, **kwargs):
#         (
#             log_posterior_joint,
#             y_joint,
#         ) = self._estimator.log_posterior_with_classifier(**kwargs)
#
#         effective_batch_size = len(
#             kwargs[self._independent_random_variables[0][0]]
#         )
#         ## Shuffle to make necessary variables independent.
#         for group in self._independent_random_variables:
#             random_indices = torch.randperm(effective_batch_size)
#             for variable in group:
#                 kwargs[variable] = kwargs[variable][
#                     random_indices
#                 ]  # Make group independent.
#
#         (
#             log_posterior_marginal,
#             y_marginal,
#         ) = self._estimator.log_posterior_with_classifier(**kwargs)
#         loss = -log_posterior_joint.mean()
#
#         # Balacing condition
#         regularizer = (1.0 - y_joint - y_marginal).mean().pow(2)
#         loss = loss + self._gamma * regularizer
#
#         return loss
#
#     def _forward_with_logits(self, **kwargs):
#         raise NotImplementedError()
