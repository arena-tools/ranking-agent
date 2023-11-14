from __future__ import annotations

from collections import namedtuple
from typing import Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import linear_model


ZERO_ONE_CLASSES = [0, 1]
MAX_SGD_ITERS = 100

ParamTuple = namedtuple("ParamTuple", ["action_id", "weight", "cov"])
ParamUpdate = Tuple[np.ndarray, np.ndarray]

class RankingBanditAgent(object):
    """Active learning agent that learns to rank.

    This agent models the probability of clicking on each item as a logistic
    function and maximizes the upper confidence bound maximum likelihood estimate of
    the probability of user clicking on an item given their ranking over the course
    of training.

    An assumption is made that the positional effects of each item is different, and
    that there is a monotonic relationship between the rank and probability of click
    which is why the weight parameter for positional effects in one scalar.

    Parameters
    ----------
    config
        The config specifying parameters of the agent.
    action_spaces
        The list of action spaces to optimize over.
    """

    def __init__(
        self,
        ksi: float = 1,
        context_dim: int = 5,
        horizon: int = 1000,
        randomization_horizon: int = 10,
        gd_n_steps: int = 10,
        learning_rate: float = 0.003,
        n_items: int = 3,
        objective: str = "click-through-rate",
        action_weight_col: Optional[str] = None,
        ucb: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.ucb = ucb
        self.ksi = ksi
        self.horizon = horizon
        self.randomization_horizon = randomization_horizon
        self.n_items = n_items
        self.gd_n_steps = gd_n_steps
        self.learning_rate = learning_rate

        self.action_weight_col = action_weight_col
        self.objective = objective

        self.context_dim: int = context_dim
        self.feature_dim: int = context_dim + 2  # plus 2 for the rank and intercept term [intercept, features, rank]

        self._init_model()
        self.time = 0

    def _init_model(self) -> None:
        # initialize empty parameter and covariance matrices
        self.params = np.random.normal(0, 1, size=[self.n_items, self.feature_dim]) # intercept, context ... , rank
        self.cov = np.zeros(
            [
                self.n_items,
                self.feature_dim,
                self.feature_dim,
            ]
        ) + np.eye(self.feature_dim)

        self.online_lr = linear_model.SGDClassifier(
            loss="log",
            penalty=None,
            fit_intercept=False,
            learning_rate="constant",
            eta0=self.learning_rate,
            max_iter=min(MAX_SGD_ITERS, 10 * self.gd_n_steps),
        )

    @property
    def _exploration_phase(self) -> bool:
        return self.time < self.randomization_horizon
    
    
    def _compute_cost_matrix(
        self, probability_matrix: np.ndarray, action_cost_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the cost matrix for the linear sum assignment problem.

        This matrix changes depending on the selected optimization problem. For optimizing the
        click through rate, we want to optimize:
            1 - prod_{k=1}^{n} (1 - P(Y_{k}=1|X_k, p_k))
        Or equivalently, maximize the probability that they click at least one item.

        Other possible formulations are:
            sum_{k=1}^{n} R_{k}P(Y_{k}=1|X_k, p_k)
        which is revenue/profit maximization, where we have access to the reward function R_{k}.

        Parameters
        ----------
        probability_matrix: np.ndarray
            The matrix of probabilities of shape (n_products, n_positions) generated by the
            logistic regression models and UCB estimates.
        action_cost_matrix: np.ndarray
            The per-action cost matrix of shape (n_products,) which needs to be provided to the
            agent by the client either at initialization or at inference time.
        """

        if self.objective != "Profit":
            return -np.log(np.maximum(1 - probability_matrix, 1e-8))
        else:

            assert (
                action_cost_matrix is not None
            ), """Profit maximization objective requires action costs to be provided 
                and the relevent column specified in the agent config, but the action_weight_col
                value is set to None.
                """

            n_products, n_positions = probability_matrix.shape
            action_cost_matrix = np.repeat(action_cost_matrix, n_positions).reshape(n_products, n_positions)
            return probability_matrix * action_cost_matrix


    def update(self, update_arr: np.ndarray, clicks: np.ndarray) -> RankingBanditAgent:
        """Updates the agent given context array and clicks.

        Parameters
        ----------
        update_arr: np.ndarray
            The matrix of context features of shape (n_items, time, n_context + rank) to update
            the logistic model.
        clicks: np.ndarray
            The array of clicks of shape (n_items, time) representing clicks for each item at
            each timestep.

        Returns
        -------
        self : RankingBanditAgent
            Returns self.
        """

        # randomization period
        if self.time < self.randomization_horizon - 1:
            pass
        elif self.time == self.randomization_horizon - 1:
            # compute covariance
            for k in range(self.n_items):
                cov_feature = update_arr[k, :self.time-1]
                cov_feature = np.hstack([np.ones((cov_feature.shape[0], 1)), cov_feature]) # stacks intercept term
                self.cov[k] = np.einsum("ij,ik->jk", cov_feature, cov_feature)

        else:  # start of active learning
            for k in range(self.n_items):
                cov_feature = np.expand_dims(update_arr[k, -1], -1)
                cov_feature = np.vstack([np.ones((1, 1)), cov_feature]) # stacks intercept term

                update_batch = np.hstack([np.ones((update_arr[k].shape[0], 1)), update_arr[k]])
                self.params[k] = self._solve_logistic_regression(update_batch, clicks[k])
                self.cov[k] += np.outer(cov_feature, cov_feature)
        self.time += 1
        return self
    
    def _solve_logistic_regression(self, item_features: np.ndarray, item_clicks: np.ndarray) -> np.ndarray:
        """Solves for parameters used to compute UCB probability with logistic regression.

        Parameters
        ----------
        k : int
            The index of item.

        Returns
        -------
        params : np.ndarray
            The array containing the optimized parameters.
        """

        # If no clicks or all clicks for this item, make partial updates
        if not np.any(item_clicks) or np.all(item_clicks):
            n_steps = min(MAX_SGD_ITERS, self.gd_n_steps*10)
            weight = np.zeros(item_features.shape[1])
            for _ in range(n_steps):
                sgd_clf = self.online_lr._partial_fit(
                    X=item_features,
                    y=item_clicks,
                    alpha=self.online_lr.alpha,
                    C=1.0,
                    loss="log",
                    learning_rate=self.online_lr.learning_rate,
                    classes=ZERO_ONE_CLASSES,
                    sample_weight=None,
                    coef_init=weight,
                    intercept_init=None,
                    max_iter=1,
                )
                weight = sgd_clf.coef_
            params = weight.squeeze()
        else:
            # solve using sklearn logistic regression
            model = linear_model.LogisticRegression(fit_intercept=False)
            params = model.fit(item_features, item_clicks).coef_
            params = np.array(params).squeeze()
        return params
    

    def rank(
        self,
        features: np.ndarray,
        action_weight_col: Any = None,
    ) -> np.ndarray:
        """Provide a ranking for an arbitrary set of state-item pairs, where state->item is a one->many relation

        Parameters
        ----------
        features : np.ndarray
            Array containing the features for items to be ranked.
        action_weight_col : Any
            If given, indicates the weight of each action in a profit setting.

        Returns
        -------
        product_rank : np.ndarray
            The array containing the predicted rankings.
        """

        # for each id, we need to evaluate its probability at position k
        ranks = np.tile(np.arange(self.n_items), self.n_items)  # [num_products * num_position]
        ones = np.ones(self.n_items * self.n_items)
        features = np.repeat(features, repeats=self.n_items, axis=0)
        features = np.hstack([np.expand_dims(ones, 1), features, np.expand_dims(ranks, 1)]).astype(float)

        # user the param map to access the indexes for the parameter and covariance matrices
        params = np.repeat(self.params, repeats=self.n_items, axis=0)

        ucb_bonus = 0
        if self.ucb:
            cov = np.repeat(self.cov, repeats=self.n_items, axis=0)
            ATx = np.stack([np.linalg.lstsq(c, z.T, rcond=None)[0] for z, c in zip(features, cov.astype(float))])
            ucb_bonus = 3 * self.ksi * np.sqrt((features * ATx).sum(axis=1))
        logit = (params * features).sum(axis=-1) + ucb_bonus

        # the probability matric has products as rows and positions as columns
        prob = self._sigmoid(logit.reshape(self.n_items, self.n_items).astype(float))

        # if applicable, extract per-product profit/revenue
        action_costs = action_weight_col if action_weight_col else None
        cost_matrix = self._compute_cost_matrix(prob, action_costs)
        product_idx, product_rank = linear_sum_assignment(cost_matrix, maximize=True)
        return product_rank

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray: 
        """Compute the sigmoid function: sigma(z) = 1 / (1 + e^{-z})"""
        return np.reciprocal(1 + np.exp(-z))

    