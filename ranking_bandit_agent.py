from __future__ import annotations

from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import linear_model


MAX_CONCURRENCY = 4
INTERCEPT_COL = "intercept"
ZERO_ONE_CLASSES = [0, 1]
MAX_SGD_ITERS = 250

ParamTuple = namedtuple("ParamTuple", ["action_id", "weight", "cov"])
ParamUpdate = Tuple[np.ndarray, np.ndarray]


class RankingBanditAgent():
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
        reward_col: str,
        state_columns: List[str],
        state_id_col: str = "user_id",
        action_id_col: str = "product_id",
        rank_col: str = "rank",
        ksi: float = 1,
        horizon: int = 1000,
        randomization_horizon: int = 10,
        gd_n_steps: int = 10,
        learning_rate: float = 0.003,
        cov_lambda: float = 1.0,
        top_k: Optional[int] = None,
        objective: str = "Profit",
        action_weight_col: Optional[str] = None,
        ucb: bool = True,
        **kwargs: Any,
    ):
        self.ucb: bool = ucb
        self.ksi: float = ksi
        self.horizon: int = horizon
        self.randomization_horizon: int = randomization_horizon
        self.cov_lambda: float = cov_lambda
        self._top_k: int | float = top_k or float("inf")
        self.gd_n_steps: int = gd_n_steps
        self.learning_rate: float = learning_rate

        self.state_columns: List[str] = state_columns
        self.state_id_col: str = state_id_col  # one for each client decision, batch_id
        self.action_id_col: str = action_id_col  # product name or product id
        self.rank_col: str = rank_col
        self.reward_col: str = reward_col

        # allow for model to optimize for different objective functions
        self.objective = objective
        self.action_weight_col: Optional[str] = action_weight_col

        self.feature_columns: List[str] = self.state_columns + [self.rank_col]
        self.state_dim: int = len(self.state_columns)
        self.feature_dim: int = len(self.feature_columns) + 1  # +1 for the intercept term

        self._init_model()
        self.time = 0

    def _init_model(self) -> None:
        # initialize empty parameter and covariance matrices
        self.params = np.zeros((0, self.feature_dim))
        self.cov = np.zeros(
            [
                0,
                self.feature_dim,
                self.feature_dim,
            ]
        ) + self.cov_lambda * np.eye(self.feature_dim)
        self.online_lr = linear_model.SGDClassifier(
            loss="log",
            penalty=None,
            fit_intercept=False,
            learning_rate="constant",
            eta0=self.learning_rate,
            max_iter=min(MAX_SGD_ITERS, 10 * self.gd_n_steps),
        )

        # instantiate a map from action/product ids to model parameters.
        self.action_param_map = pd.Series(dtype="int")

    @property
    def _exploration_phase(self) -> bool:
        return self.time < self.randomization_horizon

    def get_params(self, action_id: Hashable | List[Hashable]) -> np.ndarray | pd.Series:
        """Get the parameters for a given action/product id."""
        if isinstance(action_id, Hashable):
            return self.params[self.action_param_map[action_id]]
        if isinstance(action_id, list) and all([isinstance(a, Hashable) for a in action_id]):
            return pd.DataFrame(
                self.params[self.action_param_map[action_id].to_numpy()],
                index=action_id,
                columns=[INTERCEPT_COL] + self.feature_columns,
            )
        else:
            raise ValueError(
                f"'action_id' must be hashable or list of hashable objects/ids, got {type(action_id)} "
                f"and {type(action_id[0]) if 0 < len(action_id) else None}"
            )

    def fit(self, df: pd.DataFrame) -> RankingBanditAgent:
        """Fit the Product level Models using the observed features and clicks and update
        the Covariance matrices using the historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing historical click data with context and action features columns.
            Must contain the action_id_col to allow the correct product parameters to be updates.
        """

        self.cov, self.params = self._update_action_param_map(df)
        updated_params = self._run_concurrent(df, self.action_id_col, self._fit)
        self.assign_params(updated_params)
        return self

    def update(self, df: pd.DataFrame) -> RankingBanditAgent:
        """Update the Covariance using the observed features and clicks and estimate model parameters."""

        self.cov, self.params = self._update_action_param_map(df)
        updated_params = self._run_concurrent(df, self.action_id_col, self._update)
        self.assign_params(updated_params)
        self.time += 1
        return self

    @staticmethod
    def _run_concurrent(df: pd.DataFrame, group_by_col: str, callable_fn: Callable, *args: Any) -> List[Any]:
        """Helper function for concurrent model updates. Used in the fit and update method."""

        grouped_observations = df.groupby(group_by_col)
        with ProcessPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
            futures = [
                executor.submit(callable_fn, action_id, action_group, *args)
                for action_id, action_group in grouped_observations
            ]

        return [future.result() for future in as_completed(futures)]

    def assign_params(self, updated_params: list[ParamTuple]) -> None:
        """Assign the updated model parameters and covariance matrices to the correct action ids."""
        for param in updated_params:
            idx = self.action_param_map[param.action_id]
            self.params[idx] = param.weight
            self.cov[idx] = param.cov

    def _fit(self, action_id: int, action_group: pd.DataFrame) -> ParamTuple:
        """Same as _update, but fits models from scratch and trains to each model to convergence."""

        action_index = int(self.action_param_map[action_id])
        features = action_group[self.feature_columns].to_numpy().astype(float)
        features = np.hstack([np.ones((features.shape[0], 1)), features])
        y = action_group[self.reward_col]

        updated_cov = self.cov[action_index] + np.einsum("ij,iv->jv", features, features)
        fit_lr_weights = self._fit_logistic_regression(features, y.to_numpy().astype(int))
        return ParamTuple(action_id, fit_lr_weights, updated_cov)

    def _update(self, action_id: int, action_group: pd.DataFrame) -> ParamTuple:
        """For a selected product, udpate the covariance matrix and estimated model parameters.

        Parameters
        ----------
        action_id : str
            Product id for the product to be updated.
        action_group : pd.DataFrame
            dataframe with historical observations for the chosed focal product.
        """

        action_index = int(self.action_param_map[action_id])
        features = action_group[self.feature_columns].to_numpy().astype(float)
        features = np.hstack([np.ones((features.shape[0], 1)), features])
        y = action_group[self.reward_col].to_numpy().astype(int)

        updated_cov = self.cov[action_index] + np.einsum("ij,iv->jv", features, features)
        weight = self._partial_fit(X=features, y=y, weight=self.params[action_index])
        return ParamTuple(action_id, weight, updated_cov)

    def _fit_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Fit the logistic regression model using n-iterations of SGD.

        Checks for convergence by comparing the change in the model parameters between iterations.
        We use this as opposed to just calling online_lr.fit(), since in the case of a sample with
        only positive or negative examples, sklearn will raise an error, since model convergence is
        not well defined in this case (parameters go to +- infinite).
        """

        if np.any(y) and ~np.all(y):
            return self.online_lr.fit(X, y).coef_

        # If all examples are positive or negative, use the partial fit method.
        return self._partial_fit(X, y, np.zeros(X.shape[1]), n_steps=10 * self.gd_n_steps)

    def _partial_fit(self, X: np.ndarray, y: np.ndarray, weight: np.ndarray, n_steps: Optional[int] = None) -> np.ndarray:
        """Perform online update to the logistic regression model using n-iterations of SGD."""
        n_steps = min(MAX_SGD_ITERS, n_steps or self.gd_n_steps)
        for _ in range(n_steps):
            sgd_clf = self.online_lr._partial_fit(
                X=X,
                y=y,
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
        return weight

    def _rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes ranking of items based on UCB/MLE probabilities.

        Note that we compute ucb bonus without the pseudo-inverse by using least square solution to
        Ay = z. The lstsq solution will be ~A^{+}z, which avoids directly computing the pseudo-inverse,
        which can be computationally intensive.

        See https://www.notion.so/arena-ai/Learning-to-Rank-6853b8ccd6f84b4caed9c72de339687b for
        further documentation of the algorithm.

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe corresponding to a **single** state with N possible actions.
            This would represent a dataframe with N items to ranks for a single
            user. The context values will be constant, but the 'product' features will change.

        Returns
        -------
        rank : pd.DataFrame
            The DataFrame containing the predicted rankings.
        """

        features = df[self.state_columns]
        action_ids = df[self.action_id_col]

        # for each id, we need to evaluate its probability at position k
        n_products = action_ids.shape[0]
        n_positions = min(n_products, self._top_k)

        ranks = np.tile(np.arange(n_positions), n_products)  # [num_products * num_position]
        ones = np.ones(n_positions * n_products)
        features = np.repeat(features.to_numpy(), repeats=n_positions, axis=0)
        features = np.hstack([np.expand_dims(ones, 1), features, np.expand_dims(ranks, 1)]).astype(float)

        # user the param map to access the indexes for the parameter and covariance matrices
        action_indexes = self.action_param_map[action_ids].to_numpy().astype(int)
        params = np.repeat(self.params[action_indexes], repeats=n_positions, axis=0)

        ucb_bonus = 0
        if self.ucb:
            cov = np.repeat(self.cov[action_indexes], repeats=n_positions, axis=0)
            ATx = np.stack([np.linalg.lstsq(c, z.T, rcond=None)[0] for z, c in zip(features, cov.astype(float))])
            ucb_bonus = 3 * self.ksi * np.sqrt((features * ATx).sum(axis=1))
        logit = (params * features).sum(axis=-1) + ucb_bonus

        # the probability matric has products as rows and positions as columns
        prob = self._sigmoid(logit.reshape(n_products, n_positions).astype(float))

        # if applicable, extract per-product profit/revenue
        action_costs = df[self.action_weight_col].to_numpy() if self.action_weight_col else None
        cost_matrix = self._compute_cost_matrix(prob, action_costs)
        product_idx, product_rank = linear_sum_assignment(cost_matrix, maximize=True)

        # re-attatch original action-id values to ranking dataframe.
        top_k_products = df[self.action_id_col].iloc[product_idx]

        if self.objective == "Profit":
            top_k_revenues = df[self.action_weight_col].iloc[product_idx]
            return pd.DataFrame(
                {self.action_id_col: top_k_products, self.rank_col: product_rank, self.action_weight_col: top_k_revenues}
            )
        else:
            return pd.DataFrame({self.action_id_col: top_k_products, self.rank_col: product_rank})

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
            ), """Profit maximization objective requires action costs to be provided in the decision
                dataframe and the relevent column specified in the agent config, but the action_weight_col
                value is set to None.
                """

            n_products, n_positions = probability_matrix.shape
            action_cost_matrix = np.repeat(action_cost_matrix, n_positions).reshape(n_products, n_positions)
            return probability_matrix * action_cost_matrix


    def predict(
        self,
        df: pd.DataFrame,
        keep_all_fields: bool = False,
    ) -> pd.DataFrame:
        """Given a dataframe of observations, predict the probability of a click for each recommended product.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the user-item pairs which are to be ranked, along with the corresponding features

        Returns
        -------
        rankings : pd.DataFrame
            The DataFrame containing the predicted probabilities
        """
        return self.__rank_or_predict(self._predict, df, keep_all_fields)

    def rank(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Provide a ranking for an arbitrary set of state-item pairs, where state->item is a one->many relation

        NOTE: Ensure that the dataframe contains the USER_ID_COLUMN, PRODUCT_ID_COLUMN, & RANK_COLUMN.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the user-item pairs which are to be ranked, along with the corresponding features

        Returns
        -------
        rankings : pd.DataFrame
            The DataFrame containing the predicted rankings.
        """

        self.cov, self.params = self._update_action_param_map(df)

        return_df = pd.concat(self._run_concurrent(df, self.state_id_col, self._get_decision, self._exploration_phase))
        extra_columns = df.columns.difference(return_df.columns).tolist()
        merge_on_columns = [self.state_id_col, self.action_id_col]
        return_df = return_df.merge(df[extra_columns + merge_on_columns], on=merge_on_columns)
        return return_df

    def _get_decision(self, state_id: str, context_group: pd.GroupBy, is_exploration_phase: bool) -> pd.DataFrame:
        """Helper function for parallelizing decision method"""
        action = pd.DataFrame(columns=[])
        if is_exploration_phase:
            # randomly sampling without replacement is equivalent to a uniform random permutation
            if self.objective == "Profit":
                n_items, ids = context_group.shape[0], context_group[[self.action_id_col, self.action_weight_col]]
                rand_ids = ids.loc[np.random.choice(ids.shape[0], min(self._top_k, n_items), replace=False)].reset_index()
                action = pd.DataFrame(
                    {
                        self.action_id_col: rand_ids[self.action_id_col],
                        self.action_weight_col: rand_ids[self.action_weight_col],
                        self.rank_col: np.arange(len(rand_ids)),
                    }
                )

            else:
                n_items, ids = context_group.shape[0], context_group[self.action_id_col]
                rand_ids = np.random.choice(ids, min(self._top_k, n_items), replace=False)
                action = pd.DataFrame({self.action_id_col: rand_ids, self.rank_col: np.arange(len(rand_ids))})
        else:
            action = self._rank(context_group)

        action[self.state_id_col] = state_id
        return action

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid function: sigma(z) = 1 / (1 + e^{-z})"""
        return np.reciprocal(1 + np.exp(-z))

    def _update_action_param_map(self, df: pd.DataFrame) -> ParamUpdate:
        """Maintain map from action-id values to parameters and covariance matrices.

        The action_param_map is indexed on the action_id (a SKU or product ID) **name**, and the value
        of each element in the series is the **index** of the model parameters and covariance matrix
        in the covariance and parameter arrays. This allows us to map from the item/action names given by
        the client to their corresponding model parameters.

        Returns
        -------
        cov: np.ndarray
            Array containing covariance matrices for uncertainty estimation.
        params: np.ndarray
            Array of model parameters for per product logistic regression.
        """
        observed = set(self.action_param_map.index)
        new_observations = set(df[self.action_id_col])
        unobserved = new_observations.difference(observed)
        cov, params = self.cov, self.params

        n_new = len(unobserved)
        if 0 < n_new:
            max_idx = len(self.action_param_map)
            new_obs = pd.Series(max_idx + np.arange(n_new), index=unobserved)
            self.action_param_map = pd.concat([self.action_param_map, new_obs]).astype(int)

            new_cov = np.zeros(
                [
                    n_new,
                    self.feature_dim,
                    self.feature_dim,
                ]
            ) + (self.cov_lambda * np.eye(self.feature_dim))
            new_params = np.zeros([n_new, self.feature_dim])
            params = np.concatenate([params, new_params], axis=0)
            cov = np.concatenate([cov, new_cov], axis=0)

        return cov, params
