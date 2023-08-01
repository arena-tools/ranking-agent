from abc import ABC, abstractmethod
from collections import ChainMap
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ranking_bandit_agent import RankingBanditAgent

MAX_CONCURRENCY = 4
CLICK_PROBABILITY = "click_probability"

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)


class AbstractSimulator(ABC):
    """Abstract simulator class to test agents."""

    def __init__(self, agent: RankingBanditAgent, ground_truth_agent: RankingBanditAgent, num_items: int = 5):
        self.n_items = num_items
        self.top_k = agent._top_k
        self.n_positions = int(min(self.n_items, self.top_k))
        self.state_dim = len(agent.state_columns)

        if agent.objective == "Profit":
            self.revenue = True
        else:
            self.revenue = False
        self.revenue_col = agent.action_weight_col
        self.data_col_names = (
            agent.state_columns
            + [agent.state_id_col, agent.reward_col, agent.action_id_col, agent.rank_col, agent.action_weight_col]
        )
        self.feature_columns = agent.state_columns
        self.feature_dim = agent.feature_dim - 1
        self.agent = agent
        self.ground_truth_agent = ground_truth_agent
        self.data = pd.DataFrame()

        self.action_param_map: Dict[Any, Any] = {}

    @abstractmethod
    def setup_data(
        self,
        data_path: Any,
    ) -> None:
        ...

    @abstractmethod
    def sample_batch(
        self,
        n_batches: int,
    ) -> pd.DataFrame:
        ...

    def evaluate(
        self,
        steps: int,
        batch_size: int,
        data_path: Optional[str],
        add_noise: float = 0,
        use_all_data: bool = False,
    ) -> List[float]:
        """Runs through the evaluation procedure to test regret over time steps.

        Parameters
        ----------
        steps : int
            The number of steps to run the ranking agent.
        batch_size : int
            The number of samples to generate for each batch.
        data_path : str
            The path to the data to use for training.
            This is assumed to be a csv file please check this notebook
                http://us-lab.arena.tech:8888/lab/tree/notebooks/sam/Training_Dataframe.ipynb)
            for more information on how to generate this file.
        add_noise: float
            The magnitude of the Gaussian noise to add to the probability of clicks.
        use_all_data : bool
            Whether to accumulate all historical data when updating the agent model.
        learn_from_data : bool
            Whether data is provided. If so, use provided data to learn a set of parameters
            and if not, generate parameters for testing.

        Returns
        -------
        regret : List[float]
            The regret at each time step while training occured.
        """

        # reset the agent
        self.agent._init_model()
        self.ground_truth_agent._init_model()

        regret = []
        self.generated_data = pd.DataFrame(columns=self.data_col_names)
        self.batch_identifier = self.agent.state_id_col
        self.batch_count = 0

        # only recreate data and model the first time running simulation
        self.setup_data(data_path)

        for batch_idx in range(steps):
            # sample context
            batch_data_df = self.sample_batch(batch_size)

            # generate decisions, and then merge with batch data
            decide_df = batch_data_df[self.data_col_names].drop(columns=[self.agent.reward_col, self.agent.rank_col])
            batch_data_df = self.agent.rank(decide_df)

            # use model learned from sample data to compute the predicted ranking probabilities
            futures = self._run_parallel(batch_data_df, self._get_action_probability)
            all_click_probabilities = pd.concat([f[0] for f in futures]).reset_index()

            click_prob = dict(ChainMap(*[f[1] for f in futures]))
            click_prob = pd.DataFrame(click_prob.values(), columns=[CLICK_PROBABILITY], index=click_prob.keys())

            # compute reward. If add_nois > 0, then add Gaussian noise to the probability of clicks
            product_click_prob = all_click_probabilities[CLICK_PROBABILITY].to_numpy()
            noisy_click_prob = product_click_prob + np.random.normal(
                loc=0.0, scale=add_noise, size=product_click_prob.shape[0]
            )
            all_click_probabilities[self.agent.reward_col] = np.expand_dims(
                np.random.binomial(1, p=np.clip(noisy_click_prob.astype(float), 0, 1)), -1
            )

            batch_data_df = batch_data_df.merge(
                all_click_probabilities.drop([CLICK_PROBABILITY], axis=1),
                on=[self.agent.action_id_col, self.agent.state_id_col],
            )

            if use_all_data:
                self.generated_data = pd.concat([self.generated_data, batch_data_df])
                self.agent.update(self.generated_data[self.data_col_names])
            else:
                self.agent.update(batch_data_df[self.data_col_names])

            # generate true probabilities for all batch
            futures = self._run_parallel(batch_data_df, self._get_optimal_probability)
            optimal_click_prob = dict(ChainMap(*[f for f in futures]))
            optimal_click_prob = pd.DataFrame(
                optimal_click_prob.values(), columns=[CLICK_PROBABILITY], index=optimal_click_prob.keys()
            )

            time_step_regret = optimal_click_prob - click_prob
            regret += [np.sum(time_step_regret.to_numpy()) / batch_size]

            assert all(0 <= time_step_regret), "Regret should be a non-negative number."
            if batch_idx % 10 == 0:
                print(f"Step {batch_idx} of {steps} regret = {regret[-1]:0.4f} Cumulative Regret = {np.sum(regret):0.4f}")

        return regret

    def _run_parallel(
        self, df: pd.DataFrame, callable_fn: Callable, groupby_col: Optional[str] = None, *args: Any
    ) -> List[Any]:
        """helper function to excecute functions of a grouped df in parallel"""
        groupby_col = groupby_col or self.batch_identifier
        with ProcessPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
            futures = [executor.submit(callable_fn, batch, idx) for idx, batch in df.groupby(groupby_col)]
        return [future.result() for future in as_completed(futures)]

    def _get_action_probability(self, current_batch: pd.DataFrame, index: Any) -> Tuple[pd.DataFrame, Dict[Any, float]]:
        # get probability of learned ranking from true model

        click_probabilities = self.get_probabilities(current_batch, current_batch[self.agent.rank_col].tolist())
        batch_reward = np.array(
            [
                click_probabilities.squeeze(),
                np.zeros((len(click_probabilities))),
                current_batch[self.agent.action_id_col].to_numpy(),
                np.full((len(click_probabilities)), index),
            ]
        )
        batch_reward = pd.DataFrame(
            batch_reward.T,
            columns=[CLICK_PROBABILITY, self.agent.reward_col, self.agent.action_id_col, self.agent.state_id_col],
        )
        if self.revenue:
            click_probabilities = click_probabilities.squeeze() * current_batch[self.revenue_col]
            batch_learned_prob = np.sum(click_probabilities)
        else:
            batch_learned_prob = 1 - np.prod(1 - click_probabilities)
        return batch_reward, {index: batch_learned_prob}

    def _get_optimal_probability(self, current_batch: pd.DataFrame, index: int) -> Dict[Any, float]:
        """Helper function for computing the probability of the optimal action under the pre-fit model."""
        true_prob = self.get_probabilities(current_batch)
        cost_matrix_optimal = true_prob if self.revenue else -np.log(np.maximum(1 - true_prob, 1e-8))
        _, optimal_ranking = linear_sum_assignment(cost_matrix_optimal, maximize=True)
        optimal_prob = self.get_probabilities(current_batch, optimal_ranking.tolist())

        if self.revenue:
            optimal_prob = optimal_prob.squeeze() * current_batch[self.revenue_col]
            batch_optimal_prob = np.sum(optimal_prob)
        else:
            batch_optimal_prob = 1 - np.prod(1 - optimal_prob)

        return {index: batch_optimal_prob}

    def get_probabilities(
        self,
        batch: pd.DataFrame,
        rank: Optional[List[Number]] = None,
    ) -> np.ndarray:
        """Get probabilities of clicks based on ground truth parameters and context.
        (Taken from rank implimentation of Ranking Agent)

        Parameters
        ----------
        batch : pd.DataFrame
            The dataframe for the current batch to compute probabilities for.
        rank : List (Optional)
            If provided, compute probability of clicking given the rankings, else compute a probability
            matrix for probability of clicking at each position.
        """
        features = batch
        revenue = np.array(features[self.revenue_col]) if self.revenue else 1
        features = features[self.agent.state_columns].to_numpy()
        features = np.hstack([np.ones((features.shape[0], 1)), features])
        action_ids = batch[self.agent.action_id_col]

        # for each id, we need to evaluate its probability at position k
        n_products = action_ids.shape[0]
        n_positions = n_products

        # user the param map to access the indexes for the parameter and covariance matrices
        action_indexes = self.action_param_map[action_ids]
        params = self.ground_truth_agent.params[action_indexes.to_numpy().astype(int)]

        # either evaluate all possible ranks, or only the ones provided by 'rank'
        if rank:
            features = np.hstack([features, np.expand_dims(np.array(rank), 1)])
            n_positions = 1  # only evaluating one rank, so only one position per sku
            revenue = 1
        else:
            ranks = np.hstack([np.arange(n_positions) for _ in range(n_products)])  # [num_products * num_position]
            features = np.repeat(features, repeats=n_positions, axis=0)
            features = np.hstack([features, np.expand_dims(ranks, 1)])
            params = np.repeat(params, repeats=n_positions, axis=0)
            if self.revenue:
                revenue = np.repeat(revenue, n_positions).reshape(n_products, n_positions)

        logit = (params * features).sum(axis=-1)
        return self.agent._sigmoid(logit.reshape(n_products, n_positions).astype(float)) * revenue


class GenerativeSimulator(AbstractSimulator):
    """Simulator for generated data."""

    def generate_params(self, same_alpha: bool = False) -> np.ndarray:
        """Generate parameters for simulation if there are no data."""
        ground_truth_params = self._sample_from_norm_ball(self.feature_dim, self.n_items) / 5.0
        if same_alpha:
            return np.column_stack([np.random.rand(1).repeat(self.n_items), ground_truth_params])
        return np.column_stack([np.random.rand(self.n_items), ground_truth_params])

    def setup_data(self, data_path: str = "") -> None:
        self.ground_truth_agent.params = self.generate_params()
        self.action_param_map = pd.Series({i: i for i in range(self.n_items)}, dtype=int)
        self.ground_truth_agent.action_param_map = self.action_param_map

    def _sample_from_norm_ball(self, dim: int, n_products: int = 1, n_batches: int = 1) -> np.ndarray:
        """Helper function to sample values from a ball of norm 1."""
        x = np.random.normal(0, 1, size=[n_batches, dim]).repeat(n_products, axis=0)
        return x / np.expand_dims(np.linalg.norm(x, axis=1), 1)

    def sample_batch(self, n_batches: int) -> pd.DataFrame:
        """Sample batches for testing using values sampled from the norm ball for continuous features
        and 0-1 values for categorical features (if any).
        """
        batch_data_df = pd.DataFrame(columns=self.data_col_names)

        context = self._sample_from_norm_ball(self.state_dim, n_products=self.n_items, n_batches=n_batches)

        # generate batch_ids
        batch_ids = np.arange(self.batch_count, self.batch_count + n_batches)
        batch_ids = np.expand_dims(batch_ids, 0).repeat(self.n_positions, axis=0).flatten("F")
        batch_ids = np.expand_dims(batch_ids, -1)

        # generate item ids
        items = np.expand_dims(np.tile(np.arange(0, self.n_items), n_batches), axis=-1)
        ranking = np.zeros((self.n_positions * n_batches, 1))
        reward = np.zeros((self.n_positions * n_batches, 1))

        # add to batch data
        if self.revenue:
            revenue = np.random.choice(20, size=(n_batches * self.n_positions, 1))
            batches = np.hstack((context, batch_ids, reward, items, ranking, revenue))
        else:
            batches = np.hstack((context, batch_ids, reward, items, ranking))
        batch_data_df = pd.DataFrame(batches, columns=self.data_col_names)

        # update batch count
        self.batch_count += n_batches
        return batch_data_df

class DatasetSimulator(AbstractSimulator):
    """Simulator for generated data."""

    def _preprocess_batches(self, data: pd.DataFrame) -> pd.DataFrame:
        """enforce quality constraints on samples before using them in the training procedure"""

        def _check_batch(sample: pd.DataFrame, top_k: int, action_id_col: str) -> bool:
            "helper function to check data quality of each batch"
            has_at_least_k = top_k <= len(sample)
            return has_at_least_k

        is_valid_batch = data.groupby(self.batch_identifier).apply(
            _check_batch, top_k=self.top_k, action_id_col=self.agent.action_id_col
        )
        valid_batch = is_valid_batch[is_valid_batch]
        is_valid_row = self.data[self.batch_identifier].isin(valid_batch.index)
        LOG.info(
            f"Filtered Invalid Data: Removed {100*sum(~is_valid_row)/len(is_valid_row):0.2f}% of available Data. "
            f"({sum(~is_valid_row)} of {len(is_valid_row)} available rows)"
        )
        return self.data[is_valid_row]

    @cache
    def load_and_process_data(self, data_path: str) -> pd.DataFrame:
        """Read in and preprocess the data before evaluating the ranking agent. Also sets the data attributes
        n_items, action_param_map, and n_positions.

        Parameters
        ----------
            data_path: file path to read in the sample training/evalutaion data
        """

        LOG.info(f"Reading Data from: {data_path}")
        data = pd.read_csv(data_path, delimiter=",")

        n_rows = data.shape[0]
        data = data.drop_duplicates(subset=[self.batch_identifier, self.agent.action_id_col], keep="first")
        n_rows_new = data.shape[0]
        LOG.info(
            f"Removed duplicated Data: Removed {100*(n_rows - n_rows_new)/n_rows:0.2f}% of available Data. "
            f"({(n_rows - n_rows_new)} of {n_rows} available rows)"
        )

        self.data = data[self.data_col_names]
        action_ids = self.data[self.agent.action_id_col].tolist()

        items = list(set(action_ids))
        self.n_items = len(items)
        self.action_param_map = pd.Series({items[i]: i for i in range(self.n_items)}, dtype=int)
        self.n_positions = int(min(self.n_items, self.top_k))
        return self.data.sort_values(by=[self.batch_identifier, self.agent.action_id_col])

    def setup_data(self, data_path: str) -> None:
        # provided data to learn ground truth params
        self.data = self.load_and_process_data(data_path)
        self.ground_truth_agent.fit(self.data[self.data_col_names])
        self.data = self._preprocess_batches(self.data)

    def sample_batch(self, n_batches: int = 1) -> pd.DataFrame:
        """Samples n_batches of data from self.data."""
        batch_indices = np.random.choice(self.data[self.batch_identifier].unique(), size=n_batches, replace=False)
        sample = self.data[self.data[self.batch_identifier].isin(batch_indices)]
        return sample
