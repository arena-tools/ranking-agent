import logging
from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ranking_bandit_agent import RankingBanditAgent

LOG: logging.Logger = logging.getLogger(__name__)

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)


class AbstractSimulator(ABC):
    """Abstract simulator class to test agents."""

    def __init__(self, agent: RankingBanditAgent, ground_truth_agent: RankingBanditAgent, num_items: int = 5):
        self.n_items = num_items
        self.n_positions = self.n_items
        self.context_dim = agent.context_dim
        self.feature_dim = agent.feature_dim - 1 # subtract the intercept term

        self.agent = agent
        self.ground_truth_agent = ground_truth_agent

        self.action_param_map: Dict[Any, Any] = {}

    @abstractmethod
    def setup_data(
        self,
    ) -> None:
        ...

    @abstractmethod
    def sample_batch(
        self,
        n_batches: int,
    ) -> np.ndarray:
        ...

    def evaluate(
        self,
        steps: int,
        batch_size: int,
        add_noise: float = 0,
        use_all_data: bool = False,
    ) -> List:
        """Runs through the evaluation procedure to test regret over time steps.

        Parameters
        ----------
        steps : int
            The number of steps to run the ranking agent.
        batch_size : int
            The number of samples to generate for each batch.
        add_noise: float
            The magnitude of the Gaussian noise to add to the probability of clicks.
        use_all_data : bool
            Whether to accumulate all historical data when updating the agent model.

        Returns
        -------
        regret : List
            The regret at each time step while training occured.
        """

        # reset the agent
        self.agent._init_model()
        self.ground_truth_agent._init_model()

        regret = []
        self.generated_data = np.zeros((self.n_items, 0, self.feature_dim + 1)) # [n_items, time, feature dim + clicks]
        self.batch_count = 0
        n_batches = [batch_size] * steps


        for batch_idx in range(steps):
            # sample context
            batch = self.sample_batch(n_batches[batch_idx])
            batch = batch[:, 1:] # removes items index
            batch, clicks = batch[:, :-2], batch[:, :-1] # drops reward and rank for batch, and separate clicks

            assert batch.shape[-1] == self.context_dim, "batch does not match context dim"

            # generate decisions, and then merge with batch data
            batch_rank = self.agent.rank(batch)

            # use model learned from sample data to compute the predicted ranking probabilities
            product_click_prob, click_prob = self._get_action_probability(batch, batch_rank)
            noisy_click_prob = product_click_prob + np.random.normal(
                loc=0.0, scale=add_noise, size=(product_click_prob.shape[0], 1)
            )
            clicks = np.random.binomial(1, p=np.clip(noisy_click_prob, 0, 1))
            batch = np.hstack([batch, np.expand_dims(batch_rank, -1), clicks])

            if use_all_data:
                self.generated_data = np.concatenate([self.generated_data, np.expand_dims(batch, 1)], axis=1)
                self.agent.update(self.generated_data[:, :, :-1], self.generated_data[:, :, -1])
            else:
                self.agent.update(batch)
            
            # generate true probabilities for all batch
            optimal_click_prob = self._get_optimal_probability(batch[:,:-2])

            time_step_regret = optimal_click_prob - click_prob
            regret += [time_step_regret / n_batches[batch_idx]]

            assert 0 <= time_step_regret, "Regret should be a non-negative number."
            if batch_idx % 50 == 0:
                print(f"Step {batch_idx} of {steps} regret = {regret[-1]:0.4f} Cumulative Regret = {np.sum(regret):0.4f}")
        return regret

    def _get_action_probability(self, current_batch: np.array, rank: Any) -> Tuple[np.ndarray, Dict[Any, float]]:
        """Helper function for computing the probability of the predicted action under the pre-fit model."""
        click_probabilities = self.get_probabilities(current_batch, rank)
        return click_probabilities, 1 - np.prod(1 - click_probabilities)

    def _get_optimal_probability(self, current_batch: np.ndarray) -> Dict[Any, float]:
        """Helper function for computing the probability of the optimal action under the pre-fit model."""
        true_prob = self.get_probabilities(current_batch)
        cost_matrix_optimal = -np.log(np.maximum(1 - true_prob, 1e-8))
        _, optimal_ranking = linear_sum_assignment(cost_matrix_optimal, maximize=True)

        optimal_prob = self.get_probabilities(current_batch, optimal_ranking.tolist())
        return 1 - np.prod(1 - optimal_prob)

    def get_probabilities(
        self,
        batch: np.array,
        rank: Optional[List[Number]] = None,
    ) -> np.ndarray:
        """Get probabilities of clicks based on ground truth parameters and context.
        (Taken from rank implimentation of Ranking Agent)

        Parameters
        ----------
        batch : np.ndarray
            The array for the current batch to compute probabilities for.
        rank : List (Optional)
            If provided, compute probability of clicking given the rankings, else compute a probability
            matrix for probability of clicking at each position.
        """
        features = np.hstack([np.ones((batch.shape[0], 1)), batch])

        # for each id, we need to evaluate its probability at position k
        n_products = self.n_items
        n_positions = n_products

        # user the param map to access the indexes for the parameter and covariance matrices
        params = self.ground_truth_agent.params

        # either evaluate all possible ranks, or only the ones provided by 'rank'
        if not rank is None:
            features = np.hstack([features, np.expand_dims(np.array(rank), 1)])
            n_positions = 1  # only evaluating one rank, so only one position per sku
        else:
            ranks = np.hstack([np.arange(n_positions) for _ in range(n_products)])  # [num_products * num_position]
            features = np.repeat(features, repeats=n_positions, axis=0)
            features = np.hstack([features, np.expand_dims(ranks, 1)])
            params = np.repeat(params, repeats=n_positions, axis=0)

        logit = (params * features).sum(axis=-1)
        return self.agent._sigmoid(logit.reshape(n_products, n_positions).astype(float))


class GenerativeSimulator(AbstractSimulator):
    """Simulator for generated data."""

    def setup_data(self) -> None:
        self.ground_truth_agent.params = self.generate_params()
        self.action_param_map = pd.Series({i: i for i in range(self.n_items)}, dtype=int)
        self.ground_truth_agent.action_param_map = self.action_param_map

    def _sample_from_norm_ball(self, dim: int, n_products: int = 1, n_batches: int = 1) -> np.ndarray:
        """Helper function to sample values from a ball of norm 1."""
        x = np.random.normal(0, 1, size=[n_batches, n_products, dim])
        return (x / np.expand_dims(np.linalg.norm(x, axis=-1), -1)).squeeze()
    
    def sample_batch(self, n_batches: int) -> np.array:
        """Sample batches for testing using values sampled from the norm ball for continuous features
        and 0-1 values for categorical features (if any).
        """

        # generate continuous context
        continuous_context = self._sample_from_norm_ball(self.context_dim, n_products=self.n_items, n_batches=n_batches)

        # generate item ids
        items = np.expand_dims(np.tile(np.arange(0, self.n_items), n_batches), axis=-1)
        ranking = np.zeros((self.n_positions * n_batches, 1))
        reward = np.zeros((self.n_positions * n_batches, 1))

        # add to batch data
        batches = np.hstack((items, continuous_context, ranking, reward))

        # update batch count
        self.batch_count += n_batches
        return batches
