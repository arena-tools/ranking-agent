# UCR: Upper Confidence Ranking Bandit Agent

A simple implementation of the UCR algorithm, which is an active learning agent that learns to rank.

This agent models the probability of clicking on each item as a logistic
function and maximizes the upper confidence bound maximum likelihood estimate of
the probability of user clicking on an item given their ranking over the course
of training.

An assumption is made that the positional effects of each item is different, and
that there is a monotonic relationship between the rank and probability of click
which is why the weight parameter for positional effects in one scalar.

---

## Setup:

### Setup General Environment

Install Anaconda at https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Create a conda environment using the environment.yml file in this repository with

- `conda env create -f environment.yml`

## Tests:

### UCR Agent Regret Tests

Tests are written using pytests and can be run with the following command.

```
$ pytest test_ranking_agent.py::RankingBanditAgentSimulationTests::test_regret_generative -s
```

The test_regret_generative method runs the Ranking Agent on a generated simulation environment and calculates regret at each step by comparing with the optimal click probabilities. The adjustable parameters are

runs: number of runs to average over
steps: horizon of each run
n_items: number of items to rank
ksi_list: the hyperparameters to run the ranking agent over
reward_noise: the noise added to the click probabilities when generating clicks
batch_size: batch size for each time step

## Repository Structure

ranking_bandit_agent.py contains the main RankingBanditAgent class which is a model of the agent.

simulator.py contains the simulation environment which handles generating the environment and runs through the evaluation procedure for the ranking agent.

test_ranking_agent.py is the pytest file used to run the agent in the simulator.
