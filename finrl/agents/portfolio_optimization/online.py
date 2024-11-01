from stable_baselines3.common.type_aliases import GymEnv
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd

class CRPModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            target_weights: List[float] = None # If none, default to uniform weights
            ) -> None:
        
        # Super simple algorithm, we only need the environment
        # This environment needs to have prices for the CRP algorithm TODO check the type here

        assert env is not None 
        self.env = env

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, (defualt to uniform)
        # Uniform base case
        # Note these are the first weight reprsents the cash account, which should always be 0
        self.target_weights = np.ones(self.portfolio_length-1) / (self.portfolio_length-1)  # target weights for each asset
        # Append 0 to the beginning, for an empty cash account
        self.target_weights = np.insert(self.target_weights, 0, 0)
        
        if target_weights is not None:
            # Assert that the portfolio length matches (index 0 represents cash account)
            assert len(target_weights) == self.portfolio_length
            self.target_weights = np.array(target_weights)

    def train(self) -> None:
        # TODO this model is derministic and doesnt learn anything, it only predicts
        pass

    def learn(
        self
    ):
        # TODO this model is derministic and doesnt learn anything, it only predicts
        pass

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, # TODO not needed this is always determininistic
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # TODO much of this comes from the policies class in stable baselines
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        
        # We always just return the target CRP weights
        actions = self.target_weights.reshape(1, self.portfolio_length)

        # The state doesnt matter here
        return actions, None
    
# todo MEASURE THE REGRET

class BAHModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            target_weights: List[float] = None # If none, default to uniform weights
            ) -> None:
        
        # Super simple algorithm, we only need the environment
        # This environment needs to have prices for the BAH algorithm TODO check the type here

        assert env is not None 
        self.env = env

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, (defualt to uniform)
        # Uniform base case
        # Note these are the first weight reprsents the cash account, which should always be 0
        self.target_weights = np.ones(self.portfolio_length-1) / (self.portfolio_length-1)  # target weights for each asset
        # Append 0 to the beginning, for an empty cash account
        self.target_weights = np.insert(self.target_weights, 0, 0)

        print(self.target_weights)
        
        if target_weights is not None:
            # Assert that the portfolio length matches (index 0 represents cash account)
            assert len(target_weights) == self.portfolio_length
            self.target_weights = np.array(target_weights)

    def train(self) -> None:
        # TODO this model is derministic and doesnt learn anything, it only predicts
        pass

    def learn(
        self
    ):
        # TODO this model is derministic and doesnt learn anything, it only predicts
        pass

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, # TODO not needed this is always determininistic
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # TODO much of this comes from the policies class in stable baselines
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        
        # For BAH we just use whatever weights are already in the environment.

        # For the first step we need to do the initial buy, after that we just let the portfolio run

        # If we are on first time-step, buy the portfolio
        actions = self.target_weights.reshape(1, self.portfolio_length)

        # else 
        if len(self.env._actions_memory) > 1:
            # Use the last portfolio as the new action (keep it the same)
            actions = self.env._final_weights[-1].reshape(1, self.portfolio_length)

        # The state doesnt matter here
        return actions, None
    
class BCRPModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            ) -> None:
        
        # Super simple algorithm, we only need the environment
        # This environment needs to have prices for the BCRP algorithm TODO check the type here

        assert env is not None 
        self.env = env

        # This portoflio cheats by pulling the full price range ange getting the best portfolio weights in hindsight
        self._full_hindsight_prices = self.env._df

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, for BCRP we use hinesight to get the best possible weights over the time range
        # Here we will cheat and calculate what the best weights would have been
        # This is obviously a benchmark metric and does not work in reality (because we can't see into the future)
        
        # Pivot the DataFrame 
        pivoted_df = self._full_hindsight_prices.pivot(index='date', columns='tic', values='close') 
        # Calculate price ratios 
        price_ratios = pivoted_df / pivoted_df.iloc[0]
        # Get the magic weights
        self.target_weights = np.array(optimize_with_hindsight(price_ratios))
        # Assume no cash
        self.target_weights = np.insert(self.target_weights, 0, 0)

    def train(self) -> None:
        # TODO this model is derministic and doesnt learn anything, it only predicts
        pass

    def learn(
        self
    ):
        # TODO this model is derministic and doesnt learn anything, it only predicts
        pass

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, # TODO not needed this is always determininistic
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # TODO much of this comes from the policies class in stable baselines
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        
        # We always just return the target BCRP weights
        actions = self.target_weights.reshape(1, self.portfolio_length)

        # The state doesnt matter here
        return actions, None

# TODO - momentum type strategies

# TODO - look into regret minimuzation

class OLMARModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            target_weights: List[float] = None, # If none, default to uniform weights
            window=5, 
            eps=10, 
            alpha=0.5
            ) -> None:
        
        # Super simple algorithm, we only need the environment
        # This environment needs to have prices for the OLMAR algorithm TODO check the type here

        assert env is not None 
        self.env = env

        self.window = window
        self.eps = eps
        self.alpha = alpha

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, (defualt to uniform)
        # Uniform base case
        # Note these are the first weight reprsents the cash account, which should always be 0
        self.current_weights = np.ones(self.portfolio_length-1) / (self.portfolio_length-1)  # target weights for each asset

        # For OLMAR start with uniform and then adjust based on moving averages
        self.price_history = pd.DataFrame()

    def train(self) -> None:
        # TODO this model is derministic and doesnt learn anything, it only predicts
        pass

    def learn(
        self
    ):
        # TODO this model is derministic and doesnt learn anything, it only predicts
        pass

    def get_price_relative_SMA(self):
        """Predict next price relative using SMA."""
        return self.price_history.mean() / self.price_history.iloc[-1, :]
        
    def update_weights(self, weights, new_price_prediction, eps):
        """Update portfolio weights to satisfy constraint weights * x >= eps
        and minimize distance to previous weights."""
        price_prediction_mean = np.mean(new_price_prediction)
        excess_return = price_prediction_mean - new_price_prediction
        denominator = (excess_return * excess_return).sum()
        if denominator != 0:
            constraint = ((np.dot(weights, new_price_prediction)) - eps)
            lam = max(0.0, constraint / denominator)
        else:
            lam = 0

        # update portfolio
        weights = weights - lam * (excess_return)

        # project it onto simplex
        return simplex_proj(weights)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, # TODO not needed this is always determininistic
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # TODO much of this comes from the policies class in stable baselines
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        # Reshape the array to remove single dimensions 
        reshaped_array = observation.reshape(len(self.env._features), self.portfolio_length - 1) 

        # TODO this code his horrible Extract the three lists
        prices = reshaped_array[0].tolist()

        new_row = pd.DataFrame([prices])

        # Add to the price history
        self.price_history = pd.concat([self.price_history, new_row], ignore_index=True)
        old_weights = self.current_weights


        # Window is too short, return the starting weights
        if len(self.price_history) < self.window + 1:
            self.price_prediction = self.price_history.iloc[-1]

            # Use the last portfolio as the new action (keep it the same)
            action_weights = np.insert(old_weights, 0, 0)
            actions = action_weights.reshape(1, self.portfolio_length)

            # print("Here!!!!!!!")
            # print(actions)

            # The state doesnt matter here
            return actions, None

        else:
            self.price_prediction = self.get_price_relative_SMA()
            new_weights = self.update_weights(old_weights, self.price_prediction, self.eps)

            self.current_weights = new_weights

            # Use the last portfolio as the new action (keep it the same)
            action_weights = np.insert(new_weights, 0, 0)
            actions = action_weights.reshape(1, self.portfolio_length)

            return actions, None
    
# TODO form the universal protfolio algoirthm
def simplex_proj(y):
    """Projection of y onto simplex."""
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.0

    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    return np.maximum(y - tmax, 0.0)


import scipy.optimize as optimize
# TODO found this here:  https://github.com/Marigold/universal-portfolios/blob/master/universal/tools.py
def optimize_with_hindsight(
    prices
):
    assert prices.notnull().all().all()

    x_0 = np.ones(prices.shape[1]) / float(prices.shape[1])
    
    objective = lambda b: -np.sum(np.log(np.maximum(np.dot(prices - 1, b) + 1, 0.0001)))

    cons = ({"type": "eq", "fun": lambda b: 1 - sum(b)},)

   
    # problem optimization
    res = optimize.minimize(
        objective,
        x_0,
        bounds=[(0.0, 1.0)] * len(x_0),
        constraints=cons,
        method="slsqp"
    )

    if res.success:
        return res.x
    raise ValueError("Could not find an optimal value using the BCRP algorithm.")