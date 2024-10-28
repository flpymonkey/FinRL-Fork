from stable_baselines3.common.type_aliases import GymEnv
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np

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
        self.target_weights = np.array(opt_weights(price_ratios))
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


import scipy.optimize as optimize
# TODO found this here:  https://github.com/Marigold/universal-portfolios/blob/master/universal/tools.py
def opt_weights(
    X,
    metric="return",
    max_leverage=1,
    rf_rate=0.0,
    alpha=0.0,
    freq: float = 252,
    no_cash=False,
    sd_factor=1.0,
    **kwargs,
):
    """Find best constant rebalanced portfolio with regards to some metric.
    :param X: Prices in ratios.
    :param metric: what performance metric to optimize, can be either `return` or `sharpe`
    :max_leverage: maximum leverage
    :rf_rate: risk-free rate for `sharpe`, can be used to make it more aggressive
    :alpha: regularization parameter for volatility in sharpe
    :freq: frequency for sharpe (default 252 for daily data)
    :no_cash: if True, we can't keep cash (that is sum of weights == max_leverage)
    """
    assert metric in ("return", "sharpe", "drawdown", "ulcer")
    assert X.notnull().all().all()

    x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
    if metric == "return":
        objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)))
    # elif metric == "sharpe":
    #     objective = lambda b: -sharpe(
    #         np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)),
    #         rf_rate=rf_rate,
    #         alpha=alpha,
    #         freq=freq,
    #         sd_factor=sd_factor,
    #     )
    elif metric == "drawdown":

        def objective(b):
            R = np.dot(X - 1, b) + 1
            L = np.cumprod(R)
            dd = max(1 - L / np.maximum.accumulate(L))
            annual_ret = np.mean(R) ** freq - 1
            return -annual_ret / (dd + alpha)

    if no_cash:
        cons = ({"type": "eq", "fun": lambda b: max_leverage - sum(b)},)
    else:
        cons = ({"type": "ineq", "fun": lambda b: max_leverage - sum(b)},)

    while True:
        # problem optimization
        res = optimize.minimize(
            objective,
            x_0,
            bounds=[(0.0, max_leverage)] * len(x_0),
            constraints=cons,
            method="slsqp",
            **kwargs,
        )

        # result can be out-of-bounds -> try it again
        EPS = 1e-7
        if (res.x < 0.0 - EPS).any() or (res.x > max_leverage + EPS).any():
            X = X + np.random.randn(1)[0] * 1e-5
            print("Optimal weights not found, trying again...")
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                print("Solution does not exist, use zero weights.")
                res.x = np.zeros(X.shape[1])
            else:
                print("Converged, but not successfully.")
            break

    return res.x