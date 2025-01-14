from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import joblib  # For saving and loading models

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import joblib

class ProjectAgent:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        gamma=0.99, 
        iterations=5,  # Adjust based on performance vs. size
        n_estimators=50,  # Fewer trees reduce model size
        max_depth=10,  # Shallower trees for smaller models
        min_samples_split=10  # More samples required to split a node
    ):
        """
        Initialize the ProjectAgent with optimized parameters to reduce model size.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.iterations = iterations
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.Qfunction = None  # Single Q-function
    
    def collect_samples(self, env, horizon, disable_tqdm=False, print_done_states=False):
        """
        Collect samples by interacting with the environment using a random policy.
        """
        S, A, R, S2, D = [], [], [], [], []
        s, _ = env.reset()
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            S.append(s.astype(np.float32))  # Use float32 to reduce memory
            A.append(a)
            R.append(r)
            S2.append(s2.astype(np.float32))
            D.append(done or trunc)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("Episode ended.")
            else:
                s = s2
        return (
            np.array(S, dtype=np.float32),
            np.array(A).reshape(-1, 1),
            np.array(R, dtype=np.float32),
            np.array(S2, dtype=np.float32),
            np.array(D, dtype=np.float32),
        )
    
    def train_fqi(self, S, A, R, S2, D):
        """
        Perform a single iteration of Fitted Q Iteration.
        """
        if self.Qfunction is None:
            # Initialize Q-values with immediate rewards
            Y = R.copy()
        else:
            # Estimate target Q-values using the current Q-function
            Q_next = np.zeros((S2.shape[0], self.action_dim))
            for a in range(self.action_dim):
                A2 = np.full((S2.shape[0], 1), a)
                S2A2 = np.hstack([S2, A2])
                Q_next[:, a] = self.Qfunction.predict(S2A2)
            max_Q_next = np.max(Q_next, axis=1)
            Y = R + self.gamma * (1 - D) * max_Q_next
        
        # Prepare state-action pairs
        SA = np.hstack([S, A])
        
        # Train the Random Forest regressor
        Q = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        Q.fit(SA, Y)
        self.Qfunction = Q  # Overwrite with the latest Q-function
    
    def train(self, env, horizon_per_iteration=10000, disable_tqdm=False, print_done_states=False):
        """
        Train the agent using Fitted Q Iteration.
        """
        for iter_num in tqdm(range(self.iterations), desc="FQI Iterations", disable=disable_tqdm):
            S, A, R, S2, D = self.collect_samples(
                env, 
                horizon=horizon_per_iteration, 
                disable_tqdm=disable_tqdm, 
                print_done_states=print_done_states
            )
            self.train_fqi(S, A, R, S2, D)
    
    def act(self, observation, use_random=False, epsilon=0.1):
        """
        Select an action based on the current policy using the latest Q-function.
        """
        if use_random or self.Qfunction is None or np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            Q_values = np.zeros(self.action_dim)
            for a in range(self.action_dim):
                SA = np.hstack([observation, a]).reshape(1, -1)
                Q_values[a] = self.Qfunction.predict(SA)[0]
            return np.argmax(Q_values)
    
    def save(self, path):
        """
        Save the trained Q-function to disk with compression.
        """
        joblib.dump(self.Qfunction, path, compress=3)  # Adjust compression level as needed
    
    def load(self, path):
        """
        Load the Q-function from disk.
        """
        self.Qfunction = joblib.load(path)
