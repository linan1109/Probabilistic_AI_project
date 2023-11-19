"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
RANDOM_STATE = 0

class Queue:
    def __init__(self):
        self._x = None
        self._f = np.array([])
        self._v = np.array([])

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._x)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        x = self._x[0]
        f = self._f[0]
        v = self._v[0]
        self._x = self._x[1:]
        self._f = self._f[1:]
        self._v = self._v[1:]
        return x, f, v

    def add(self, x, f, v):
        """Add object to end of queue."""
        if self._x is None:
            self._x = np.atleast_2d(x)
        else:
            self._x = np.vstack([self._x, x])
        self._f = np.append(self._f, f)
        self._v = np.append(self._v, v)


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        #  RBF kernel with variance 0.5 and length-scale 1
        self._kernel_f = Matern(nu=2.5) 
        # 0.5 * RBF(length_scale=1.0) 
        self._f_sigma = 0.15
        self._f = GaussianProcessRegressor(
            kernel=self._kernel_f, 
            alpha= self._f_sigma**2,
            random_state=RANDOM_STATE)
        
        

        self._kernel_v = Matern(nu=2.5) + DotProduct(sigma_0=0)
        # (2**0.5) * RBF(length_scale=1.0)
        self._v_sigma = 0.0001
        self.x_sample = []
        self._v = GaussianProcessRegressor(
            kernel=self._kernel_v, 
            alpha= self._v_sigma**2,
            random_state=RANDOM_STATE)
        
        self._queue = Queue()
        self._beta = 3
        

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        # if(len(self.x_sample) == 0):
        #     return get_initial_safe_point()
        
        x_opt = self.optimize_acquisition_function()
        x_opt = np.atleast_2d(x_opt)
        return x_opt

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        mean, std = self._f.predict(x, return_std=True)
        sa = self._v.predict(x)
        res = mean + self._beta * std if sa < SAFETY_THRESHOLD else  -10086
        return res
        

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        # select the data point that has max uncertainty
        
        x = np.atleast_2d(x)
        self._queue.add(x, f, v)
        self._f.fit(self._queue._x, self._queue._f)
        self._v.fit(self._queue._x, self._queue._v)
        
        
    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
        f_max = -np.inf
        x_opt = 0
        for x in x_domain:
            mean, std = self._f.predict([x], return_std=True)
            sa = self._v.predict([x])
            if sa < SAFETY_THRESHOLD and mean + self._beta * std > f_max:
                f_max = mean + self._beta * std
                x_opt = x
        return x_opt

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn() 
        cost_val = v(x) + np.random.randn() 
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
