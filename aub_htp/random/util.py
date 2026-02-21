import numpy as np

def get_random_state_generator(random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        # bridge old API to new
        return np.random.default_rng(random_state.randint(0, 2**32 - 1))
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(random_state)
    raise TypeError("Invalid random_state argument")