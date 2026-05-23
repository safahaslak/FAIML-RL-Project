import gymnasium as gym
import numpy as np

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(1.0, 1.0),
        mode="none",
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range

        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):
        if self.mode == "none":
            return None
        elif self.mode == "udr":
            # UDR (Uniform Domain Randomization): Randomly select the mass within a given minimum and maximum range.
            return np.random.uniform(self.mass_min_limit, self.mass_max_limit)
        else:
            raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented yet.")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        # Optionally, you can add here extra logic
        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------

    def reset(self, **kwargs):
        # TODO: sample new mass
        # Randomly sample the mass using the _sample_mass function
        new_mass = self._sample_mass() 

        if new_mass is not None:
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min_limit:.2f},{self.mass_max_limit:.2f}]"
            )

        return super().reset(**kwargs)
