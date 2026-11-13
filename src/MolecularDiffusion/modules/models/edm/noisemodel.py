import numpy as np
import torch
from typing import List 



def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_schedule(timesteps: int, nu_array: List[float], s: float = 1e-4):
    """
    A noise schedule based on the cosine schedule from Nichol & Dhariwal 2021.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)

    alphas2s = []
    for nu in nu_array:
        f = lambda t: np.cos(((t / steps)**nu + s) / (1 + s) * np.pi / 2) ** 2
        alphas2 = f(x)
        alphas2 = alphas2 / alphas2[0]  # Normalize so that alpha^2 at t=0 is 1
        alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

        alphas2 = alphas2.reshape(-1, 1)
        alphas2s.append(alphas2.reshape(-1, 1))
    alphas2s = np.concatenate(alphas2s, axis=1)
    return alphas2s



def polynomial_schedule(
    timesteps: int,
    powers: list = [2],
    s: float = 1e-4,
):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)

    alphas2s = []
    for power in powers:
        alphas2 = (1 - np.power(x / steps, power)) ** 2
        alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
        precision = 1 - 2 * s
        alphas2 = precision * alphas2 + s
        alphas2s.append(alphas2.reshape(-1, 1))
    alphas2s = np.concatenate(alphas2s, axis=1)
    return alphas2s


class NoiseModel:
    def __init__(
        self,
        timestep,
        noise_precision=1e-5,
        nu_arr=[0.5, 0.5, 0.5, 0.5],
        mapping=[
            "pos",
            "categorical",
            "integer",
            "extra",
        ],  # to key must be in the order of pos, categorical, integer
        device="cpu",
    ):

        self.mapping = mapping
        # There can be more keys in the mapping, e.g,. E, y
        # also assume h_extra is categorical
        self.inverse_mapping = {m: i for i, m in enumerate(self.mapping)}
        self.T = timestep
        self.nu_arr = nu_arr

        self._alpha2_bar = polynomial_schedule(
            self.T, powers=self.nu_arr, s=noise_precision
        )

        self._sigma2_bar = 1 - self._alpha2_bar

        log_alphas2 = np.log(self._alpha2_bar)
        log_sigmas2 = np.log(self._sigma2_bar)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self._gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        ).to(device)

        print("gamma", -log_alphas2_to_sigmas2)
        print("alpha2", self._alpha2_bar)
        self._alphas_bar = torch.sqrt(torch.sigmoid(-self._gamma)).to(device)
        self._sigma_bar = torch.sqrt(torch.sigmoid(self._gamma)).to(device)

    def get_alpha_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        a = self._alphas_bar.to(t_int.device)[t_int.long()]
        if key is None:
            return a.float()
        else:
            if "extra" in key:
                return a[..., 3:].float()
            else:
                return a[..., self.inverse_mapping[key]].float()

    def get_sigma_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        s = self._sigma_bar.to(t_int.device)[t_int]
        if key is None:
            return s.float()
        else:
            if "extra" in key:
                return s[..., 3:].float()
            else:
                return s[..., self.inverse_mapping[key]].float()

    def get_alpha_sigma_t(self, t_int):

        at = {}
        for key in self.mapping:
            at[key] = self.get_alpha_bar(t_int=t_int, key=key).unsqueeze(1)
        st = {}
        for key in self.mapping:
            st[key] = self.get_sigma_bar(t_int=t_int, key=key).unsqueeze(1)

        return at, st


try: 
    import matplotlib.pyplot as plt
    def plot_cosine_scheduler(timesteps, powers=[2], s=1e-4):
        alpha2_bar = cosine_schedule(timesteps, nu_array=powers, s=s)

        print(alpha2_bar.shape)
        sigma2_bar = 1 - alpha2_bar
        timesteps_arr = np.arange(timesteps + 1)
        log_alphas2 = np.log(alpha2_bar)
        log_sigmas2 = np.log(sigma2_bar)
        gamma = log_alphas2 - log_sigmas2
        print(gamma.shape)

        plt.figure(figsize=(8, 5))
        for i, power in enumerate(powers):
            plt.plot(timesteps_arr, gamma[:, i], label=f"power={power}")
        plt.xlabel("Timestep")
        plt.ylabel("Alpha Bar (ᾱₜ)")
        plt.title("Polynomial Noise Schedule: ᾱₜ vs Timestep")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_polynomial_scheduler(timesteps, powers=[2], s=1e-4):
        alpha2_bar = polynomial_schedule(timesteps, powers=powers, s=s)
        print(alpha2_bar.shape)
        sigma2_bar = 1 - alpha2_bar
        timesteps_arr = np.arange(timesteps + 1)
        log_alphas2 = np.log(alpha2_bar)
        log_sigmas2 = np.log(sigma2_bar)
        gamma = log_alphas2 - log_sigmas2
        print(gamma.shape)

        plt.figure(figsize=(8, 5))
        for i, power in enumerate(powers):
            plt.plot(timesteps_arr, gamma[:, i], label=f"power={power}")
        plt.xlabel("Timestep")
        plt.ylabel("Alpha Bar (ᾱₜ)")
        plt.title("Polynomial Noise Schedule: ᾱₜ vs Timestep")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
except ImportError:
    def plot_cosine_scheduler(timesteps, powers=[2], s=1e-4):
        print("matplotlib is not installed. Skipping cosine scheduler plot.")

    def plot_polynomial_scheduler(timesteps, powers=[2], s=1e-4):
        print("matplotlib is not installed. Skipping polynomial scheduler plot.")
