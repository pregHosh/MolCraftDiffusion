import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

class gradient_clipping:
    def __init__(self, m=1, max_len=200):
        self.max_grad_norm = None
        self.max_grad_norms = []
        self.max_len = max_len
        self.m = m
        self.FACTOR = 100

    def __call__(self, model, gradnorm_queue):
        self.max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()
        if len(self.max_grad_norms) == 0:
            self.max_grad_norms.append(self.max_grad_norm)
        else:
            #max_grad_norm_mean = torch.mean(torch.tensor(self.max_grad_norms))
            previous_max_grad_norm = self.max_grad_norms[-1]
            # if the current max_grad_norm is greater than the mean of the previous max_grad_norms
            if self.max_grad_norm > previous_max_grad_norm:
                self.max_grad_norm = previous_max_grad_norm * self.m
                if self.max_grad_norm > previous_max_grad_norm * 1e5:
                    self.max_grad_norm = previous_max_grad_norm * self.m / self.FACTOR

            self.max_grad_norms.append(self.max_grad_norm)

        if len(self.max_grad_norms) > self.max_len:
            self.max_grad_norms.pop(0)
        # Clips gradient and returns the norm

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.max_grad_norm, norm_type=2.0
        )
        if float(grad_norm) > self.max_grad_norm:
            gradnorm_queue.add(float(self.max_grad_norm))
        else:
            gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > self.max_grad_norm:
            logger.info(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {self.max_grad_norm:.1f}"
            )
        return grad_norm
    
class gradient_clipping_0:
    def __init__(self, m=1, max_len=200):
        self.max_grad_norm = None
        self.max_grad_norms = []
        self.max_len = max_len
        self.m = m

    def __call__(self, model, gradnorm_queue):
        self.max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()
        if len(self.max_grad_norms) == 0:
            self.max_grad_norms.append(self.max_grad_norm)
        else:
            max_grad_norm_mean = torch.mean(torch.tensor(self.max_grad_norms))
            if self.max_grad_norm > max_grad_norm_mean:
                self.max_grad_norm = max_grad_norm_mean * self.m
                if self.max_grad_norm > max_grad_norm_mean * 1e5:
                    self.max_grad_norm = max_grad_norm_mean * self.m / 10
            self.max_grad_norms.append(self.max_grad_norm)

        if len(self.max_grad_norms) > self.max_len:
            self.max_grad_norms.pop(0)
        # Clips gradient and returns the norm

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.max_grad_norm, norm_type=2.0
        )
        if float(grad_norm) > self.max_grad_norm:
            gradnorm_queue.add(float(self.max_grad_norm))
        else:
            gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > self.max_grad_norm:
            print(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {self.max_grad_norm:.1f}"
            )
        return grad_norm


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class SP_regularizer:
    def __init__(
        self,
        regularizer: str,
        lambda_: float = 10,
        lambda_2: float = 100,
        lambda_update_value: float = 50,
        lambda_update_step: int = 2500,
        polynomial_p: float = 1.5,
        warm_up_steps: int = 100,
    ):
        """
        Self-paced regularizer for curriculum learning
        Args:
            regularizer (str): Regularizer to use. Options are:
                - hard
                - linear
                - logaritmic
                - logistic
            lambda_ (float): Initial lambda value
            lambda_2 (float): Initial lambda value for the second regularizer
            lambda_update_value (float): Value to update lambda
            lambda_update_step (int): Number of steps to update lambda
            polynomial_p (float): Value of p for polynomial regularizer
            warm_up_steps (int): Number of steps to use the regularizer
        """

        self.regularizer = regularizer
        self.lambda_ = lambda_
        self.lambda_2 = lambda_2
        self.n_calls = 1
        self.lambda_update_value = lambda_update_value
        self.lambda_update_step = lambda_update_step
        self.p = polynomial_p
        self.warm_up_steps = warm_up_steps

    def __call__(self, losses: torch.Tensor):

        # TODO during warm up steps, keep the losses infomation, to be used to determine lambda
        if self.n_calls < self.warm_up_steps:
            self.n_calls += 1
            return losses
        else:
            if self.regularizer == "hard":
                weighted_loss = self.hard(losses)
            elif self.regularizer == "linear":
                weighted_loss = self.linear(losses)
            elif self.regularizer == "logaritmic":
                weighted_loss = self.logaritmic(losses)
            elif self.regularizer == "logistic":
                weighted_loss = self.logistic(losses)
            elif self.regularizer == "polynomial":
                weighted_loss = self.polynomial(losses)
            elif self.regularizer == "hard_relax":
                weighted_loss = self.hard_relax(losses)
            else:
                raise ValueError("Regularizer not implemented")
            self.n_calls += 1
            self.update_lambda()
            return weighted_loss

    def update_lambda(self):
        if self.n_calls % self.lambda_update_step == 0:
            self.lambda_ += self.lambda_update_value
            self.lambda_2 += self.lambda_update_value
        elif self.n_calls == 0:
            self.lambda_ = self.lambda_
            self.lambda_2 = self.lambda_2

    def hard(self, losses: torch.Tensor):

        weights = (losses <= self.lambda_).float()
        sp_loss = losses * weights

        return sp_loss

    def hard_relax(self, losses: torch.Tensor):
        weights = torch.where(
            losses < self.lambda_,
            torch.ones_like(losses),
            (1 - losses / self.lambda_2) ** (1 / (self.p - 1)),
        )
        idces_zero = torch.where(losses > self.lambda_2)
        weights[idces_zero] = 0
        weights = torch.clamp(weights, 0, 1)
        sp_loss = losses * weights

        return sp_loss

    def linear(self, losses: torch.Tensor):
        weights = torch.where(
            losses > self.lambda_, torch.zeros_like(losses), 1 - losses / self.lambda_
        )
        weights = torch.clamp(weights, 0, 1)
        sp_loss = losses * weights

        return sp_loss

    def logaritmic(self, losses: torch.Tensor):

        weights = torch.where(
            losses > self.lambda_,
            torch.zeros_like(losses),
            torch.log(2 - losses / self.lambda_),
        )
        weights = torch.clamp(weights, 0, 1)
        sp_loss = losses * weights

        return sp_loss

    def logistic(self, losses: torch.Tensor):

        weights = torch.where(
            losses > self.lambda_,
            torch.zeros_like(losses),
            (1 - torch.exp(torch.tensor(self.lambda_)))
            / (1 - torch.exp(losses - self.lambda_)),
        )
        weights = torch.clamp(weights, 0, 1)
        sp_loss = losses * weights

        return sp_loss

    def polynomial(self, losses: torch.Tensor):

        weights = torch.where(
            losses > self.lambda_,
            torch.zeros_like(losses),
            (1 - losses / self.lambda_) ** (1 / (self.p - 1)),
        )
        weights = torch.clamp(weights, 0, 1)
        sp_loss = losses * weights

        return sp_loss

