import math

from torch import inf


class CosineAnnealing:
    def __init__(
        self,
        T_max,
        eta_min=0,
        last_epoch=-1,
    ):
        """
        Args:
            T_max (int): Maximum number of iterations.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        if T_max <= 0:
            raise ValueError("Expected positive T_max, but got {}".format(T_max))
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self._reset()

    def _reset(self):
        """Resets last_epoch counter."""
        self.last_epoch = -1

    def step(self, metrics, rate):
        """
        Perform a step to update the learning rate.

        Args:
            rate (float): Current learning rate.

        Returns:
            float: Updated learning rate.
        """
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch > self.T_max:
            return self.eta_min

        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.T_max))
        new_rate = self.eta_min + (rate - self.eta_min) * cosine_decay
        return new_rate


class ReduceLROnPlateau:

    def __init__(
        self,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        eps=1e-8,
    ):

        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, rate):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            return rate
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown
            return rate
        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return rate * self.factor
        else:
            return rate

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

