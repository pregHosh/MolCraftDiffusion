import pprint
from itertools import islice, chain
from omegaconf import DictConfig
from typing import  Any, Callable, Dict, Tuple
from .io import RankedLogger
from importlib.util import find_spec

log = RankedLogger(__name__, rank_zero_only=True)

separator = ">" * 30
line = "-" * 30


class Ellipsis(object):

    def __repr__(self):
        return "..."


ellipsis = Ellipsis()


class PrettyPrinter(pprint.PrettyPrinter):

    truncation = 10
    display = 3

    def _format_items(self, items, stream, indent, allowance, context, level):
        if self._compact and len(items) > self.truncation:
            items = chain(islice(items, self.display), [ellipsis], islice(items, len(items) - self.display, None))
        super(PrettyPrinter, self)._format_items(items, stream, indent, allowance, context, level)


def print(object, *args, **kwargs):
    """
    Print a python object to a stream.
    """
    return PrettyPrinter(*args, **kwargs).pprint(object)


def format(object, *args, **kwargs):
    """
    Format a python object as a string.
    """
    return PrettyPrinter(*args, **kwargs).pformat(object)


def time(seconds):
    """
    Format time as a string.

    Parameters:
        seconds (float): time in seconds
    """
    sec_per_min = 60
    sec_per_hour = 60 * 60
    sec_per_day = 24 * 60 * 60

    if seconds > sec_per_day:
        return "%.2f days" % (seconds / sec_per_day)
    elif seconds > sec_per_hour:
        return "%.2f hours" % (seconds / sec_per_hour)
    elif seconds > sec_per_min:
        return "%.2f mins" % (seconds / sec_per_min)
    else:
        return "%.2f secs" % seconds


def long_array(array, truncation=10, display=3):
    """
    Format an array as a string.

    Parameters:
        array (array_like): array-like data
        truncation (int, optional): truncate array if its length exceeds this threshold
        display (int, optional): number of elements to display at the beginning and the end in truncated mode
    """
    if len(array) <= truncation:
        return "%s" % array
    return "%s, ..., %s" % (str(array[:display])[:-1], str(array[-display:])[1:])


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap