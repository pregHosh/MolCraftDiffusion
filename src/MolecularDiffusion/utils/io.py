import os
import sys
import ast
import tempfile
from contextlib import contextmanager
import logging
from typing import Mapping, Optional
from rdkit import RDLogger


def input_choice(prompt, choice=("y", "n")):
    """
    Print a prompt on the command line and wait for a choice.

    Parameters:
         prompt (str): prompt string
         choice (tuple of str, optional): candidate choices
    """
    prompt = "%s (%s)" % (prompt, "/".join(choice))
    choice = set([c.lower() for c in choice])
    result = input(prompt)
    while result.lower() not in choice:
        result = input(prompt)
    return result


def literal_eval(string):
    """
    Evaluate an expression into a Python literal structure.
    """
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


@contextmanager
def no_rdkit_log():
    """
    Context manager to suppress all rdkit loggings.
    """
    RDLogger.DisableLog("rdApp.*")
    yield
    RDLogger.EnableLog("rdApp.*")


class CaptureStdIO(object):

    def __init__(self, stdout=True, stderr=False):
        self.stdout = stdout
        self.stderr = stderr
        self.file = tempfile.TemporaryFile("w+")

    def __enter__(self):
        if self.stdout:
            stdout_fd = sys.stdout.fileno()
            self.stdout_fd = os.dup(stdout_fd)
            os.dup2(self.file.fileno(), stdout_fd)
        if self.stderr:
            stderr_fd = sys.stderr.fileno()
            self.stderr_fd = os.dup(stderr_fd)
            os.dup2(self.file.fileno(), stderr_fd)
        return self

    def __exit__(self, type, value, traceback):
        if self.stdout:
            os.dup2(self.stdout_fd, sys.stdout.fileno())
            os.close(self.stdout_fd)
        if self.stderr:
            os.dup2(self.stderr_fd, sys.stderr.fileno())
            os.close(self.stderr_fd)
        self.file.seek(0)
        self.content = self.file.read().rstrip()
        self.file.close()


def capture_rdkit_log():
    """
    Context manager to capture all rdkit loggings.

    Example::

        >>> with utils.capture_rdkit_log() as log:
        >>>     ...
        >>> print(log.content)
    """
    return CaptureStdIO(True, True)


# Fallback rank utility — can be manually set if needed
class RankState:
    rank: Optional[int] = None

    @classmethod
    def get_rank(cls) -> int:
        if cls.rank is not None:
            return cls.rank
        for env_var in ["LOCAL_RANK", "RANK"]:
            if env_var in os.environ:
                return int(os.environ[env_var])
        return 0  # default to rank 0 if unknown

    @classmethod
    def set_rank(cls, value: int):
        cls.rank = value


def rank_prefixed_message(msg: str, rank: int) -> str:
    return f"[rank: {rank}] {msg}"


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = RankState.get_rank()
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None or current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
