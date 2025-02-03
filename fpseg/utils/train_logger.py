import yaml

import os
import time
from typing import Dict


class TrainLogger:
    """
    A utility class for logging the training process.

    Attributes:
        base_dir (str): The base directory where logs are stored.
        log_dir (Optional[str]): The directory for the current log session.
        log_file (Optional[str]): The path to the primary log file.
    """

    def __init__(self, base_dir: str = "output", config: Dict[str, any] = None):
        """
        Initialize the Logger class and create the log directory and log file.

        Args:
            base_dir (str): The base directory for logs. Defaults to "logs".
            config (Dict[str, any]): Configuration data to be logged.
        """
        self.base_dir = base_dir
        self.log_dir = None
        self.log_file = None

        self._create_log_dir()
        if config is not None:
            self._initialize_log_file(config)

    def _create_log_dir(self) -> None:
        """
        Create a new log directory with a timestamp as file name.
        """

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.base_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)

    def _initialize_log_file(self, config: Dict[str, any]) -> None:
        """
        Initialize the log file and write the configuration.

        Args:
            config (Dict[str, any]): Configuration data to be logged.
        """
        self.log_file = os.path.join(self.log_dir, "log.txt")
        with open(self.log_file, "w") as file:
            file.write("Training Log\n")
            file.write("=" * 50 + "\n")
            file.write("Configuration:\n")
            yaml.dump(config, file, default_flow_style=False)
            file.write("=" * 50 + "\n")

    def append(self, message: str) -> None:
        """
        Append a custom message to the log file.

        Args:
            message (str): The message to append to the log file.

        Raises:
            RuntimeError: If the log file is not set.
        """
        if self.log_file is None:
            raise RuntimeError("Log file is not set. Ensure initialization was successful.")

        with open(self.log_file, "a") as file:
            file.write(message + "\n")

    def get_dir(self) -> str:
        """
        Get the log directory.

        Returns:
            str: The name of the log file.
        """
        return self.log_dir
