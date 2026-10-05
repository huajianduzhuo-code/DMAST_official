from abc import ABC, abstractmethod
from dataclasses import dataclass

from agentlab.llm.base_api import AbstractChatModel


@dataclass
class BaseModelArgs(ABC):
    """Base class for all model arguments."""

    model_name: str
    max_total_tokens: int = 32768
    max_input_tokens: int = 32768 - 2048
    max_new_tokens: int = 4096
    temperature: float = 0.1
    vision_support: bool = False
    log_probs: bool = False

    @abstractmethod
    def make_model(self) -> AbstractChatModel:
        pass

    def prepare_server(self):
        pass

    def close_server(self):
        pass