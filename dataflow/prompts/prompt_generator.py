from abc import ABC, abstractmethod
from typing import Any

class PromptGeneratorABC(ABC):
    @abstractmethod
    def build_prompt(self, data) -> Any:
        pass