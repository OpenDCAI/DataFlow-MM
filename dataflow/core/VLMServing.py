from .LLMServing import LLMServingABC
from abc import ABC, abstractmethod
from typing import Any, List
class VLMServingABC(LLMServingABC):
    """Abstract base class for VLM serving. Which may be used to generate data from a model or API. Called by operators
    """
    pass