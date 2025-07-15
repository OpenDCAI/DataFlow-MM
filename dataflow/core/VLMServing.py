from .LLMServing import LLMServingABC
from abc import ABC, abstractmethod
from typing import Any, List
class VLMServingABC(LLMServingABC):
    """Abstract base class for VLM serving. Which may be used to generate data from a model or API. Called by operators
    """
    @abstractmethod
    def generate_from_input_messages(
        self,
        conversations: List[List[dict]],
        image_list: List[List[str]] = None,
        video_list: List[List[str]] = None,
        audio_list: List[List[str]] = None
    ) -> List[str]:
        """
        Generate data from input conversations.
        """
        pass