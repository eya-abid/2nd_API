from abc import ABC, abstractmethod


class Abstract(ABC):
    @abstractmethod
    def predict(self, text: str):
        pass
