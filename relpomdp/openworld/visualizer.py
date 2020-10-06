from abc import ABC, abstractmethod

class WorldViz:

    @abstractmethod
    def update(self, object_states):
        pass

    @abstractmethod
    def clear(self):
        pass
