from abc import abstractmethod, ABC
import time


class Labeler(ABC):

    @abstractmethod
    def __call__(self, id1 : int, id2 : int):
        """
        label the pair (id1, id2)

        returns
        -------
            float : the label for the pair
        """
        pass

class GoldLabeler(Labeler):

    def __init__(self, gold):
        self._gold = gold

    def __call__(self, id1, id2):
        return 1.0 if (id1, id2) in self._gold else 0.0

class DelayedGoldLabeler(Labeler):

    def __init__(self, gold, delay_secs):
        self._gold = gold
        # the number of seconds that the labeler waits until it outputs the label
        # this is used to simulate human labeling
        self._delay_secs = delay_secs

    def __call__(self, id1, id2):
        time.sleep(self._delay_secs)
        return 1.0 if (id1, id2) in self._gold else 0.0
