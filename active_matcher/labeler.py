from abc import abstractmethod, ABC


class Labeler(ABC):

    @abstractmethod
    def __call__(self, id1 : int, id2 : int):
        """
        label the pair (id1, id2)

        returns
        -------
            bool or float : the label for the pair
        """
        pass

class GoldLabeler(Labeler):

    def __init__(self, gold):
        self._gold = gold

    def __call__(self, id1, id2):
        return (id1, id2) in self._gold

