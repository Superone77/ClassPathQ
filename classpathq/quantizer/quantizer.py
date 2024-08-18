import abc

class Quantizer(object):
    """
    basic class of quantizer
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def process_quant(self):
        pass