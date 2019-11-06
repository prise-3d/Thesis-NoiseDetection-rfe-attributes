# main imports
import logging

# module imports
from ..Operator import Operator

# define policy to choose `operator` function at current iteration
class Policy():

    # here you can define your statistical variables for choosing next operator to apply

    def __init__(self, _operators):
        self.operators = _operators


    def select(self):
        """
        Select specific operator to solution and returns solution
        """
        raise NotImplementedError
        
    def apply(self, solution, secondSolution=None):
        """
        Apply specific operator chosen to solution and returns solution
        """
        
        operator = self.select()

        logging.info("---- Applying %s on %s" % (type(operator).__name__, solution))

        # check kind of operator
        if operator.kind == Operator.CROSSOVER:
            newSolution = operator.apply(solution, secondSolution)
        
        if operator.kind == Operator.MUTATOR:
            newSolution = operator.apply(solution)

        logging.info("---- Obtaining %s" % (solution))

        return newSolution