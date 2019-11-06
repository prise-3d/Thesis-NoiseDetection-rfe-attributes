# main imports
import random
import sys

# module imports
from .Mutation import Mutation

from ...solutions.BinarySolution import BinarySolution
from ...solutions.Solution import Solution


class SimpleMutation(Mutation):

    def apply(self, solution):
        size = solution.size

        firstCell = 0
        secondCell = 0

        # copy data of solution
        currentData = solution.data.copy()

        while firstCell == secondCell:
            firstCell = random.randint(0, size - 1) 
            secondCell = random.randint(0, size - 1)

        temp = currentData[firstCell]

        # swicth values
        currentData[firstCell] = currentData[secondCell]
        currentData[secondCell] = temp
        
        # create solution of same kind with new data
        return globals()[type(solution).__name__](currentData, size)

