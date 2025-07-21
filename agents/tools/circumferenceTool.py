from langchain.tools import BaseTool
from typing import Union
from math import pi

class CircumferenceTool(BaseTool):
    name = "Circumference calculator"
    description = "use this tool when you need to calculate a circumference using radius of circle"

    def _run(self, radius: Union[int,float]) -> float:
        return float(radius)*2.0*pi
    
    def _arun(self, radius: Union[int,float]):
        raise NotImplementedError('This tool does not support async')
