from dataclasses import dataclass



@dataclass
class FaultPoint:

    x: float
    y: float
    identifier: str



class Fault:


    def __init__(self, name):

        self.name = name
        self.points = []


    def add_point(
        self,
        x,
        y
    ):

        self.points.append(
            FaultPoint(
                float(x),
                float(y),
                self.name
            )
        )



class FaultSet:


    def __init__(self):

        self.faults = {}


    def add(
        self,
        fault
    ):

        self.faults[fault.name] = fault