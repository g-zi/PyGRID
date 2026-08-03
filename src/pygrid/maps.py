from dataclasses import dataclass
import numpy as np


@dataclass
class SurfacePoint:
    x: float
    y: float
    z: float
    identifier: str



class Surface:

    def __init__(self, name):
        self.name = name
        self.points = []


    def add_point(
        self,
        x,
        y,
        z,
        identifier
    ):
        self.points.append(
            SurfacePoint(
                float(x),
                float(y),
                float(z),
                str(identifier)
            )
        )


    def xyz(self):

        return np.array(
            [
                [
                    p.x,
                    p.y,
                    p.z
                ]
                for p in self.points
            ]
        )



class TopSurface(Surface):

    def __init__(self):
        super().__init__("TOP")



class BottomSurface(Surface):

    def __init__(self):
        super().__init__("BOTTOM")



class ThicknessSurface(Surface):

    def __init__(self):
        super().__init__("THICKNESS")