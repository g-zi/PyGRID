from dataclasses import dataclass
from pathlib import Path


@dataclass
class MapPoint:
    x: float
    y: float
    z: float
    id: str


class ContourMap:

    def __init__(self, filename):
        self.filename = Path(filename)
        self.points = []

    def read(self):

        with open(self.filename, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                if line.startswith("*"):
                    continue

                parts = line.split()

                if len(parts) < 4:
                    continue

                self.points.append(
                    MapPoint(
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        parts[3]
                    )
                )

        return self


class ThicknessMap:

    def __init__(self, filename):
        self.filename = Path(filename)
        self.points = []

    def read(self):

        with open(self.filename, "r") as f:

            for line in f:

                line=line.strip()

                if not line:
                    continue

                parts=line.split()

                if len(parts)<4:
                    continue

                self.points.append(
                    MapPoint(
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        parts[3]
                    )
                )

        return self