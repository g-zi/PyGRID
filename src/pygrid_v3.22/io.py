from .maps import TopSurface, BottomSurface, ThicknessSurface
from .faults import Fault, FaultSet



def read_surface(filename):

    if filename.endswith(".thk"):
        surface = ThicknessSurface()

    else:
        surface = TopSurface()


    with open(filename) as f:

        for line in f:

            if line.startswith("*"):
                continue

            values=line.split()

            if len(values) < 4:
                continue


            x,y,z,id = values[:4]

            surface.add_point(
                x,
                y,
                z,
                id
            )


    return surface




def read_faults(filename):

    faults=FaultSet()


    with open(filename) as f:

        for line in f:

            if line.startswith("*"):
                continue


            values=line.split()


            if len(values)<3:
                continue


            x,y,name=values[:3]


            if name not in faults.faults:

                faults.add(
                    Fault(name)
                )


            faults.faults[name].add_point(
                x,
                y
            )


    return faults