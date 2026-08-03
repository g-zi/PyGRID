class GridModel:

    def __init__(
        self,
        nx,
        ny,
        nz
    ):
        self.nx = nx
        self.ny = ny
        self.nz = nz

class EclipseGrid:

    def __init__(
        self,
        nx,
        ny,
        nz,
        xmin,
        xmax,
        ymin,
        ymax
    ):

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.xmin=xmin
        self.xmax=xmax

        self.ymin=ymin
        self.ymax=ymax

        self.dx=(xmax-xmin)/nx
        self.dy=(ymax-ymin)/ny


    def specgrid(self):

        return (
            "SPECGRID\n"
            f" {self.nx} {self.ny} {self.nz} 1 F /\n"
        )


    def coord(self):

        text=[]

        for j in range(self.ny+1):

            y=self.ymin+j*self.dy

            for i in range(self.nx+1):

                x=self.xmin+i*self.dx

                text.append(
                    f"{x:12.3f} {y:12.3f} 0.0 "
                    f"{x:12.3f} {y:12.3f} 1000.0\n"
                )

        return "".join(text)
