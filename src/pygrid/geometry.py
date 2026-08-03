from grid import EclipseGrid


grid = EclipseGrid(
    nx=10,
    ny=10,
    nz=22,
    xmin=2120.05881763783,
    xmax=3552.95308296579,
    ymin=-4607.127,
    ymax=-3232.79687109415
)


with open("test.GRDECL", "w") as f:

    f.write(grid.specgrid())

    f.write("\nCOORD\n")
    f.write(grid.coord())

    f.write("\n/")


print("GRID file written")