from pygrid.io import read_surface, read_faults


top = read_surface(
    "TinyECL.cnt"
)

faults = read_faults(
    "TinyECL.flt"
)


print(
    "Surface points:",
    len(top.points)
)


print(
    "Faults:",
    list(faults.faults.keys())
)