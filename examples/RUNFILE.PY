# RUNFILE.PY written from TinyECL
# PyGRID input file

from pygrid import *

model = GridModel(
    name = 'TinyECL',
    units = 'METRES',
    xmin = 643874.932838014,
    ymin = 2708563.10246777,
    xmax = 647513.805247266,
    ymax = 2712843.95490576,
    nx = 50,
    ny = 50
)

# Grid boundary
model.extent = 'TinyECL.ext'

# Top contour map
model.top = 'TinyECL.cnt'

# Bottom contour map
model.bottom = 'TinyECL.cnb'

# Thickness map
model.thickness = 'TinyECL.thk'

# Fault definitions
model.top_faults = 'TinyECL.flt'
model.bottom_faults = 'TinyECL.flb'

# Wells
model.wells = 'TinyECL.wll'

# Layer definition
model.layer_mode = 'THICKNESS'
model.layers = 27

# Rotate grid
model.rotate(
    angle = -5.6099190088477,
    x = 645694.36904264,
    y = 2710703.52868677
)

# Split grid along faults
model.split_faults = True

# Output Eclipse GRDECL

model.write_grdecl(
    specgrid = True,
    coord = True,
    zcorn = True
)

# Output faults
model.write_faults(
    filename = 'TinyECL.FAULTS'
)

# End of RUNFILE.PY

print('PyGRID completed')
