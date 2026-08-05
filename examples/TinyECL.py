#api_version=v0_0_107e

from __main__.tnav.workflow import *
from tnav_debug_utilities import *
from datetime import datetime, timedelta

declare_workflow (workflow_name='TinyECL', variables=[])
Data_Import_Workflow_variables = {}
def TinyECL(variables = Data_Import_Workflow_variables):

    begin_user_imports ()
    end_user_imports ()

# delete previous generated from TinyECL
    begin_wf_item (Index = 1)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name='Boundary', obj_type='Curve3d')]), ignore_if_not_exists=True)
    end_wf_item (Index = 1)
    begin_wf_item (Index = 2)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name= 'Top', obj_type= 'PointSet3d')]), ignore_if_not_exists=True)
    end_wf_item (Index = 2)
    begin_wf_item (Index = 3)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name= 'Thick', obj_type= 'PointSet3d')]), ignore_if_not_exists=True)
    end_wf_item (Index = 3)
    begin_wf_item (Index = 4)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name= 'Base', obj_type= 'PointSet3d')]), ignore_if_not_exists=True)
    end_wf_item (Index = 4)
    begin_wf_item (Index = 5)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name= 'Top', obj_type= 'Horizon')]), ignore_if_not_exists=True)
    end_wf_item (Index = 5)
    begin_wf_item (Index = 6)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name= 'Thick', obj_type= 'Horizon')]), ignore_if_not_exists=True)
    end_wf_item (Index = 6)
    begin_wf_item (Index = 7)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name= 'Mid', obj_type= 'Horizon')]), ignore_if_not_exists=True)
    end_wf_item (Index = 7)
    begin_wf_item (Index = 8)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name= 'Base', obj_type= 'Horizon')]), ignore_if_not_exists=True)
    end_wf_item (Index = 8)
    begin_wf_item (Index = 9)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name= 'TINYECL_Layer', obj_type= 'Table')]), ignore_if_not_exists=True)
    end_wf_item (Index = 9)
    begin_wf_item (Index = 10)
    object_delete (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name='TinyECL', obj_type='Grid3d')]), ignore_if_not_exists=True)
    end_wf_item (Index = 10)
    begin_wf_item (Index = 11)
    object_erase (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name='Polygons', obj_type='TypedFolder')]))
    end_wf_item (Index = 11)
    begin_wf_item (Index = 12)
    object_erase (object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name='Faults', obj_type='TypedFolder')]))
    end_wf_item (Index = 12)


# create extension boundary for the grid
    begin_wf_item(Index = 13)
    polygon_import_txt_format (use_tags_to_assign=False, tags_to_assign=[], files_table=[{'file_name' : 'C:/Mac/Home/Documents/TinyECL/GRID/TinyECL.ext', 'prefix' : 'Boundary'}])
    end_wf_item(Index = 13)


# read top contouring as pointset
    begin_wf_item(Index = 14)
    pointset_import_xyz_format (pointset_files_table=[{'file_name' : 'C:/Mac/Home/Documents/TinyECL/GRID/TinyECL.cnt' , 'point_set' : find_object (name='Top', type='PointSet3d')}],
      tabulator=TableFormat (separator='all spaces', comment='#', skip_lines=1, columns=['X', 'Y', 'Z']), use_xy_units=True, xy_units='metres', use_z_units=True, z_units='metres')
    end_wf_item(Index = 14)

# create top horizon from point sets
    begin_wf_item(Index = 15)
    horizon_interpolate (result_horizon=find_object (name='Top', type='Horizon'),
      point_sets_table=[{'use' : True, 'point_set' : find_object (name='Top', type='PointSet3d')}], simplify_faults_geometry=True, wells=find_object (name='Wells', type='gt_wells_entity'),
      grid_2d_settings=Grid2DSettings (grid_2d_settings_shown=True, autodetect_box=True,  autodetect_angle=True, autodetect_grid=False,
      grid_adjust_mode = 'step',
      step_x = 10,
      step_y = 10,
      sample_object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name='Top', obj_type='PointSet3d')]), autodetect_during_wf_calculation=True),
      use_polygon=True, polygon=find_object (name='Boundary', type='Curve3d'), interpolation_type='convergent', power_parameter=5)
    end_wf_item(Index = 15)

# smoothing of top horizon
    begin_wf_item (Index = 16)
    horizon_smooth (input_horizon=find_object (name='Top', type='Horizon'),
      output_horizon=find_object (name='Top', type='Horizon'), smoothing_method='moving_average', radius=100, seed=1)
    end_wf_item (Index = 16)


# read bottom contouring as pointset
    begin_wf_item(Index = 17)
    pointset_import_xyz_format (pointset_files_table=[{'file_name' : 'C:/Mac/Home/Documents/TinyECL/GRID/TinyECL.btm' , 'point_set' : find_object (name='Base', type='PointSet3d')}],
      tabulator=TableFormat (separator='all spaces', comment='#', skip_lines=1, columns=['X', 'Y', 'Z']), use_xy_units=True, xy_units='metres', use_z_units=True, z_units='metres')
    end_wf_item(Index = 17)

# create bottom horizon from point sets
    begin_wf_item(Index = 18)
    horizon_interpolate (result_horizon=find_object (name='Base', type='Horizon'),
      point_sets_table=[{'use' : True, 'point_set' : find_object (name='Base', type='PointSet3d')}], simplify_faults_geometry=True, wells=find_object (name='Wells', type='gt_wells_entity'),
      grid_2d_settings=Grid2DSettings (grid_2d_settings_shown=True, autodetect_box=True,  autodetect_angle=True, autodetect_grid=False,
      grid_adjust_mode = 'step',
      step_x = 10,
      step_y = 10,
      sample_object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name='Base', obj_type='PointSet3d')]), autodetect_during_wf_calculation=True),
      use_polygon=True, polygon=find_object (name='Boundary', type='Curve3d'), interpolation_type='convergent', power_parameter=5)
    end_wf_item(Index = 18)

# smoothing of bottom horizon
    begin_wf_item (Index = 19)
    horizon_smooth (input_horizon=find_object (name='Base', type='Horizon'),
      output_horizon=find_object (name='Base', type='Horizon'), smoothing_method='moving_average', radius=100, seed=1)
    end_wf_item (Index = 19)


# read thickness contouring as pointset
    begin_wf_item(Index = 20)
    pointset_import_xyz_format (pointset_files_table=[{'file_name' : 'C:/Mac/Home/Documents/TinyECL/GRID/TinyECL.thk' , 'point_set' : find_object (name='Thick', type='PointSet3d')}],
      tabulator=TableFormat (separator='all spaces', comment='#', skip_lines=1, columns=['X', 'Y', 'Z']), use_xy_units=True, xy_units='metres', use_z_units=True, z_units='metres')
    end_wf_item(Index = 20)

# create thickness horizon from point sets
    begin_wf_item(Index = 21)
    horizon_interpolate (result_horizon=find_object (name='Thick', type='Horizon'),
      point_sets_table=[{'use' : True, 'point_set' : find_object (name='Thick', type='PointSet3d')}], simplify_faults_geometry=True, wells=find_object (name='Wells', type='gt_wells_entity'),
      grid_2d_settings=Grid2DSettings (grid_2d_settings_shown=True, autodetect_box=True, autodetect_angle=True, autodetect_grid=False,
      grid_adjust_mode = 'step',
      step_x = 10,
      step_y = 10,
      sample_object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name='Thick', obj_type='PointSet3d')]), autodetect_during_wf_calculation=True),
      use_polygon=True, polygon=find_object (name='Boundary', type='Curve3d'), interpolation_type='convergent', power_parameter=5)
    end_wf_item(Index = 21)

# smoothing of thickness horizon
    begin_wf_item (Index = 22)
    horizon_smooth (input_horizon=find_object (name='Thick', type='Horizon'),
      output_horizon=find_object (name='Thick', type='Horizon'), smoothing_method='moving_average', radius=100, seed=1)
    end_wf_item (Index = 22)

# bottom horizon from top horizon + thickness map
    begin_wf_item (Index = 23)
    horizon_calculator (result_horizon=find_object (name='Mid', type='Horizon'),
      grid_2d_settings=Grid2DSettings (grid_2d_settings_shown=True, autodetect_box=True, autodetect_angle=True, autodetect_grid=False,
      grid_adjust_mode='step',
      step_x=10,
      step_y=10,
      sample_object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name='Top', obj_type='Horizon')])),
      formula='Top+Thick')
    end_wf_item (Index = 23)


# create proportional table for layering
    begin_wf_item (Index = 24)
    fill_table (data_table=[{'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 1, 'column' : 1, 'data' : 'Zone1'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 2, 'column' : 1, 'data' : '44.8474576271183'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 3, 'column' : 1, 'data' : '7.55254237288136'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 4, 'column' : 1, 'data' : '5.34242424242439'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 5, 'column' : 1, 'data' : '16.059659090909'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 6, 'column' : 1, 'data' : '31.9529166666671'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 7, 'column' : 1, 'data' : '12.5290909090909'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 8, 'column' : 1, 'data' : '8.69507575757598'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 9, 'column' : 1, 'data' : '5.95833333333303'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 10, 'column' : 1, 'data' : '6.52916666666624'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 11, 'column' : 1, 'data' : '6.36333333333368'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 12, 'column' : 1, 'data' : '6.83666666666704'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 13, 'column' : 1, 'data' : '13.0630630630631'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 14, 'column' : 1, 'data' : '33.5461323392356'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 15, 'column' : 1, 'data' : '15.4841379310346'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 16, 'column' : 1, 'data' : '14.7448543689316'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 17, 'column' : 1, 'data' : '8.25985151342138'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 18, 'column' : 1, 'data' : '6.02839756592311'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 19, 'column' : 1, 'data' : '13.2556770395286'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 20, 'column' : 1, 'data' : '9.385430038511'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 21, 'column' : 1, 'data' : '5.2236842105267'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 22, 'column' : 1, 'data' : '11.9741807348555'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 23, 'column' : 1, 'data' : '21.742924528302'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 24, 'column' : 1, 'data' : '5.70471014492796'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 25, 'column' : 1, 'data' : '6.33695652173901'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 26, 'column' : 1, 'data' : '9.55476190476202'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 27, 'column' : 1, 'data' : '8.59874686716739'},
      {'filled_table' : find_object (name='TinyECL_Layer', type='Table'), 'row' : 28, 'column' : 1, 'data' : '4.42982456140362'},
      ])
    end_wf_item (Index = 24)


# Import Faults as Polygons
    begin_wf_item (Index = 25)
    polygon_import_txt_table_format (splitter=True, files_table=[{'file_name' : 'C:/Mac/Home/Documents/TinyECL/GRID/TinyECL.flb', 'prefix' : 'TinyECL'}],
      tab_group=True, tabulator=TableFormat (separator='all spaces', skip_lines=0, columns=['X', 'Y', 'Z', 'Component', 'Polygon']), splitter3=True)
    end_wf_item (Index = 25)

# Create vertical Faults with Polygons
    begin_wf_item (Index = 26)
    faults_create_vertical_by_polygons (top=15200, bottom=17800, polygon_table=[
      {'use' : True, 'polygon' : find_object (name='TinyECL_T_F1', type='Curve3d'), 'fault_name' : 'F1'},
      {'use' : True, 'polygon' : find_object (name='TinyECL_T_F2', type='Curve3d'), 'fault_name' : 'F2'},
      {'use' : True, 'polygon' : find_object (name='TinyECL_T_F3', type='Curve3d'), 'fault_name' : 'F3'},
      {'use' : True, 'polygon' : find_object (name='TinyECL_T_F4', type='Curve3d'), 'fault_name' : 'F4'},
      {'use' : True, 'polygon' : find_object (name='TinyECL_T_F6', type='Curve3d'), 'fault_name' : 'F6'},
      {'use' : True, 'polygon' : find_object (name='TinyECL_T_F5', type='Curve3d'), 'fault_name' : 'F5'},
      {'use' : True, 'polygon' : find_object (name='TinyECL_T_F3', type='Curve3d'), 'fault_name' : 'F3'},
      ])
    end_wf_item (Index = 26)



# create grid by horizons, faults and proportional table
    begin_wf_item (Index = 27)
    grid_3d_create_by_horizons_with_faults (grid=find_object (name='TinyECL', type='Grid3d'), horizons_group_box=True, use_one_layering_column=True, use_individual_minimum_zone_thickness_column=True, use_proportions_table=True, proportions_table=find_object (name='TinyECL_Layer',type='Table'), 
      horizons_table=[{'horizon' : find_object (name='Top',  type='Horizon'), 'zone' : 'Zone1', 'partition_type' : 'explicit_proportional', 'counts_step' : 27, 'individual_minimum_zone_thickness' : 0, 'horizon_type' : 'conformable', 'fault_lines' : find_object (name='Top',  type='FaultLines'), 'fault_lines_usage' : 'calculate'},
                      {'horizon' : find_object (name='Base', type='Horizon'), 'zone' : 'Zone2', 'fault_lines' : find_object (name='Base',  type='FaultLines'), 'fault_lines_usage' : 'calculate'},
      ],
      faults_group_box=True, faults=[
      {'use' : True, 'fault' : find_object (name='F1', type='Fault3d'), 'structure' : True, 'zigzag' : False },
      {'use' : True, 'fault' : find_object (name='F2', type='Fault3d'), 'structure' : True, 'zigzag' : False },
      {'use' : True, 'fault' : find_object (name='F3', type='Fault3d'), 'structure' : True, 'zigzag' : False },
      {'use' : True, 'fault' : find_object (name='F4', type='Fault3d'), 'structure' : True, 'zigzag' : False },
      {'use' : True, 'fault' : find_object (name='F6', type='Fault3d'), 'structure' : True, 'zigzag' : False },
      {'use' : True, 'fault' : find_object (name='F5', type='Fault3d'), 'structure' : True, 'zigzag' : False },
      {'use' : True, 'fault' : find_object (name='F3', type='Fault3d'), 'structure' : True, 'zigzag' : False },
      ],
      individual_fault_amplitudes=True, residual_maps_suffix='_Discrepancy', general_settings=True, use_auto_filtration_radius=True, filtration_radius=100,
      grid_2d_settings=Grid2DSettings (grid_2d_settings_shown=True, autodetect_box=True, autodetect_angle=True, 
      grid_adjust_mode ='counts', counts_x=50, counts_y=50,
      sample_object=absolute_object_name (name=None, typed_names=[typed_object_name (obj_name='Boundary', obj_type='Curve3d')]), autodetect_during_wf_calculation=True),
      algorithm='least_squares', first_derivative_coefficient=0.3, second_derivative_coefficient=0.1,
      do_drag=True, drag_iterations=10, drag_coefficient=0.5, convergent_refinement_ratio=1.5, local_structure_faults_effect=True, extend_faults_to_z_borders=True,
      extended_segments=find_object (name='', type='Grid3dProperty'), smooth_radius=0, debug_faults_in_inner_space=True, use_clear_horizons_in_grid_calculation=True)
    end_wf_item (Index = 27)

# export simulation grid
    begin_wf_item (Index = 28)
    grid_3d_export_grdecl_format (file_name='C:/Mac/Home/Documents/TinyECL/GRID/TinyECL.GRDECL',
      mesh=find_object (name='TinyECL', type='Grid3d'), one_file_sign=True)
    end_wf_item (Index = 28)


