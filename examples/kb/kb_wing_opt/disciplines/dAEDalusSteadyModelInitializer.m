function [m_wing, initial_grid] = dAEDalusSteadyModelInitializer(input_cpacs, n_seg_x, n_seg_y) 
    try
        evalin('base', 'clear geometrical_model');
        evalin('base', 'clear structural_model');
        evalin('base', 'clear aerodynamic_model');
    catch e
        
    end
    
    % Create Aircraft data structure
    geometric_model = class_aircraft.create_from_cpacs(input_cpacs);
    geometric_model.grid_settings.aerodynamic_fuselage = 0;
    geometric_model.wings = geometric_model.wings(1);
    geometric_model.wings_structural_properties = geometric_model.wings_structural_properties(1);

    % Obtain smallest chord length
    wing = geometric_model.wings(1);
    n_wing_segments = length(wing.wing_segments);
    c_min = 100;
    for i = 1:n_wing_segments
        c_min = min(c_min, wing.wing_segments(i).c_r);
        c_min = min(c_min, wing.wing_segments(i).c_t);
        
        geometric_model.wings(1).wing_segments(i).n_chord = n_seg_x;
        geometric_model.wings(1).wing_segments(i).n_span = n_seg_y;
    end
    
    geometric_model.grid_settings.dy_max_struct_grid = geometric_model.reference.b_ref/40;
    
    % Compute grid and initialize structural model
    geometric_model = geometric_model.compute_grid();
    [geometric_model, structural_model] = create_structural_model(geometric_model);
    structural_model.beam(1) = structural_model.beam(1).f_add_boundary_condition(class_boundary_condition(ceil(length(structural_model.beam.node_coords)/2),[1 1 1 1 1 1],[0 0 0 0 0 0]));
    geometric_model = geometric_model.compute_force_interpolation_matrix(structural_model);
        
    % Assemble structural model
    structure_solver_settings = class_wingstructure_solver_settings;
    structure_solver_settings.gravity = 0;
    structural_model = structural_model.f_set_solver_settings(structure_solver_settings);
    structural_model = structural_model.f_assemble(1,0);
    
    % Make sure the deflected grid of the geometrical model is equal to the
    % initial grid
    geometric_model.grid_deflected = geometric_model.grid;
    
    % Compute total wingbox mass
    structural_model = structural_model.f_calc_mass(geometric_model.weights);
    m_wing = structural_model.beam(1).m_total - structural_model.beam(1).m_fuel_total;
    
    initial_grid = geometric_model.grid;
            
    % Store these objects in the workspace for future use
    assignin('base', 'geometric_model', geometric_model);
    assignin('base', 'structural_model', structural_model);
    
end