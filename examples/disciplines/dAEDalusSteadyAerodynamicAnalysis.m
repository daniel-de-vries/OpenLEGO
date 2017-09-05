function [C_D_i] = dAEDalusSteadyAerodynamicAnalysis(deflected_grid, C_L)
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/stdlib'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/geometry'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/aerodynamics'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/aircraft'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/criticaldesign'));
    
    % Obtain geometric_model
    geometric_model = evalin('base', 'geometric_model');
    aerodynamic_model = evalin('base', 'aerodynamic_model');

    aerodynamic_model.set_grid(deflected_grid, geometric_model.panels);
    aerodynamic_model = aerodynamic_model.f_solve_for_Cl_fast(C_L);
    
    % Obtain C_D_i
    C_D_i = aerodynamic_model.Cdi;
    
    assignin('base', 'aerodynamic_model', aerodynamic_model);
end