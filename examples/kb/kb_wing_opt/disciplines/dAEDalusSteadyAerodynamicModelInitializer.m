function [C_L, C_D_f] = dAEDalusSteadyAerodynamicModelInitializer( M, H, n )

    geometric_model = evalin('base', 'geometric_model');

    % Create flight state
    ref_state = critical_ref_state(geometric_model, M, H);
    critical_state = critical_g_maneuver_state(ref_state, n);
    
    % Find the required C_l
    C_L = critical_state.get_Cl(geometric_model.reference.S_ref);
    
    % Compute C_D_f
    ac = geometric_model.compute_CD_f(critical_state.aerodynamic_state, geometric_model.reference.S_ref);
    C_D_f = ac.CD_f;
    
    % Initialize aerodynamic model
    aerodynamic_model = class_VLM_solver(...
        geometric_model.grid, ... % geometric_model.grid_deflected
        geometric_model.te_idx, ...
        geometric_model.panels, ...
        critical_state.aerodynamic_state, ...
        geometric_model.reference); 
    
    % Compute the influence coefficients
    aerodynamic_model.f_calc_coeffs();
    
    assignin('base', 'aerodynamic_model', aerodynamic_model);

end

