function [sigma_sp_fr, sigma_sp_re, sigma_sk_up, sigma_sk_lo, deflected_grid] = dAEDalusSteadyStructuralAnalysis()
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/stdlib'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/geometry'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/aerodynamics'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/structures'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/aircraft'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/criticaldesign'));
    
    % Obtain geometric_model
    geometric_model = evalin('base', 'geometric_model');
    structural_model = evalin('base', 'structural_model');
    aerodynamic_model = evalin('base', 'aerodynamic_model');
    
    % Transforms aeroloads to structure
    geometric_model = geometric_model.compute_beam_forces(aerodynamic_model.F_body, structural_model);
    for i = 1:length(structural_model.beam)
        if  isa(structural_model.beam(i),'class_wing')
            structural_model.beam(i) = structural_model.beam(i).f_set_aeroloads(geometric_model.wings(i));

        end
    end
    
    % Solve structural model to get the deflections
    structural_model = structural_model.f_solve();
    
    % Compute the deflected grid of the geometrical model
    geometric_model = geometric_model.compute_deflected_grid(structural_model.f_get_deflections);
    deflected_grid = geometric_model.grid_deflected;
    %deflected_grid = deflected_grid(:, 1:(size(deflected_grid, 2)/2));
     
    % Calculate the stresses
    structural_model = structural_model.f_calc_stresses();  
    
    % Gather all the stresses in arrays for front/rear spars and top/bottom skins
    beam = structural_model.beam;
    n_segments = length(geometric_model.wings.wing_segments); 
    
    sigma_sp_fr = zeros(1, n_segments);
    sigma_sp_re = zeros(1, n_segments);
    sigma_sk_up = zeros(1, n_segments);
    sigma_sk_lo = zeros(1, n_segments);
    
    for i = 1:length(beam.beamelement)
        cs = beam.beamelement(i).crosssection;
        i_segment = cs.segment_index;
        
        sigma_sp_fr(i_segment) = max([sigma_sp_fr(i_segment), cs.sigma_sp_fr]);
        sigma_sp_re(i_segment) = max([sigma_sp_re(i_segment), cs.sigma_sp_re]);
        sigma_sk_up(i_segment) = max([sigma_sk_up(i_segment), cs.sigma_sk_up]);
        sigma_sk_lo(i_segment) = max([sigma_sk_lo(i_segment), cs.sigma_sk_lo]);
    end  
    
    assignin('base', 'geometric_model', geometric_model);
    assignin('base', 'structural_model', structural_model);
    
end