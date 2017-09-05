function [ y_norm, l_norm ] = dAEDalusSteadyLiftDistribution( )
%DAEDALUSSTEADYLIFTDISTRIBUTION Summary of this function goes here
%   Detailed explanation goes here

    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/geometry'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/aerodynamics'));
    addpath(genpath('../daedalus/dAEDalus_min/dAEDalus/aircraft'));

    geometric_model = evalin('base', 'geometric_model');
    aerodynamic_model = evalin('base', 'aerodynamic_model');
    
    % Compute the lift distribution
    % Finding chord lengths
    x_p_r_le = geometric_model.grid_deflected(:,aerodynamic_model.panels(1,:));	% Panel root LE locations
    x_p_r_te = geometric_model.grid_deflected(:,aerodynamic_model.panels(4,:));	% Panel root TE locations
    x_p_t_le = geometric_model.grid_deflected(:,aerodynamic_model.panels(2,:));	% Panel tip LE locations
    x_p_t_te = geometric_model.grid_deflected(:,aerodynamic_model.panels(3,:));	% Panel tip TE locations
    
    idx_te = find(geometric_model.is_te);                  % indices of TE panels
    idx_le = [1,idx_te(1:end-1)+1];                 % indices of LE panels
    x_s_r_le = x_p_r_le(:,idx_le);                  % strip root LE positions
    x_s_r_te = x_p_r_te(:,idx_te);                  % strip root TE positions
    x_s_t_le = x_p_t_le(:,idx_le);                  % strip tip LE positions
    x_s_t_te = x_p_t_te(:,idx_te);                  % strip tip TE positions
    
    x_s_le = x_s_r_le + 0.5*(x_s_t_le - x_s_r_le);  % strip middle LE positions
    
    n_panels = size(aerodynamic_model.panels,2);
    I1 = 1:n_panels <= idx_te';
    I2 = 1:n_panels >= idx_le';
    I = I1.*I2;

    L_s = (I*aerodynamic_model.F_aero(3,:)')';           % strip lift forces 
    b_s = (x_s_t_le(2,:) - x_s_r_le(2,:) + x_s_t_te(2,:) - x_s_r_te(2,:))/2; % strip spans 
    
    l = L_s./b_s;                  % section lift
    l_mean = sum(L_s)/sum(b_s); % mean section lift
    l_norm = l/l_mean; % normalized section lift
    
    [y,iy] = sort(x_s_le(2,:));                     % sorted y positions of LEs
    y_norm = y./(sum(b_s)/2);
    l_norm = l_norm(iy);

    y_norm = y_norm(length(y_norm)/2+1:end);
    l_norm = l_norm(length(l_norm)/2+1:end);

end

