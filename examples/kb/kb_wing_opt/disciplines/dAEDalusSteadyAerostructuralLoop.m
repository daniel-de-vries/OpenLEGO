function [m_wing_struct, C_L, C_D_f, C_D_i, sigma_sp_fr, sigma_sp_re, sigma_sk_up, sigma_sk_lo, deflected_grid, y_norm, l_norm] = dAEDalusSteadyAerostructuralLoop(cpacs, n_x, n_y, M, H, n)
    [m_wing_struct, initial_grid] = dAEDalusSteadyModelInitializer(cpacs, n_x, n_y);

    [C_L, C_D_f] = dAEDalusSteadyAerodynamicModelInitializer(M, H, n);

    deflected_grid_guess = initial_grid;
    tol = 1e-6;
    iter = 0;
    while 1
        C_D_i = dAEDalusSteadyAerodynamicAnalysis(deflected_grid_guess, C_L);
        [sigma_sp_fr, sigma_sp_re, sigma_sk_up, sigma_sk_lo, deflected_grid] = dAEDalusSteadyStructuralAnalysis();

        r_norm = sqrt(sum((deflected_grid - deflected_grid_guess).^2, 1));
        err = rms(r_norm);
        %fprintf(1, 'iter: %d\t err: %.7f\n', iter, err);
        iter = iter + 1;
        if err < tol
            break
        end
        deflected_grid_guess = deflected_grid;
    end
    [y_norm, l_norm] = dAEDalusSteadyLiftDistribution( );
end