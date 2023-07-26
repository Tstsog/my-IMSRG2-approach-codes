% This Matlab code computes  the ground state energy for helium atom by solving the
% flow-type equation combined with an imaginary time (it) generator
% in the in-medium similarity renormalization group approach using 
% Gaussian-type orbital (GTO) with n=2 (2 s-function)
%
% Refs: H. Hergert, S. K. Bogner, J. G. Lietz, T. D. Morris, S. J. Novario, N.M. Parzuchowski, and F. Yuan,   
% Lecture Notes in Physics - An Advanced Course in Computational Nuclear Physics, volume, 936, 477 (2017).
%
% Written by Tsogbayar Tsednee (PhD)
% Email: tsog215@gmail.com
% July 12, 2023 & University of North Dakota 
%
function [] = he_gto_n2_imsrg2_it
%
clear; clc; format long
%
dim2B = 4.; % size of spin-orbital basis sets
tol = 1e-3;
%
[E0, En_orb, tei] = he_scf_gto_n2; % gives ground state energy (E0), orbital energy (En_orb) and two-electron integral data (tei)
En_scf = E0; % The self-consistent field (scf) ground state energy
%
f = kron(diag(En_orb), eye(2)); % Fock matrix elements in molecular spin-orbital basis sets
%
Nhole_n = [1, 2]; % hole states
Nvir_n  = [3, 4]; % virtial (particle) states 
%
Gamma = zeros(dim2B,dim2B,dim2B,dim2B);  % antisymmetrized two-electron integral (Gamma(p,q,r,s))
for p = 1:dim2B
    for q = 1:dim2B
        for r = 1:dim2B
            for s = 1:dim2B
                p1 = floor((p+1)/2);
                q1 = floor((q+1)/2);
                r1 = floor((r+1)/2);                
                s1 = floor((s+1)/2);
                val1 = tei(p1,r1,q1,s1) * (rem(p,2)==rem(r,2)) * (mod(q,2)==rem(s,2));
                val2 = tei(p1,s1,q1,r1) * (rem(p,2)==rem(s,2)) * (rem(q,2)==rem(r,2));
                Gamma(p,q,r,s) = val1 - val2;
            end
        end
    end
end
%

%%%%%%%%%%%
fileID_save_data_1 = fopen('IMSRG2_im_time_He_GTO_n2_imag_time.txt','w');
Niter = 20000.;
ds = 1e-3;          % step size of flow parameter s
for iter = 1:Niter
    iter;
    % 
     [eta1B_k1, eta2B_k1] = eta_imtime(f, Gamma, dim2B, Nvir_n, Nhole_n);
    [dE_sum_k1, df_k1, dGamma_k1] = imsrg2(eta1B_k1, eta2B_k1, f, Gamma, dim2B, Nhole_n);
    %
    [eta1B_k2, eta2B_k2] = eta_imtime(f + 0.5 *ds * df_k1, Gamma + 0.5 * ds * dGamma_k1, dim2B, Nvir_n, Nhole_n);
    [dE_sum_k2, df_k2, dGamma_k2] = imsrg2(eta1B_k2, eta2B_k2, f + 0.5 *ds * df_k1, Gamma + 0.5 * ds * dGamma_k1, dim2B, Nhole_n);
    %
    [eta1B_k3, eta2B_k3] = eta_imtime(f + 0.5 *ds * df_k2, Gamma + 0.5 * ds * dGamma_k2, dim2B, Nvir_n, Nhole_n);
    [dE_sum_k3, df_k3, dGamma_k3] = imsrg2(eta1B_k3, eta2B_k3, f + 0.5 *ds * df_k2, Gamma + 0.5 * ds * dGamma_k2, dim2B, Nhole_n);
    %
    [eta1B_k4, eta2B_k4] = eta_imtime(f + ds * df_k3, Gamma + ds * dGamma_k3, dim2B, Nvir_n, Nhole_n);
    [dE_sum_k4, df_k4, dGamma_k4] = imsrg2(eta1B_k4, eta2B_k4, f + ds * df_k3, Gamma + ds * dGamma_k3, dim2B, Nhole_n);
    %
    E1 = E0 + (ds/6.) * (dE_sum_k1 + 2.*dE_sum_k2 + 2.*dE_sum_k3 + dE_sum_k4);
    f1 = f  + (ds/6.) * (df_k1 + 2.*df_k2 + 2.*df_k3 + df_k4) ;
    Gamma1 = Gamma + (ds/6.) * (dGamma_k1 + 2.*dGamma_k2 + 2.*dGamma_k3 + dGamma_k4);          % the 4th order Runge-Kutta method
    %
    E0 = E1;
    f = f1;
    Gamma = Gamma1;
    %
    [f_od, Gamma_od] = norm_fo_Gamma_od(f, Gamma, Nvir_n, Nhole_n);
    [d_E2] = mbpt2(f, Gamma, Nvir_n, Nhole_n);
    [d_E3] = mbpt3(f, Gamma, Nvir_n, Nhole_n);
    E_mbpt23 = E0 + d_E2 + d_E3;
    %
    output = [iter*ds, E0, E_mbpt23, f_od, Gamma_od];
    %
    fprintf(fileID_save_data_1, '%4.8f \t %4.12f \t %4.12f \t %4.12f \t %8.10f\n', output);      
%
    if (abs(Gamma_od/E0) < tol)
        break 
    end
end
fclose(fileID_save_data_1);

%%%
En_imsrg2 = E0;
[En_scf, En_imsrg2] % the SCF and IMSRG(2) energies
% [En_scf, En_imsrg2] = -2.747066128454680  -2.757820278383074

%%%
return
end
%%%

%%%%%%%%%%%%%%%%%%%%%%%%
function [eta1B, eta2B] = eta_imtime(f, Gamma, dim2B, Nvir_n, Nhole_n)
%
eta1B = zeros(dim2B,dim2B);
eta2B = zeros(dim2B,dim2B,dim2B,dim2B);
for a = Nvir_n
    for i = Nhole_n
        dE = f(a,a) - f(i,i) + Gamma(a,i,a,i);
        val = sign(dE) * f(a,i);
        eta1B(a,i) = val;
        eta1B(i,a) = -val;
    end
end
%
for a = Nvir_n
    for b = Nvir_n
        for i = Nhole_n
            for j = Nhole_n
                dE = f(a,a) + f(b,b) - f(i,i) - f(j,j) ...
                     + Gamma(a,b,a,b) ...
                     + Gamma(i,j,i,j) ...
                     - Gamma(a,i,a,i) ...
                     - Gamma(a,j,a,j) ...
                     - Gamma(b,i,b,i) ...
                     - Gamma(b,j,b,j);
                val = sign(dE) * Gamma(a,b,i,j);
                eta2B(a,b,i,j) =  val;
                eta2B(i,j,a,b) = -val;
            end
        end
    end
end
%
return
end
%

%
function [dE_sum, df, dGamma] = imsrg2(eta1B, eta2B, f, Gamma, dim2B, Nhole_n)
%
% Ref: equation (10.104)
dE_sum = 0.;
for a = Nhole_n
    for b = 1:dim2B
        dE_sum = dE_sum + eta1B(a,b) * f(b,a); 
    end
end
for a = 1:dim2B
    for b = Nhole_n
        dE_sum = dE_sum - eta1B(a,b) * f(b,a); 
    end
end
%
for a = Nhole_n
    for b = Nhole_n
        for c = 1:dim2B
            for d = 1:dim2B
                dE_sum = dE_sum + 0.5 * eta2B(a,b,c,d) * Gamma(c,d,a,b);
            end
        end
    end
end
%
for a = Nhole_n
    for b = Nhole_n
        for c = Nhole_n
            for d = 1:dim2B
                dE_sum = dE_sum - 0.5 * eta2B(a,b,c,d) * Gamma(c,d,a,b);
            end
        end
    end
end
%
for a = Nhole_n
    for b = Nhole_n
        for c = 1:dim2B
            for d = Nhole_n
                dE_sum = dE_sum - 0.5 * eta2B(a,b,c,d) * Gamma(c,d,a,b);
            end
        end
    end
end
%
for a = Nhole_n
    for b = Nhole_n
        for c = Nhole_n
            for d = Nhole_n
                dE_sum = dE_sum + 0.5 * eta2B(a,b,c,d) * Gamma(c,d,a,b);
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%

% Ref: equation (10.105)
df = zeros(dim2B,dim2B);
% 1B - 1B
for i = 1:dim2B
    for j = 1:dim2B
        for a = 1:dim2B
            df(i,j) = df(i,j) + (eta1B(i,a) * f(a,j) + eta1B(j,a) * f(a,i));
        end
    end
end
%
% 1B - 2B
for i = 1:dim2B
    for j = 1:dim2B
        for a = Nhole_n
            for b = 1:dim2B
                df(i,j) = df(i,j) + (eta1B(a,b) * Gamma(b,i,a,j) - f(a,b) * eta2B(b,i,a,j));
            end
        end
    end
end
%
for i = 1:dim2B
    for j = 1:dim2B
        for a = 1:dim2B
            for b = Nhole_n
                df(i,j) = df(i,j) - (eta1B(a,b) * Gamma(b,i,a,j) - f(a,b) * eta2B(b,i,a,j));
            end
        end
    end
end
%df;
% 2B- 2B
df_2B_2B_1 = zeros(dim2B,dim2B);
for i = 1:dim2B
    for j = 1:dim2B
       for a = Nhole_n        %
          for b = Nhole_n          % 
             for c = 1:dim2B        % 
                  df_2B_2B_1(i,j) = df_2B_2B_1(i,j) + (eta2B(c,i,a,b) * Gamma(a,b,c,j) + ...  
                                                       eta2B(c,j,a,b) * Gamma(a,b,c,i));
             end
          end
        end
    end
end
%
df_2B_2B_2 = zeros(dim2B,dim2B);
for i = 1:dim2B
    for j = 1:dim2B
       for a = 1:dim2B        %
          for b = 1:dim2B          % 
             for c = Nhole_n        % 
                  df_2B_2B_1(i,j) = df_2B_2B_1(i,j) + (eta2B(c,i,a,b) * Gamma(a,b,c,j) + ...  
                                                       eta2B(c,j,a,b) * Gamma(a,b,c,i));
             end
          end
        end
    end
end
%
df_2B_2B_3 = zeros(dim2B,dim2B);
for i = 1:dim2B
    for j = 1:dim2B
       for a = Nhole_n        %
          for b = 1:dim2B          % 
             for c = Nhole_n        % 
                  df_2B_2B_1(i,j) = df_2B_2B_1(i,j) + (eta2B(c,i,a,b) * Gamma(a,b,c,j) + ...  
                                                       eta2B(c,j,a,b) * Gamma(a,b,c,i));
             end
          end
        end
    end
end
%
df_2B_2B_4 = zeros(dim2B,dim2B);
for i = 1:dim2B
    for j = 1:dim2B
       for a = 1:dim2B        %
          for b = Nhole_n          % 
             for c = Nhole_n        % 
                  df_2B_2B_1(i,j) = df_2B_2B_1(i,j) + (eta2B(c,i,a,b) * Gamma(a,b,c,j) + ...  
                                                       eta2B(c,j,a,b) * Gamma(a,b,c,i));
             end
          end
        end
    end
end
%
df = df + 0.5* (df_2B_2B_1 + df_2B_2B_2 - df_2B_2B_3 - df_2B_2B_4);
%%%


%%%%%%%%%%%%%%%%
% dGamma part
% Ref: equation (10.106)
dGamma = zeros(dim2B,dim2B,dim2B,dim2B);
%
% 1B- 2B
for i = 1:dim2B
    for j = 1:dim2B
        for k = 1:dim2B
            for l = 1:dim2B
                for a = 1:dim2B
                    dGamma(i,j,k,l) = dGamma(i,j,k,l) + (eta1B(i,a) * Gamma(a,j,k,l) - f(i,a) * eta2B(a,j,k,l)) - ...
                                                        (eta1B(j,a) * Gamma(a,i,k,l) - f(j,a) * eta2B(a,i,k,l)) - ...
                                                        (eta1B(a,k) * Gamma(i,j,a,l) - f(a,k) * eta2B(i,j,a,l)) + ...
                                                        (eta1B(a,l) * Gamma(i,j,a,k) - f(a,l) * eta2B(i,j,a,k));
                end
            end
        end
    end
end
%dGamma;
% 2B - 2B - particle and hole ladders
dGamma_2B_2B_1 = zeros(dim2B,dim2B,dim2B,dim2B);
% 2B- 2B
for i = 1:dim2B
    for j = 1:dim2B
        for k = 1:dim2B
            for l = 1:dim2B
                for a = 1:dim2B
                    for b = 1:dim2B
                         dGamma_2B_2B_1(i,j,k,l) = dGamma_2B_2B_1(i,j,k,l) + (eta2B(i,j,a,b) * Gamma(a,b,k,l) - ...  
                                                                              Gamma(i,j,a,b) * eta2B(a,b,k,l));
                    end
                end
            end
        end
    end
end
dGamma_2B_2B_2 = zeros(dim2B,dim2B,dim2B,dim2B);
% 2B- 2B
for i = 1:dim2B
    for j = 1:dim2B
        for k = 1:dim2B
            for l = 1:dim2B
                for a = Nhole_n
                    for b = 1:dim2B
                         dGamma_2B_2B_2(i,j,k,l) = dGamma_2B_2B_2(i,j,k,l) + (eta2B(i,j,a,b) * Gamma(a,b,k,l) - ...  
                                                                              Gamma(i,j,a,b) * eta2B(a,b,k,l));
                    end
                end
            end
        end
    end
end
dGamma_2B_2B_3 = zeros(dim2B,dim2B,dim2B,dim2B);
% 2B- 2B
for i = 1:dim2B
    for j = 1:dim2B
        for k = 1:dim2B
            for l = 1:dim2B
                for a = 1:dim2B
                    for b = Nhole_n
                         dGamma_2B_2B_3(i,j,k,l) = dGamma_2B_2B_3(i,j,k,l) + (eta2B(i,j,a,b) * Gamma(a,b,k,l) - ...  
                                                                              Gamma(i,j,a,b) * eta2B(a,b,k,l));
                    end
                end
            end
        end
    end
end
dGamma = dGamma + 0.5.*(dGamma_2B_2B_1 - dGamma_2B_2B_2 - dGamma_2B_2B_3);

%%%%%%%%%%%%%%%%%%%%%%%%%
%  2B - 2B - particle-hole chain
dGamma_2B_2B_phc_na_1 = zeros(dim2B,dim2B,dim2B,dim2B);
% 2B- 2B
for i = 1:dim2B
    for j = 1:dim2B
        for k = 1:dim2B
            for l = 1:dim2B
                for a = Nhole_n
                    for b = 1:dim2B
                         dGamma_2B_2B_phc_na_1(i,j,k,l) = dGamma_2B_2B_phc_na_1(i,j,k,l) + (eta2B(a,i,b,k) * Gamma(b,j,a,l) - ...  
                                                                                            eta2B(a,i,b,l) * Gamma(b,j,a,k) - ...
                                                                                            eta2B(a,j,b,k) * Gamma(b,i,a,l) + ...  
                                                                                            eta2B(a,j,b,l) * Gamma(b,i,a,k)); 
                    end
                end
            end
        end
    end
end
%
dGamma_2B_2B_phc_nb_1 = zeros(dim2B,dim2B,dim2B,dim2B);
% 2B- 2B
for i = 1:dim2B
    for j = 1:dim2B
        for k = 1:dim2B
            for l = 1:dim2B
                for a = 1:dim2B
                    for b = Nhole_n
                         dGamma_2B_2B_phc_nb_1(i,j,k,l) = dGamma_2B_2B_phc_nb_1(i,j,k,l) + (eta2B(a,i,b,k) * Gamma(b,j,a,l) - ...  
                                                                                            eta2B(a,i,b,l) * Gamma(b,j,a,k) - ...
                                                                                            eta2B(a,j,b,k) * Gamma(b,i,a,l) + ...  
                                                                                            eta2B(a,j,b,l) * Gamma(b,i,a,k)); 
                    end
                end
            end
        end
    end
end
%
dGamma = dGamma + (dGamma_2B_2B_phc_na_1 - dGamma_2B_2B_phc_nb_1);


%%%
return
end

%%%
function [f_od, Gamma_od] = norm_fo_Gamma_od(f, Gamma, Nvir_n, Nhole_n) 
%
norm_fo = 0.;
for a = Nvir_n
    for i = Nhole_n
        norm_fo = norm_fo + f(a,i)^2 + f(i,a)^2;
    end
end 
f_od = sqrt(norm_fo);
%
norm_Gamma_od = 0.;
for a = Nvir_n
    for b =  Nvir_n
        for i = Nhole_n
            for j = Nhole_n
                norm_Gamma_od = norm_Gamma_od + Gamma(a,b,i,j)^2 + Gamma(i,j,a,b)^2;
            end
        end
    end
end
Gamma_od = sqrt(norm_Gamma_od);
%
return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [d_E2] = mbpt2(f, Gamma, Nvir_n, Nhole_n)
%
% Refs: 8.5 Many-Body Perturbation Theory & equation (8.23)
d_E2 = 0.;
for i = Nhole_n
    for j = Nhole_n 
        for a = Nvir_n
            for b = Nvir_n
                denom = f(i,i) + f(j,j) - f(a,a) - f(b,b);  
                val = Gamma(a,b,i,j);
                d_E2 = d_E2 + 0.25.*val.*val./denom;
            end
        end
    end
end
%
return
end
%
function [d_E3] = mbpt3(f, Gamma,Nvir_n, Nhole_n)
%
% Refs: 8.5 Many-Body Perturbation Theory & equation (8.24) - (8.26)
d_E3pp = 0.;
d_E3hh = 0.;
d_E3ph = 0.;
%
for a = Nvir_n
    for b = Nvir_n
        for c = Nvir_n
            for d = Nvir_n
                for i = Nhole_n
                    for j = Nhole_n
                        denom = (f(i,i) + f(j,j) - f(a,a) - f(b,b)) * (f(i,i) + f(j,j) - f(c,c) - f(d,d));
                        val = Gamma(i,j,a,b) * Gamma(a,b,c,d) * Gamma(c,d,i,j);
                        d_E3pp = d_E3pp + 0.125.*val./denom;
                    end
                end
            end
        end
    end
end
%
for i = Nhole_n
    for j = Nhole_n
        for k = Nhole_n
            for l = Nhole_n
                for a = Nvir_n
                    for b = Nvir_n
                        denom = (f(i,i) + f(j,j) - f(a,a) - f(b,b)) * (f(k,k) + f(l,l) - f(a,a) - f(b,b));
                        val = Gamma(a,b,k,l) * Gamma(k,l,i,j) * Gamma(i,j,a,b);
                        d_E3hh = d_E3hh + 0.125.*val./denom;
                    end
                end
            end
        end
    end
end
%
for i = Nhole_n
    for j = Nhole_n
        for k = Nhole_n
            for a = Nvir_n
                for b = Nvir_n
                    for c = Nvir_n
                        denom = (f(i,i) + f(j,j) - f(a,a) - f(b,b)) * (f(k,k) + f(j,j) - f(a,a) - f(c,c));
                        val = Gamma(i,j,a,b) * Gamma(k,b,i,c) * Gamma(a,c,k,j);
                        d_E3ph = d_E3ph - val./denom;
                    end
                end
            end
        end
    end
end
d_E3 = d_E3pp + d_E3hh + d_E3ph; 
%
return
end

%%%

function [E0, En_orb, tei] = he_scf_gto_n2
%
%clear; clc; format long
itermax = 60; tol = 1e-12;
%
z_h = 2.; % nuclear charge for helium atom 
%
xi1 = 0.532149; d1 = 0.82559; % from S. Huzinaga, J. Chem. Phys. 42, 1293–1302 (1965), Table IX
xi2 = 4.097728; d2 = 0.28317;
%
N = 512; % number of grid points along radial distance r; you may change it
a = 0.0;  % starting point od coordinate r
b = 50.; % end point of cooridnate r; you may change it
%
[r,wr,D]=legDC2(N,a,b); % D is the differentation matrix of the first order
D1 = (2/(b-a))*D; rr = r;
%
% GTO: basis functions & chi = d * exp(-xi*r^2) * (2*xi/pi)^(3/4)
chi_1 = d1.*exp(-xi1.*r.^2).*(2.*xi1./pi).^(3/4);    
chi_2 = d2.*exp(-xi2.*r.^2).*(2.*xi2./pi).^(3/4);     

%%%%%%%%%%%%%%%%%%%%
% Hamiltonian matrix elements in atomic orbital basis functions 
[h11,s11] = H0_elements_ss(z_h,D1,r,wr,chi_1,chi_1);
[h12,s12] = H0_elements_ss(z_h,D1,r,wr,chi_1,chi_2);
[h22,s22] = H0_elements_ss(z_h,D1,r,wr,chi_2,chi_2);
%
%%%
dim = 2;
d_coef = zeros(1,dim);
d_coef(1) = d1*(2.*xi1./pi).^(3/4);
d_coef(2) = d2*(2.*xi2./pi).^(3/4);
%
xi_coef = zeros(1,dim);
xi_coef(1) = xi1;
xi_coef(2) = xi2;
%
H_core = [h11, h12; % the core hamiltonian: matrix elements
          h12, h22];
%
S_ov = [s11, s12; % overlap matrix elements  
        s12, s22];
%
P_old = 0.5 * ones(dim,dim); % initial charge population
%
for iter = 1:itermax
    iter;
    P = P_old;
    %
    F = H_core;
    for p = 1:dim
        for q = 1:dim
            for r = 1:dim
                for s = 1:dim
                    F(p,q) = F(p,q) + P(r,s) * (tei_ssss(p,q,r,s, d_coef, xi_coef) - 0.5.*tei_ssss(p,r,q,s, d_coef, xi_coef));
                end
    
            end
    
        end
    end
    Ham_fock = F ;     % Fock matrix
    S_mat_fock = S_ov;

    [Vec,En] = eig(Ham_fock,S_mat_fock);                                     % Eigenvalue problem: F*c = En*S*c - Roothaan equation
    En = diag(En);
    [foo, ij] = sort(En);
    En = En(ij);
    [En(1), En(2)];  % orbital energies
    %
    Vec = Vec(:,ij);                       % expansion coefficients 
    %
    for i = 1:dim
        norm = 0.;
        for p = 1:dim
            for q = 1:dim
                norm = norm + Vec(p,i) * Vec(q,i) * S_ov(p,q);
            end
        end
        Vec(:,i) = Vec(:,i)/sqrt(norm);
    end
    %
    P_new = zeros(dim,dim);
    for i = 1:z_h/2
        for pp = 1:dim
            for qq = 1:dim
                P_new(pp,qq) = P_new(pp,qq) + 2*Vec(pp,i)*Vec(qq,i);
            end
        end
    end
    %
     if (abs(P_new-P_old) < tol)
            break 
     end
    %        
    P_old = P_new;

end
%%%

En_0 = (sum(0.5*diag(P(:,:)*(H_core(:,:) + F(:,:))))); % ground state energy in atomic unit
[En(1), En_0];
% [En(1), En_0] = -0.858910329320571  -2.747066128454680; vs [-0.858911, -2.7470661] from S. Huzinaga, J. Chem. Phys. 42, 1293–1302 (1965)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V1 = Vec(:,1); V2 = Vec(:,2);   % the expansion coefficients in basis functions
%%%
phi_1 = V1(1).*chi_1 + V1(2).*chi_2; % phi_1: molecular orbital basis function 
phi_2 = V2(1).*chi_1 + V2(2).*chi_2;
%%%
[h11,s11] = H0_elements_ss(z_h,D1,rr,wr,phi_1,phi_1);  % the core hamiltonian matrix elements in molecular orbital basis functions
[h12,s12] = H0_elements_ss(z_h,D1,rr,wr,phi_1,phi_2);
[h22,s22] = H0_elements_ss(z_h,D1,rr,wr,phi_2,phi_2);

%%%
P = Vec'; % charge density matrix
%
tei = zeros(dim,dim,dim,dim);  % two-electron integral (tei) in molecular orbital basis sets
for ii = 1:dim
    for jj = 1:dim
        for kk = 1:dim
            for ll = 1:dim
                for mm = 1:dim
                    for nn = 1:dim
                        for oo = 1:dim
                            for pp = 1:dim
                                tei(ii,jj,kk,ll) =  tei(ii,jj,kk,ll) + P(ii,mm)*P(jj,nn)*P(kk,oo)*P(ll,pp)*tei_ssss(mm,nn,oo,pp, d_coef, xi_coef);
                            end
                        end
                    end
                end
            end
        end
    end
end
%
E0_n = 2*h11 + tei(1,1,1,1); % the ground state energy E0_n = -2.747066128454680, from molecular orbital basis functions
%
E0 = E0_n; % ground state energy
En_orb = En; % orbital energy
%tei; % two-electron integral in molecular orbital basis sets

%%%
return
end

%%%%%%%%%%%%%%%%
function [h11,s11] = H0_elements_ss(z_h,D1,r,wr,chi_i,chi_j)
% compute kinetic and potential energies and overlap matrix
%
T_11 = sum(wr.*(D1*chi_i).*(D1*chi_j).*r.*r) *(4*pi) ;
V_11 = sum(wr.*chi_i.*(-z_h).*chi_j.*r) * (4*pi) ;
h11 = 0.5*T_11 + V_11;
s11 = sum(wr.*chi_i.*chi_j.*r.*r) * (4*pi);
%
%%%
return
end
%%%

%%%
function [Q_pqrs] = tei_ssss(p,q,r,s, d_coef, xi_coef)
% analytical expression for the two-electron integral 
%
Q_pqrs_numer = 2.*pi.^(5/2);
Q_pqrs_denun = (xi_coef(p) + xi_coef(q))*(xi_coef(r) + xi_coef(s))*sqrt(xi_coef(p) + xi_coef(q) + xi_coef(r) + xi_coef(s));
Q_pqrs = Q_pqrs_numer/Q_pqrs_denun; 
%
Q_pqrs = d_coef(p) * d_coef(q) * d_coef(r) * d_coef(s) * Q_pqrs;
%
return
end




    function [xi,w,D]=legDC2(N,a,b)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            % legDc.m
            %
            % Computes the Legendre differentiation matrix with collocation at the
            % Legendre-Gauss-Lobatto nodes.
            %
            % Reference:
            %   C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Tang, "Spectral Methods
            %   in Fluid Dynamics," Section 2.3. Springer-Verlag 1987
            %
            % Written by Greg von Winckel - 05/26/2004
            % Contact: gregvw@chtm.unm.edu
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            % Truncation + 1
            N1=N+1;
            
            % CGL nodes
            xc=cos(pi*(0:N)/N)';
            
            % Uniform nodes
            xu=linspace(-1,1,N1)';
            
            % Make a close first guess to reduce iterations
            if N<3
                x=xc;
            else
                x=xc+sin(pi*xu)./(4*N);
            end
            
            % The Legendre Vandermonde Matrix
            P=zeros(N1,N1);
            
            % Compute P_(N) using the recursion relation
            % Compute its first and second derivatives and
            % update x using the Newton-Raphson method.
            
            xold=2;
            while max(abs(x-xold))>eps
                
                xold=x;
                
                P(:,1)=1;    P(:,2)=x;
                
                for k=2:N
                    P(:,k+1)=( (2*k-1)*x.*P(:,k)-(k-1)*P(:,k-1) )/k;
                end
                
                x=xold-( x.*P(:,N1)-P(:,N) )./( N1*P(:,N1) );
            end
            
            X=repmat(x,1,N1);
            Xdiff=X-X'+eye(N1);
            
            L=repmat(P(:,N1),1,N1);
            L(1:(N1+1):N1*N1)=1;
            D=(L./(Xdiff.*L'));
            D(1:(N1+1):N1*N1)=0;
            D(1)=(N1*N)/4;
            D(N1*N1)=-(N1*N)/4;
            
            % Linear map from[-1,1] to [a,b]
            xi=(a*(1-x)+b*(1+x))/2;        % added by Tsogbayar Tsednee
            
            % Compute the weights
            w=(b-a)./(N*N1*P(:,N1).^2);    % added by Tsogbayar Tsednee
            
    end
