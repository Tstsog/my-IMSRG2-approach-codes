% This Matlab code computes  the ground state energy for helium atom by solving the
% flow-type equation combined with an imaginary time (it) generator
% in the in-medium similarity renormalization group approach using 
% the cc-pVDZ basis set.
%
% The self-consistent field (SCF) calculation was carried out using the UNDMOL
% computational chemistry package at University of North Dakota; the
% two-electron integral (He_tei_data_cc_pvdz.txt) data was obtained from the SCF calculation. 
%
% Refs: H. Hergert, S. K. Bogner, J. G. Lietz, T. D. Morris, S. J. Novario, N.M. Parzuchowski, and F. Yuan,   
% Lecture Notes in Physics - An Advanced Course in Computational Nuclear Physics, volume, 936, 477 (2017).
%
% Written by Tsogbayar Tsednee (PhD)
% Email: tsog215@gmail.com
% July 12, 2023 & University of North Dakota 
%
function [] = he_cc_pvdz_imsrg2_it
%
clear; clc; format long
%
dim2B = 10; % size of spin-orbital basis sets
tol = 1e-3;
%
tei_n = 120;
%
read_tei_data = fopen('He_tei_data_cc_pvdz.txt', 'r');
tei_data_n5 = textscan(read_tei_data, '%d %d %d %d %f');
%
p = zeros(tei_n,1); q = zeros(tei_n,1); r = zeros(tei_n,1); s = zeros(tei_n,1); vals = zeros(tei_n,1);
p(1:tei_n) = tei_data_n5{1};
q(1:tei_n) = tei_data_n5{2};
r(1:tei_n) = tei_data_n5{3};
s(1:tei_n) = tei_data_n5{4};
vals(1:tei_n) = tei_data_n5{5};
for i = 1:tei_n
    tei(p(i),q(i),r(i),s(i)) = vals(i);
    tei(q(i),p(i),r(i),s(i)) = vals(i);    
    tei(p(i),q(i),s(i),r(i)) = vals(i);    
    tei(q(i),p(i),s(i),r(i)) = vals(i);   
    %
    tei(r(i),s(i),p(i),q(i)) = vals(i);    
    tei(s(i),r(i),p(i),q(i)) = vals(i);        
    tei(r(i),s(i),q(i),p(i)) = vals(i);        
    tei(s(i),r(i),q(i),p(i)) = vals(i);            
end
%
E0 =  -2.855160477243;                                          % the SCf ground state energy
En_scf = E0;
En_orb = [2.524372,  2.524372, 2.524372, 1.397442, -0.914148];  % An orbital energies
%
f = kron(diag(En_orb), eye(2)); % Fock matrix elements in molecular spin-orbital basis sets
%
Nhole_n = [10, 9]; % hole states
Nvir_n  = [8, 7, 6, 5, 4, 3, 2, 1]; % virtial (particle) states 
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
fileID_save_data_1 = fopen('IMSRG2_im_time_He_cc_pvdz_imag_time.txt','w');
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
    output = [iter*ds, E0, E_mbpt23, f_od, Gamma_od]
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
% [En_scf, En_imsrg2] = -2.855160477243  -2.887908663011788 

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