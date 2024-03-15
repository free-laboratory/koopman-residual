% simulate_sys
%
% Use this script to generate simulation data of a system

%% Set values of simulation parameters (USER)

% sys_name = 'simple_nonlinear(10)';
% sys_name = 'vanderpol(10)';
% sys_name = 'duffing';
% sys_name = 'new_duffing';
% sys_name = 'simple_linear';
% sys_name = 'pendulum';
% sys_name = 'vanderpol(10)';
% sys_name = 'arm_3-mods_6-links_mass-50g_kjoints-5_damp-05_ufixed(2)';
% sys_name = 'arm_2-mods_4-links_mass-50g_kjoints-2pi_damp-01';
% sys_name = 'double_pend';
% sys_name = 'arm_2-mods_4-links_mass-50g_kjoints-1o2pi_damp-0';
% sys_name = 'arm_2-mods_4-links_mass-50g_kcubic_damp-001';
% sys_name = 'arm_2-mods_4-links_mass-1kg_k-0_damp-001_u-se';
% sys_name = 'arm_1-mods_2-links_mass-50g_k-0_damp-001_u-se(3)';
% sys_name = 'elastic_pendulum';
% sys_name = 'simple_linear';
% sys_name = 'pendulum_varlen';
% sys_name = 'pendulum';
% sys_name = 'elastic_pendulum_damped';
% sys_name = 'cargo_crane_M1';    % dt = 1/50
% sys_name = 'psi-arm_3-mods_14-links_mass-50g_kjoints-5_damp-05_ufixed_Bf-79'; % dt = 1/25
sys_name = 'cargo_crane_real';  

num_trials = 60;    % number of separate trials
dt = (1/50);% 1e-3; %(1/100); % (1/50) % timestep length
tend = 30;  % length of each trial
num_ramps = 15;  % number of ramps per trial

%% Initialize dependent parameter values

% Load system object into the workspace
sys_path = 'systems';
load([ sys_path , filesep , sys_name , '.mat' ]);

% sys.u_domain = [-10,10];    % DEBUG!!!!!!! For M1 model only

t_trial = (0 : dt : tend)';
t_trial_ramps = linspace(0, tend, num_ramps)';

data = cell( num_trials , 1 );

%% Simulate trials

for i = 1 : num_trials
    % x0_rand = 2*rand( size(sys.x) ) - 1;  % random starting state in [-1 , 1]
    % x0_rand = (sys.x_domain(:,2) - sys.x_domain(:,1)) .* rand( size(sys.x) ) + sys.x_domain(:,1);  % random starting state
    % x0_rand = zeros(size(sys.x));   % DEBUG: Zero initial condition
    x0_rand = [ (pi/2)*rand( length(sys.x)/2 , 1 ) - pi/2 ; zeros( length(sys.x)/2 , 1) ];  % random starting state in [-pi/2, pi/2], (for cargo crane)

    % RAMP INPUTS
%     ramp_points = 2 * Arm.params.umax * (2*rand( num_ramps , Arm.params.nu ) - 1);  % random sequence of inputs (make sure not too large or state bounds will be violated)
%     ramp_points = 200e3 * rand( num_ramps , Arm.params.nu );
%     ramp_points = Arm.params.umax * rand( num_ramps , Arm.params.nu ); % defined in Arm_setup (positive only)
    ramp_points = (sys.u_domain(:,2)-sys.u_domain(:,1))' .* rand( num_ramps , length(sys.u) ) + sys.u_domain(:,1)' ; % drawn from [-umax,umax]
    % ramp_points = ones(num_ramps, length(sys.u));    % set input to constant 1 for vanderpol
    % ramp_points = 0.2 * rand(1,1) * ones(num_ramps, length(sys.u));    % set input to constant in [0,0.2] for duffing

    if num_ramps > 1
        uin_rand = interp1( t_trial_ramps , ramp_points , t_trial );% ramps
        % uin_rand = interp1( t_trial_ramps , ramp_points , t_trial , 'previous');% steps
    else
        uin_rand = ramp_points .* ones(size(t_trial));  % constant input of over whole trial
    end

    % simulate arm one timestep at a time
    data{i}.t = t_trial(1);
    data{i}.u = uin_rand(1,:);
    data{i}.x = x0_rand';
    data{i}.x_clean = data{i}.x(1,:);
    data{i}.y = sys.get_y( x0_rand' );
    for j = 2 : length(t_trial)    % should preallocate vars for speed
        options = odeset( 'Mass' , @(t,x) sys.vf_massMtx( t , x , [] ) );
        [ tout , xout ] = ode45( @(t,x) sys.vf_RHS(t,x,uin_rand(j-1,:)') , [t_trial(j-1) , t_trial(j)] , data{i}.x_clean(end,:)' , options);
        meas_noise = normrnd(0, 0.025, size( xout(end,:) ) );% mu = 0, sigma = 1;
        data{i}.t = [ data{i}.t ; tout(end) ];
        data{i}.x_clean = [ data{i}.x_clean ; xout(end,:) ];
        data{i}.x = [ data{i}.x ; xout(end,:) + meas_noise ];
        data{i}.y = [ data{i}.y ; sys.get_y( xout(end,:) + meas_noise ) ];
        data{i}.u = [ data{i}.u ; uin_rand(j,:) ];
    end
end

%% Save data

save_path = [ sys_path , filesep , 'simulations_with_noise_025' ];
file_name = [ sys_name, '_' , datestr( now , 'yyyy-mm-dd_HH-MM' ) , '.mat' ];
% file_name = [ sys_name, '.mat' ];
save( [ save_path , filesep , file_name ] , 'data' );
