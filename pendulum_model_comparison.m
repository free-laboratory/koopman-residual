% pendulum_cart_model_comparison
%
% Identifies Koopman models for simple nonlinear model and compares their
% predictions.

data_phys = cell(9,1);
data_data = cell(9,1);
data_comb = cell(9,1);

err_phys_tot = [];
err_data_tot = [];
err_comb_tot = [];



%% Load in simple models (not arm models)

% sys_name = 'elastic_pendulum';
% sys_name = 'simple_linear';
% sys_name = 'pendulum';
% sys_name = 'pendulum_varlen';
sys_name = 'cargo_crane_real';
% sys_name = 'elastic_pendulum_damped';

% template system model
% temp_sys_name = 'elastic_pendulum';
% temp_sys_name = 'simple_linear';
% temp_sys_name = 'pendulum';
% temp_sys_name = 'pendulum_varlen';
temp_sys_name = 'cargo_crane_temp';
% temp_sys_name = 'elastic_pendulum_damped';

real_sys_name = sys_name;
datafile_name = sys_name;

% % Many to explain one system
% temp_sys_name = sys_name;
% real_sys_name = 'vanderpol';
% datafile_name = 'vanderpol';

load([ 'systems' , filesep , 'simulations_with_noise_025' , filesep , datafile_name , '.mat' ] );
train_data = data(2:2);
% train_data = data(1);
for j = 1 : length(train_data)
    train_data{j}.t = train_data{j}.t(1:200,:);
    train_data{j}.u = train_data{j}.u(1:200,:);
    train_data{j}.x = train_data{j}.x(1:200,:);
    train_data{j}.y = train_data{j}.y(1:200,:);
    if isfield( train_data{j} , 'x_clean' )
        train_data{j}.x_clean = train_data{j}.x_clean(1:200,:);   % DEBUG
    end
end

val_data = data(1); % DEBUG: should be data(10), but want to check overfitting
for j = 1 : length(val_data)
    val_data{j}.t = val_data{j}.t;%(1:300,:);
    val_data{j}.u = val_data{j}.u;%(1:300,:);
    val_data{j}.x = val_data{j}.x;%(1:300,:);
    val_data{j}.y = val_data{j}.y;%(1:300,:);
    if isfield( val_data{1} , 'x_clean' )
        val_data{j}.x_clean = val_data{j}.x_clean;%(1:300,:);   % DEBUG
    end
end

load([ 'systems' , filesep , temp_sys_name , '.mat' ] );
sys_temp = sys;
% sys_temp.x_domain = max(max(abs( train_data{1}.x ))) * [-1,1];
% sys_temp.x_domain = [-2.5,2.5];

% load([ 'systems' , filesep , real_sys_name , '.mat' ] );
% sys_real = sys;
% % sys_real.x_domain = max(max(abs( train_data{1}.x ))) * [-1,1];
% sys_real.x_domain = [-2.5,2.5];


% specify timestep in the data
dt = data{1}.t(2) - data{1}.t(1);

%% Identify physics-based Koopman model

clear Klift;
% build class from template dynamics
Klift = Klift( sys_temp ,...
    'model_type' , 'linear' ,...
    'basis_degree' , 6 ,...
    'basis_type' , 'hermite' ,...
    'has_massmtx' , true ,...
    'num_samples' , 1e6 ,... % 4e6
    'integral_res' , 10 ,... % how finely to discretize each dimension when approximating IPs
    'integration_type', 'montecarlo' ,...
    'timestep' , dt ... %1e-3 ...
    );


%% Identify data-driven residual Koopman model (This assumes timestep of learned model is same as the data)

clear Kres;
Kres = Kres( Klift , train_data, 'lasso' , 0.0);

%% Compare the models

comp_trial_data = val_data{1};
% comp_trial_data.x = comp_trial_data.Q;    % create 'x' field for data (FOR ARMS)
% comp_trial_data.x = [ comp_trial_data.y(:,end/2-2:end/2) , comp_trial_data.Q ];    % create 'x' field for data. For xyz_embed systems. DEBUG*******

[ data_comb , data_phys , data_data ] = Kres.compare_models( comp_trial_data , 1);

%% Quantify the overall error

data_phys.rmse = sum( sqrt( sum( (data_phys.x - comp_trial_data.x).^2 , 2) ) ) / length( data_phys.t );
data_data.rmse = sum( sqrt( sum( (data_data.x - comp_trial_data.x).^2 , 2) ) ) / length( data_data.t );
data_comb.rmse = sum( sqrt( sum( (data_comb.x - comp_trial_data.x).^2 , 2) ) ) / length( data_comb.t );

% running ledger of error across all systems
err_phys_tot = [ err_phys_tot ; data_phys.x - comp_trial_data.x ];
err_data_tot = [ err_data_tot ; data_data.x - comp_trial_data.x ];
err_comb_tot = [ err_comb_tot ; data_comb.x - comp_trial_data.x ];

% plot phase space
figure; plot( comp_trial_data.x(:,1), comp_trial_data.x(:,2) );
hold on; plot( data_phys.x(:,1), data_phys.x(:,2) )
hold on; plot( data_data.x(:,1), data_data.x(:,2) )
hold on; plot( data_comb.x(:,1), data_comb.x(:,2) )
grid on; box on;
xlabel('x_1');
ylabel('x_2');
legend({'Real', 'Physics-based', 'Data-driven', 'Combined'}, 'Location', 'southeast');


%% Compute total RMSE over all trials
rmse_phys_tot = sum( sqrt( sum( err_phys_tot.^2 , 2) ) ) / length( err_phys_tot );
rmse_data_tot = sum( sqrt( sum( err_data_tot.^2 , 2) ) ) / length( err_data_tot );
rmse_comb_tot = sum( sqrt( sum( err_comb_tot.^2 , 2) ) ) / length( err_comb_tot );



