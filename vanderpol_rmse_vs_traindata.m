% vanderppol_model_comparison
%
% Identifies Koopman models for simple nonlinear model and compares their
% predictions.

start_data = 300; % minimum amount of training data to start with       
inc_data = 100; % number of points to increment training data by

data_phys = cell(9,1);
data_data = cell(9,1);
data_comb = cell(9,1);

err_phys_tot = [];
err_data_tot = [];
err_comb_tot = [];

for ii = 1:9    % iterate through data amounts

    err_phys_run = [];
    err_data_run = [];
    err_comb_run = [];

    for i = 2:10 %[2:3,5,7:10]   % iterate through system versions

        %% Load in simple models (not arm models)

        sys_name = [ 'vanderpol(', num2str(i), ')' ];
        % sys_name = 'vanderpol(2)';

        % One to explain many systems
        temp_sys_name = 'vanderpol';
        % temp_sys_name = sys_name;
        real_sys_name = sys_name;
        datafile_name = sys_name;

        % % Many to explain one system
        % temp_sys_name = sys_name;
        % real_sys_name = 'vanderpol';
        % datafile_name = 'vanderpol';

        % datafile_name = 'vanderpol(2)_2024-02-15_17-50'; % DEBUG: manual override to try with noise
        % datafile_name = 'vanderpol(4)_2024-02-15_18-15'; % DEBUG: manual override to try with noise
        % datafile_name = 'vanderpol(7)_2024-02-15_18-29'; % DEBUG: manual override to try with noise
        load([ 'systems' , filesep , 'simulations_with_noise_025' , filesep , datafile_name , '.mat' ] );
        train_data = data(1:ii);
        % train_data = data(1);
        for j = 1 : length(train_data)
            if j == 1
                train_data{j}.t = train_data{j}.t(1:start_data,:);
                train_data{j}.u = train_data{j}.u(1:start_data,:);
                train_data{j}.x = train_data{j}.x(1:start_data,:);
                train_data{j}.y = train_data{j}.y(1:start_data,:);
                if isfield( train_data{j} , 'x_clean' )
                    train_data{j}.x_clean = train_data{j}.x_clean(1:start_data,:);   % DEBUG
                end
            else
                train_data{j}.t = train_data{j}.t(1:inc_data,:);
                train_data{j}.u = train_data{j}.u(1:inc_data,:);
                train_data{j}.x = train_data{j}.x(1:inc_data,:);
                train_data{j}.y = train_data{j}.y(1:inc_data,:);
                if isfield( train_data{j} , 'x_clean' )
                    train_data{j}.x_clean = train_data{j}.x_clean(1:inc_data,:);   % DEBUG
                end
            end
        end

        val_data = data(10);
        val_data{1}.t = val_data{1}.t(1:300,:);
        val_data{1}.u = val_data{1}.u(1:300,:);
        val_data{1}.x = val_data{1}.x(1:300,:);
        val_data{1}.y = val_data{1}.y(1:300,:);
        if isfield( val_data{1} , 'x_clean' )
            val_data{1}.x_clean = val_data{1}.x_clean(1:300,:);   % DEBUG
        end

        load([ 'systems' , filesep , temp_sys_name , '.mat' ] );
        sys_temp = sys;
        % sys_temp.x_domain = max(max(abs( train_data{1}.x ))) * [-1,1];
        sys_temp.x_domain = [-2.5,2.5];

        load([ 'systems' , filesep , real_sys_name , '.mat' ] );
        sys_real = sys;
        % sys_real.x_domain = max(max(abs( train_data{1}.x ))) * [-1,1];
        sys_real.x_domain = [-2.5,2.5];


        % specify timestep in the data
        dt = data{1}.t(2) - data{1}.t(1);

        %% Identify physics-based Koopman model

        clear Klift;
        % build class from template dynamics
        Klift = Klift( sys_temp ,...
            'model_type' , 'linear' ,...
            'basis_degree' , 3 ,...
            'basis_type' , 'hermite' ,...
            'has_massmtx' , true ,...
            'num_samples' , 1e4 ,...
            'integral_res' , 10 ,... % how finely to discretize each dimension when approximating IPs
            'timestep' , dt ... %1e-3 ...
            );


        %% Identify data-driven residual Koopman model (This assumes timestep of learned model is same as the data)

        clear Kres;
        Kres = Kres( Klift , train_data, 'lasso' , 0 );

        %% Compare the models

        comp_trial_data = val_data{1};
        % comp_trial_data.x = comp_trial_data.Q;    % create 'x' field for data (FOR ARMS)
        % comp_trial_data.x = [ comp_trial_data.y(:,end/2-2:end/2) , comp_trial_data.Q ];    % create 'x' field for data. For xyz_embed systems. DEBUG*******

        [ data_comb{ii}{i} , data_phys{ii}{i} , data_data{ii}{i} ] = Kres.compare_models( comp_trial_data );

        %% Quantify the overall error

        data_phys{ii}{i}.rmse = sum( sqrt( sum( (data_phys{ii}{i}.x - comp_trial_data.x).^2 , 2) ) ) / length( data_phys{ii}{i}.t );
        data_data{ii}{i}.rmse = sum( sqrt( sum( (data_data{ii}{i}.x - comp_trial_data.x).^2 , 2) ) ) / length( data_data{ii}{i}.t );
        data_comb{ii}{i}.rmse = sum( sqrt( sum( (data_comb{ii}{i}.x - comp_trial_data.x).^2 , 2) ) ) / length( data_comb{ii}{i}.t );

        % running ledger of error across all systems for ii-th data amount
        err_phys_run = [ err_phys_run ; data_phys{ii}{i}.x - comp_trial_data.x ];
        err_data_run = [ err_data_run ; data_data{ii}{i}.x - comp_trial_data.x ];
        err_comb_run = [ err_comb_run ; data_comb{ii}{i}.x - comp_trial_data.x ];

        % % running ledger of error across all systems
        % err_phys_tot = [ err_phys_tot ; data_phys{ii}{i}.x - comp_trial_data.x ];
        % err_data_tot = [ err_data_tot ; data_data{ii}{i}.x - comp_trial_data.x ];
        % err_comb_tot = [ err_comb_tot ; data_comb{ii}{i}.x - comp_trial_data.x ];

        % % plot phase space
        % figure; plot( comp_trial_data.x(:,1), comp_trial_data.x(:,2) );
        % hold on; plot( data_phys{ii}{i}.x(:,1), data_phys{ii}{i}.x(:,2) )
        % hold on; plot( data_data{ii}{i}.x(:,1), data_data{ii}{i}.x(:,2) )
        % hold on; plot( data_comb{ii}{i}.x(:,1), data_comb{ii}{i}.x(:,2) )
        % grid on; box on;
        % xlabel('x_1');
        % ylabel('x_2');
        % legend({'Real', 'Physics-based', 'Data-driven', 'Combined'}, 'Location', 'southeast');
    end

    % running ledger of error across all systems
    err_phys_tot = [ err_phys_tot ; err_phys_run ];
    err_data_tot = [ err_data_tot ; err_data_run ];
    err_comb_tot = [ err_comb_tot ; err_comb_run ];

    % Compute RMSE over each trial
    rmse_phys(ii) = sum( sqrt( sum( err_phys_run.^2 , 2) ) ) / length( err_phys_run );
    rmse_data(ii) = sum( sqrt( sum( err_data_run.^2 , 2) ) ) / length( err_data_run );
    rmse_comb(ii) = sum( sqrt( sum( err_comb_run.^2 , 2) ) ) / length( err_comb_run );

end

% Compute total RMSE over all trials
rmse_phys_tot = sum( sqrt( sum( err_phys_tot.^2 , 2) ) ) / length( err_phys_tot );
rmse_data_tot = sum( sqrt( sum( err_data_tot.^2 , 2) ) ) / length( err_data_tot );
rmse_comb_tot = sum( sqrt( sum( err_comb_tot.^2 , 2) ) ) / length( err_comb_tot );

%% plot error verses amount of data 

cmap = colormap("lines");   % Matlab's default color map

figure;
hold on;
plot((0:ii-1)*inc_data+start_data,rmse_phys, '--o', 'Color', cmap(2,:), 'MarkerFaceColor', cmap(2,:), 'MarkerSize', 8 );
plot((0:ii-1)*inc_data+start_data,rmse_data, '--o', 'Color', cmap(3,:), 'MarkerFaceColor', cmap(3,:), 'MarkerSize', 8 );
plot((0:ii-1)*inc_data+start_data,rmse_comb, '--o', 'Color', cmap(4,:), 'MarkerFaceColor', cmap(4,:), 'MarkerSize', 8 );
hold off;
grid on; box on;
xlabel('Training Data Snapshots');
ylabel('RMSE');
xlim([start_data,(ii-1)*inc_data+start_data]);
legend('Physics-based','Data-driven','Combined');
