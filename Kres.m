classdef Kres
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        kphys;  % stores a copy of the Klift class
        num_trials;
        timestep;
        data;
        model;  % LTI system matrices for 
        
        K_phys;
        K_residual; % discrete Koopman matrix describing residual model
        K_data; % discrete Koopman matrix for pure data driven model
        K_data_half;    % data driven model from half as much training data
        K_combined; % combined physics and data driven koopman model
        scaleup;
        scaledown;
        PCmtx;  % matrix of principle components of the data
        lasso;  % value of lasso regularization parameter
    end

    methods
        function obj = Kres( Klift , data , varargin )
            %Kres Construct an instance of this class
            %   Fits a Koopman residual model based on data and a
            %   physics-based Koopman model.
            %
            %   Inputs:
            %       Klift - Physics-based Koopman model class
            %       data - cell array containing data collected from actual
            %              sytem
            %       varargin - Name, Value pairs specifying various options

            obj.kphys = Klift;
            obj.timestep = Klift.timestep;

            obj.K_phys = expm( Klift.Kmtx * obj.timestep );
            % % DEBUG: scale K_phys so it doesn't have unstable eigenvalues
            % obj.K_phys = obj.K_phys ./ max(abs(eig(obj.K_phys)));
            
            obj.num_trials = length(data);
            obj.data = data;    % save copy of the training data in class (may remove later)
            obj.scaleup = Klift.scaleup;
            obj.scaledown = Klift.scaledown;
            obj.lasso = 0;  % default to least-squares solution

            % replace default values with user input values
            obj = obj.parse_args( varargin{:} );

            % identify the residual Koopman model
            obj = obj.get_K_residual( data );

            if strcmp( obj.kphys.model_type,'linear' )
                obj = obj.get_lti_models;
            end
        end

        function obj = parse_args( obj , varargin )
            %parse_args Parses the Name, Value pairs in varargin of the
            % constructor, and assigns property values
            for idx = 1:2:length(varargin)
                obj.(varargin{idx}) = varargin{idx+1} ;
            end
        end

        %% Identify residual model

        % NEW VERSION
        function obj = get_K_residual_new( obj , data )
            %get_K_residual Identify residual Koopman model from data
            %   Detailed explanation goes here
            
            % assemble snapshot pairs (ONLY WORKS FOR LINEAR MODELS)
            alpha = [];
            beta = [];
            beta_dataonly = [];
            beta_phys = [];
            for i = 1 : obj.num_trials
                x = data{i}.x;  % FOR NON-ARM SYSTEMS

                data_basis = obj.kphys.lift.basis.x( x' )';     % lifting functions are defined in template model class
%                 data_fullbasis = obj.kphys.lift.fullbasis.x( x' , data{i}.u' )'; % lifting functions are defined in template model class
                data_fullbasis_a = obj.kphys.lift.fullbasis.x( x(1:end-1,:)' , data{i}.u(1:end-1,:)' )'; % lifting functions are defined in template model class
                data_fullbasis_b = obj.kphys.lift.fullbasis.x( x(2:end,:)' , data{i}.u(1:end-1,:)' )';

%                 alpha_i = data_fullbasis( 1 : end-1 , : );
                alpha_i = data_fullbasis_a;    % ( 1 : end-1 , : );

%                 beta_i = [ data_basis( 2 : end , : ) , data{i}.u(1:end-1,:) ] - data_fullbasis( 1 : end-1 , : ) * obj.scaledown.z * obj.K_phys' * obj.scaleup.z;
                beta_i = data_fullbasis_b - data_fullbasis_a * obj.scaledown.z * obj.K_phys' * obj.scaleup.z;
                
                beta_dataonly_i = data_fullbasis_b;  % for learning pure data-driven model
                beta_phys_i = data_fullbasis_a * obj.scaledown.z * obj.K_phys' * obj.scaleup.z;

                alpha = [ alpha ; alpha_i ];
                beta = [ beta ; beta_i ];
                beta_dataonly = [ beta_dataonly ; beta_dataonly_i ];
                beta_phys = [ beta_phys ; beta_phys_i ];
            end

            % find scaling factor so no data is larger than 1
%             obj.scaleup.res = diag( max( abs( [ alpha ; beta ] ) ) );    % diagonal matrix of largest elemnt of each state
            obj.scaleup.res = diag( max( abs( alpha ) ) );    % diagonal matrix of largest element of each state
            obj.scaleup.res_err = diag( max( abs( beta ) ) );
            % obj.scaleup.res = eye(obj.kphys.params.Nfull);  % DEBUG: remove scaling %%%%%%%%%%%%%%%%%%
            obj.scaledown.res =  obj.scaleup.res^(-1);
            obj.scaledown.res_err =  obj.scaleup.res_err^(-1);
            obj.scaledown.res(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs
            obj.scaledown.res_err(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs

            % Scale the data
            alpha_scaled = alpha * obj.scaledown.res;
            % beta_scaled = beta * obj.scaledown.res;
            beta_scaled = beta * obj.scaledown.res_err;
            beta_dataonly_scaled = beta_dataonly * obj.scaledown.res;

            %% get PCmtx

            % DEBUG: Do PCA on residual data to reduce the number of basis functions
            diff_data = beta_dataonly - alpha; % DEBUG: Do PCA on unscaled data (shouldn't actually matter since both scaled the same)
            % diff_data = beta_dataonly_scaled - alpha_scaled; % THIS WORKS WAY BETTER THAN DOING PCA ON RESIDUAL DATA
            % diff_data = beta; % just use error between physics based model and real
            % diff_data = [ alpha_scaled; beta_dataonly_scaled];  % DEBUG want basis that describes all data points (DOES NOT WORK, RESPONSE GOES TO 0 or INF)

            % [PCmtx,score,latent,tsquared,explained,mu] = pca(diff_data, 'Centered', true);
            % % PCmtx = fliplr(PCmtx);  % order PCs in order of least to most variance
            % % explained = flipud(explained);
            % for P = 1 : size(PCmtx,2)   % take first P PCs that explain 99% of data
            %     if sum(explained(1:P)) > 99.9
            %         break;
            %     end
            % end

            % P=obj.kphys.params.Nfull;    % DEBUG: fix value of P

            [ PCmtx, P ] = obj.cull_basis( diff_data , 0.66 );   % DEBUG: Try manually culling basis functions rather than PCA

            % PCmtx = orth( diff_data' ); % DEBUG 2: Try using null space projection as transformation
            % P = size(PCmtx,2);
            
            % foo = round( pinv(alpha)*beta_dataonly(:,1:obj.kphys.params.n), 6);   % round small components down to zero  % round small components down to zero
            % bar = double( logical( abs( diag( sum(foo,2) ) ) ) );   % matrix that eliminates certain lifted states   
            % PCmtx = bar(:, find(sum(bar) ~= 0));  % DEBUG 3: Only keep lifted states that help predictions of the actual states the most
            % P = size(PCmtx,2);

            % PCmtx = eye(obj.kphys.params.Nfull);    % don't change basis at all
            % P = size(PCmtx,2);
            
            obj.PCmtx = PCmtx;  % save PC matrix as a class property
            
            %% A bunch of new scaling for the reduced order snapshot pairs

            A_data = alpha_scaled * PCmtx(:,1:P);   % DEBUG
            % B_data = beta_scaled * PCmtx(:,1:P); % DEBUG

            % beta = beta_dataonly * PCmtx(:,1:P) - alpha * PCmtx(:,1:P) * PCmtx(:,1:P)' * (obj.scaledown.z * obj.K_phys' * obj.scaleup.z) * PCmtx(:,1:P);    % DEBUG: have to recompute diff with reduced basis
            obj.scaleup.res_err_PC = diag( max( abs( beta ) ) );
            obj.scaledown.res_err_PC =  obj.scaleup.res_err_PC^(-1);
            obj.scaleup.res_PC = PCmtx(:,1:P)' * obj.scaleup.res * PCmtx(:,1:P);    % P x P scaling matrix
            obj.scaledown.res_PC =  obj.scaleup.res_PC^(-1);
            obj.scaledown.res_PC(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs
            obj.scaledown.res_err_PC(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs
            B_data = beta * obj.scaledown.res_err_PC;

            %%

            % Identify residual Koopman using Lasso, one column at a time
            N = obj.kphys.params.Nfull;
            K = size( alpha_scaled , 1 );
            Kres_trans = zeros(P);  % N: DEBUG
            for i = 1 : P % DEBUG
                b_data = B_data(:,i);
                % Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false );   % No Lasso penalty, used for real arm system (allegedly)
                % Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2.95e-2, 'Intercept',false ); % best I've been able to get for real arm system, not sure how zero lasso worked before ****
                % Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2e-3, 'Intercept',false ); % this number works well 
                % Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0.39e-4, 'Intercept',false );   % 2e-2 for bilinear2
%                 Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2e-2, 'Intercept',false );   % 2e-2 for bilinear2
%                 Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 4e-3, 'Intercept',false );   % for psi 3 module model
                Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false ); % this number works well for vanderpol model
            end
            % Kres_trans = pinv(A_data) * B_data;   % DEBUG: straight least squares instead of Lasso
            % obj.K_residual = Kres_trans';
            obj.K_residual = obj.scaleup.res_err_PC * Kres_trans' * obj.scaledown.res_PC;   % do scaling here instead of in K_combined
            obj.K_residual = PCmtx(:,1:P) * obj.K_residual * PCmtx(:,1:P)'; % DEBUG: make it NxN again (temporary)
            % obj.K_residual(find(abs(obj.K_residual)<1e-2)) = 0; % DEBUG: round small matrix elements down to zero


            % DEBUG: Do PCA on pure data to reduce the number of basis functions
            diff_data = beta_dataonly - alpha; % DEBUG: Do PCA on unscaled data (shouldn't actually matter since both scaled the same)
            % diff_data = beta_dataonly_scaled - alpha_scaled;

            % [PCmtx,score,latent,tsquared,explained,mu] = pca(diff_data, 'Centered' , false );
            % for P = 1 : size(PCmtx,2)   % take first P PCs that explain 99% of data
            %     if sum(explained(1:P)) > 99
            %         break;
            %     end
            % end
            % P=obj.kphys.params.Nfull;    % DEBUG: fix value of P
            [ PCmtx, P ] = obj.cull_basis( diff_data , 0.1e-6 );   % DEBUG: Try manually culling basis functions rather than PCA
            A_data = alpha_scaled * PCmtx(:,1:P);   % DEBUG
            B_dataonly = beta_dataonly_scaled * PCmtx(:,1:P); % DEBUG

            % Identify data-driven only Koopman using Lasso, one column at a time
            N = obj.kphys.params.Nfull;
            Kdata_trans = zeros(P); % DEBUG: zeros(N);
            for i = 1 : P   % DEBUG: was N before
                b_data = B_dataonly(:,i);
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false ); % No Lasso penalty
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2e-3, 'Intercept',false ); % this number works well
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 1e-4, 'Intercept',false );
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0.5e-4, 'Intercept',false ); % used for real arm system ****
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 1e-3, 'Intercept',false ); % this number works well for vanderpol model with limited data
                Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false ); % this number works well for vanderpol model with 10x data
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 5e-4, 'Intercept',false ); % this number works well for duffing model
            end
            % Kdata_trans = pinv(A_data) * B_dataonly;   % DEBUG: straight least squares instead of Lasso
            obj.K_data = Kdata_trans';
            obj.K_data = PCmtx(:,1:P) * obj.K_data * PCmtx(:,1:P)'; % DEBUG: make it NxN again (temporary)
            % obj.K_data(find(abs(obj.K_data)<1e-2)) = 0; % DEBUG: round small matrix elements down to zero


            % % DEBUG: scale down K_data by its largest eigenvalue
            % max_eig_kdata = max( abs( eig(obj.K_data) ) );
            % obj.K_data = obj.K_data ./ (max_eig_kdata + 1e-4);

%             % Only care about correcting value of x (not lifting functions)
%             N = obj.kphys.params.Nfull;
%             beta_scaled = beta_scaled(:,1:obj.kphys.params.n);  % just first n components
%             Kres_trans_rect = pinv(alpha_scaled) * ( beta_scaled );
%             Kres_rect = Kres_trans_rect';
%             obj.K_residual = [ Kres_rect ; zeros(N-obj.kphys.params.n , N) ];

            % model that combines the physics-based and residual models
            % obj.K_combined = ( obj.scaleup.res * obj.K_residual * obj.scaledown.res + obj.scaleup.z * obj.K_phys * obj.scaledown.z );
            % obj.K_combined = ( obj.scaleup.res_err * obj.K_residual * obj.scaledown.res + (PCmtx*PCmtx') * obj.scaleup.z * obj.K_phys * obj.scaledown.z * (PCmtx*PCmtx') );  % added the PCmtx transform to the physics-based part
            obj.K_combined = ( obj.K_residual + (PCmtx(:,1:P)*PCmtx(:,1:P)') * obj.scaleup.z * obj.K_phys * obj.scaledown.z * (PCmtx(:,1:P)*PCmtx(:,1:P)') );  % added the PCmtx transform to the physics-based part, and do scaling of K_residual before here

            % % DEBUG: scale down K_combined by its largest eigenvalue
            % max_eig_kcombined = max( abs( eig(obj.K_combined) ) );
            % obj.K_combined = obj.K_combined ./ (max_eig_kcombined);
        end

        % VERSION with weighted residual matrix
        function obj = get_K_residual( obj , data )
            %get_K_residual Identify residual Koopman model from data
            %   Detailed explanation goes here
            
            % assemble snapshot pairs (ONLY WORKS FOR LINEAR MODELS)
            alpha = [];
            beta = [];
            beta_dataonly = [];
            beta_phys = [];
            for i = 1 : obj.num_trials
                x = data{i}.x;  % FOR NON-ARM SYSTEMS

                data_basis = obj.kphys.lift.basis.x( x' )';     % lifting functions are defined in template model class
%                 data_fullbasis = obj.kphys.lift.fullbasis.x( x' , data{i}.u' )'; % lifting functions are defined in template model class
                data_fullbasis_a = obj.kphys.lift.fullbasis.x( x(1:end-1,:)' , data{i}.u(1:end-1,:)' )'; % lifting functions are defined in template model class
                data_fullbasis_b = obj.kphys.lift.fullbasis.x( x(2:end,:)' , data{i}.u(1:end-1,:)' )';

%                 alpha_i = data_fullbasis( 1 : end-1 , : );
                alpha_i = data_fullbasis_a;    % ( 1 : end-1 , : );

%                 beta_i = [ data_basis( 2 : end , : ) , data{i}.u(1:end-1,:) ] - data_fullbasis( 1 : end-1 , : ) * obj.scaledown.z * obj.K_phys' * obj.scaleup.z;
                beta_i = data_fullbasis_b - data_fullbasis_a * obj.scaledown.z * obj.K_phys' * obj.scaleup.z;
                
                beta_dataonly_i = data_fullbasis_b;  % for learning pure data-driven model
                beta_phys_i = data_fullbasis_a * obj.scaledown.z * obj.K_phys' * obj.scaleup.z;

                alpha = [ alpha ; alpha_i ];
                beta = [ beta ; beta_i ];
                beta_dataonly = [ beta_dataonly ; beta_dataonly_i ];
                beta_phys = [ beta_phys ; beta_phys_i ];
            end

            % % shuffle the order of the snapshots (rows)
            % new_order = randperm( size(alpha,1) );
            % alpha = alpha(new_order,:);
            % beta = beta(new_order,:);
            % beta_dataonly = beta_dataonly(new_order,:);
            % beta_phys = beta_phys(new_order,:);

            % find scaling factor so no data is larger than 1
%             obj.scaleup.res = diag( max( abs( [ alpha ; beta ] ) ) );    % diagonal matrix of largest elemnt of each state
            obj.scaleup.res = diag( max( abs( alpha ) ) );    % diagonal matrix of largest element of each state
            obj.scaleup.res_err = diag( max( abs( beta ) ) );
            % obj.scaleup.res = eye(obj.kphys.params.Nfull);  % DEBUG: remove scaling %%%%%%%%%%%%%%%%%%
            obj.scaledown.res =  obj.scaleup.res^(-1);
            obj.scaledown.res_err =  obj.scaleup.res_err^(-1);
            % obj.scaledown.res(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs
            % obj.scaledown.res_err(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs

            % Scale the data
            alpha_scaled = alpha * obj.scaledown.res;
            % beta_scaled = beta * obj.scaledown.res;
            beta_scaled = beta * obj.scaledown.res_err;
            beta_dataonly_scaled = beta_dataonly * obj.scaledown.res;

            %% get PCmtx

            % DEBUG: Do PCA on residual data to reduce the number of basis functions
            diff_data = beta_dataonly - alpha; % DEBUG: Do PCA on unscaled data (shouldn't actually matter since both scaled the same)
            % diff_data = beta_dataonly_scaled - alpha_scaled; % THIS WORKS WAY BETTER THAN DOING PCA ON RESIDUAL DATA
            % diff_data = beta; % just use error between physics based model and real
            % diff_data = [ alpha_scaled; beta_dataonly_scaled];  % DEBUG want basis that describes all data points (DOES NOT WORK, RESPONSE GOES TO 0 or INF)

            % [PCmtx,score,latent,tsquared,explained,mu] = pca(diff_data, 'Centered', false);
            % % PCmtx = fliplr(PCmtx);  % order PCs in order of least to most variance
            % % explained = flipud(explained);
            % for P = 1 : size(PCmtx,2)   % take first P PCs that explain 99% of data
            %     if sum(explained(1:P)) > 99.9
            %         break;
            %     end
            % end

            % P=obj.kphys.params.Nfull;    % DEBUG: fix value of P

            % [ PCmtx, P ] = obj.cull_basis( diff_data , 0.66 );   % DEBUG: Try manually culling basis functions rather than PCA

            % PCmtx = orth( diff_data' ); % DEBUG 2: Try using null space projection as transformation
            % P = size(PCmtx,2);
            
            % foo = round( pinv(alpha)*beta_dataonly(:,1:obj.kphys.params.n), 6);   % round small components down to zero  % round small components down to zero
            % bar = double( logical( abs( diag( sum(foo,2) ) ) ) );   % matrix that eliminates certain lifted states   
            % PCmtx = bar(:, find(sum(bar) ~= 0));  % DEBUG 3: Only keep lifted states that help predictions of the actual states the most
            % P = size(PCmtx,2);

            PCmtx = eye(obj.kphys.params.Nfull);    % don't change basis at all
            P = size(PCmtx,2);
            
            obj.PCmtx = PCmtx;  % save PC matrix as a class property
            
            %% A bunch of new scaling for the reduced order snapshot pairs

            A_data = alpha_scaled * PCmtx(:,1:P);   % DEBUG
            % B_data = beta_scaled * PCmtx(:,1:P); % DEBUG

            % beta = beta_dataonly * PCmtx(:,1:P) - alpha * PCmtx(:,1:P) * PCmtx(:,1:P)' * (obj.scaledown.z * obj.K_phys' * obj.scaleup.z) * PCmtx(:,1:P);    % DEBUG: have to recompute diff with reduced basis
            obj.scaleup.res_err_PC = diag( max( abs( beta ) ) );
            obj.scaledown.res_err_PC =  obj.scaleup.res_err_PC^(-1);
            obj.scaleup.res_PC = PCmtx(:,1:P)' * obj.scaleup.res * PCmtx(:,1:P);    % P x P scaling matrix
            obj.scaledown.res_PC =  obj.scaleup.res_PC^(-1);
            % obj.scaledown.res_PC(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs
            % obj.scaledown.res_err_PC(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs
            B_data = beta * obj.scaledown.res_err_PC;

            %%

            % Identify data-driven only Koopman using Lasso, one column at a time
            A_data = alpha_scaled * PCmtx(:,1:P);   % DEBUG
            B_dataonly = beta_dataonly_scaled * PCmtx(:,1:P); % DEBUG
            N = obj.kphys.params.Nfull;
            Kdata_trans = zeros(P); % DEBUG: zeros(N);
            % for i = 1 : P   % DEBUG: was N before
            %     b_data = B_dataonly(:,i);
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false ); % No Lasso penalty
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2e-3, 'Intercept',false ); % this number works well
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 1e-4, 'Intercept',false );
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0.5e-4, 'Intercept',false ); % used for real arm system ****
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 1e-3, 'Intercept',false ); % this number works well for vanderpol model with limited data
            %     Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', obj.lasso, 'Intercept',false ); % this number works well for vanderpol model with 10x data
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 5e-4, 'Intercept',false ); % this number works well for duffing model
            % end
            Kdata_trans = pinv(A_data) * B_dataonly;   % DEBUG: straight least squares instead of Lasso
            obj.K_data = Kdata_trans';
            obj.K_data = PCmtx(:,1:P) * obj.K_data * PCmtx(:,1:P)'; % DEBUG: make it NxN again (temporary)
            % obj.K_data(find(abs(obj.K_data)<1e-2)) = 0; % DEBUG: round small matrix elements down to zero

            % Split the training data
            % alpha_half{1} = alpha(1:floor(end - end*(1/9)),:);
            % alpha_half{2} = alpha(floor(end - end*(1/9))+1:end,:);
            % beta_dataonly_half{1} = beta_dataonly(1:floor(end - end*(1/9)),:);
            % beta_dataonly_half{2} = beta_dataonly(floor(end - end*(1/9))+1:end,:);
            %
            alpha_half{1} = alpha(1:floor(end - end*(0.1)),:);
            alpha_half{2} = alpha(floor(end - end*(0.1))+1:end,:);
            beta_dataonly_half{1} = beta_dataonly(1:floor(end - end*(0.1)),:);
            beta_dataonly_half{2} = beta_dataonly(floor(end - end*(0.1))+1:end,:);
            %
            % alpha_half{1} = alpha;
            % alpha_half{2} = alpha;
            % beta_dataonly_half{1} = beta_dataonly;
            % beta_dataonly_half{2} = beta_dataonly;

            alpha_half_scaled{1} = alpha_half{1} * obj.scaledown.res;   
            alpha_half_scaled{2} = alpha_half{2} * obj.scaledown.res;
            beta_dataonly_half_scaled{1} = beta_dataonly_half{1} * obj.scaledown.res;
            beta_dataonly_half_scaled{2} = beta_dataonly_half{2} * obj.scaledown.res;
            % alpha_scaled_half{1} = alpha_scaled(1:floor(end - end/2),:);
            % alpha_scaled_half{2} = alpha_scaled(floor(end - end/2)+1:end,:);
            % beta_scaled_dataonly_half{1} = beta_dataonly_scaled(1:floor(end - end/2),:);
            % beta_scaled_dataonly_half{2} = beta_dataonly_scaled(floor(end - end/2)+1:end,:);

            % Identify data-driven model from just first half of the data (should put this in its own function)
            A_data = alpha_half_scaled{1} * PCmtx(:,1:P);   % DEBUG
            B_dataonly = beta_dataonly_half_scaled{1} * PCmtx(:,1:P); % DEBUG
            N = obj.kphys.params.Nfull;
            Kdata_trans = zeros(P); % DEBUG: zeros(N);
            % for i = 1 : P   % DEBUG: was N before
            %     b_data = B_dataonly(:,i);
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false ); % No Lasso penalty
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2e-3, 'Intercept',false ); % this number works well
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 1e-4, 'Intercept',false );
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0.5e-4, 'Intercept',false ); % used for real arm system ****
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 1e-3, 'Intercept',false ); % this number works well for vanderpol model with limited data
            %     Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', obj.lasso, 'Intercept',false ); % this number works well for vanderpol model with 10x data
            %     % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 5e-4, 'Intercept',false ); % this number works well for duffing model
            % end
            Kdata_trans = pinv(A_data) * B_dataonly;   % DEBUG: straight least squares instead of Lasso
            obj.K_data_half = Kdata_trans';
            obj.K_data_half = PCmtx(:,1:P) * obj.K_data_half * PCmtx(:,1:P)'; % DEBUG: make it NxN again (temporary)
            
            % Define K_residual: Residual Koopman matrix
            obj.K_residual = obj.scaleup.res * obj.K_data_half * obj.scaledown.res - obj.K_phys;
            % obj.K_residual = obj.scaleup.res * obj.K_data * obj.scaledown.res - obj.K_phys;

            % obj.K_data = obj.K_data_half; % DEBUG, only letting the data driven model use half the data

            % Identify the optimal weight for the residual matrix
            lambda = obj.get_residual_weight( obj.K_phys, obj.K_residual, alpha_half{2}, beta_dataonly_half{2} );
            % lambda = obj.get_residual_weight( obj.K_phys, obj.K_residual, alpha_half_scaled{2}, beta_dataonly_half_scaled{2} );
            % lambda = obj.get_residual_weight( obj.K_phys, obj.K_residual, alpha_scaled, beta_dataonly_scaled );
            % lambda = obj.get_residual_weight( obj.K_phys, obj.K_residual, alpha, beta_dataonly );

            % Define K_combined: model that combines the physics-based and residual models
            % obj.K_combined = ( obj.scaleup.res * obj.K_residual * obj.scaledown.res + obj.scaleup.z * obj.K_phys * obj.scaledown.z );
            % obj.K_combined = ( obj.scaleup.res_err * obj.K_residual * obj.scaledown.res + (PCmtx*PCmtx') * obj.scaleup.z * obj.K_phys * obj.scaledown.z * (PCmtx*PCmtx') );  % added the PCmtx transform to the physics-based part
            % obj.K_combined = ( obj.K_residual + (PCmtx(:,1:P)*PCmtx(:,1:P)') * obj.scaleup.z * obj.K_phys * obj.scaledown.z * (PCmtx(:,1:P)*PCmtx(:,1:P)') );  % added the PCmtx transform to the physics-based part, and do scaling of K_residual before here
            obj.K_combined = obj.K_phys + lambda * obj.K_residual;

            % % DEBUG: replaced K_combined with a matrix that multiplies K_phys rather than adding to it
            % poop_trans = pinv(alpha * obj.K_phys') * beta_dataonly;
            % obj.K_combined = poop_trans' * obj.K_phys;
        end

        % OLD VERSION
        function obj = get_K_residual_old( obj , data )
            %get_K_residual Identify residual Koopman model from data
            %   Detailed explanation goes here
            
            % assemble snapshot pairs (ONLY WORKS FOR LINEAR MODELS)
            alpha = [];
            beta = [];
            beta_dataonly = [];
            for i = 1 : obj.num_trials
                
%                 % get x using approximation of inverse kinematics (this is slow)
%                 x = zeros( length(data{i}.t) , obj.kphys.params.n );
%                 for j = 1 : length( data{i}.t )
%                     x(j,:) = obj.kphys.sys.est_x_ik( data{i}.y(j,:)' )';  % train on output observations
%                 end
                % x = data{i}.Q_ik;    % this assumes you can measure joint angles directly (USE FOR ARMS)
%                 x = [ data{i}.y(:,end/2-2:end/2) , data{i}.Q_ik ]; % for xyz embedded models DEBUG****
                x = data{i}.x;  % FOR NON-ARM SYSTEMS

                data_basis = obj.kphys.lift.basis.x( x' )';     % lifting functions are defined in template model class
%                 data_fullbasis = obj.kphys.lift.fullbasis.x( x' , data{i}.u' )'; % lifting functions are defined in template model class
                data_fullbasis_a = obj.kphys.lift.fullbasis.x( x(1:end-1,:)' , data{i}.u(1:end-1,:)' )'; % lifting functions are defined in template model class
                data_fullbasis_b = obj.kphys.lift.fullbasis.x( x(2:end,:)' , data{i}.u(1:end-1,:)' )';

%                 alpha_i = data_fullbasis( 1 : end-1 , : );
                alpha_i = data_fullbasis_a;    % ( 1 : end-1 , : );

%                 beta_i = [ data_basis( 2 : end , : ) , data{i}.u(1:end-1,:) ] - data_fullbasis( 1 : end-1 , : ) * obj.scaledown.z * obj.K_phys' * obj.scaleup.z;
                beta_i = data_fullbasis_b - data_fullbasis_a * obj.scaledown.z * obj.K_phys' * obj.scaleup.z;
                
                beta_dataonly_i = data_fullbasis_b;  % for learning pure data-driven model

                alpha = [ alpha ; alpha_i ];
                beta = [ beta ; beta_i ];
                beta_dataonly = [ beta_dataonly ; beta_dataonly_i ];
            end

            % find scaling factor so no data is larger than 1
%             obj.scaleup.res = diag( max( abs( [ alpha ; beta ] ) ) );    % diagonal matrix of largest elemnt of each state
            obj.scaleup.res = diag( max( abs( alpha ) ) );    % diagonal matrix of largest elemnt of each state
            obj.scaleup.res_err = diag( max( abs( beta ) ) );
            % obj.scaleup.res = eye(obj.kphys.params.Nfull);  % DEBUG: remove scaling %%%%%%%%%%%%%%%%%%
            obj.scaledown.res =  obj.scaleup.res^(-1);
            obj.scaledown.res_err =  obj.scaleup.res_err^(-1);
            obj.scaledown.res(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs
            obj.scaledown.res_err(end-obj.kphys.params.m+1:end,end-obj.kphys.params.m+1:end) = zeros(obj.kphys.params.m); % get rid of Infs

            % Scale the data
            alpha_scaled = alpha * obj.scaledown.res;
            beta_scaled = beta * obj.scaledown.res;
%             beta_scaled = beta * obj.scaledown.res_err;
            beta_dataonly_scaled = beta_dataonly * obj.scaledown.res;

%             % Idendify the error Koopman matrix
%             koop_err_trans = pinv(alpha_scaled)*beta_scaled;
%             obj.K_residual = koop_err_trans';
% %             obj.K_residual = obj.K_residual * eye(size(obj.K_residual)) * eigs(obj.K_residual,1)^(-1);  % DEBUG: impose more stability
        
%             % Identify the error Koopman matrix using Lasso regression
%             N = obj.kphys.params.Nfull;
%             K = size( alpha_scaled , 1 );
%             A_data = zeros( N*K , N^2 );
%             for i = 1 : K
%                 A_data( N*(i-1)+1:N*i , : ) = kron( eye(N) , alpha_scaled(i,:) );
%             end
%             b_data = reshape( beta_scaled' , [numel(beta_scaled),1] );
%             Kres_trans_vec = lasso( A_data, b_data, 'Lambda', 5e-5, 'Intercept',false );
% %             Kres_trans_vec = lasso( A_data, b_data, 'Lambda', 1e-4, 'Intercept', false, 'Alpha', 0.5 );
%             Kres_trans = reshape( Kres_trans_vec, [ N , N ] );
%             obj.K_residual = Kres_trans';

            % Identify error Koopman using Lasso, one column at a time
            N = obj.kphys.params.Nfull;
            K = size( alpha_scaled , 1 );
            A_data = alpha_scaled;
            Kres_trans = zeros(N);
            for i = 1 : N
                b_data = beta_scaled(:,i);
                % Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false );   % No Lasso penalty, used for real arm system (allegedly)
                % Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2.95e-2, 'Intercept',false ); % best I've been able to get for real arm system, not sure how zero lasso worked before ****
                % Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2e-3, 'Intercept',false ); % this number works well 
                % Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0.39e-4, 'Intercept',false );   % 2e-2 for bilinear2
%                 Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2e-2, 'Intercept',false );   % 2e-2 for bilinear2
%                 Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 4e-3, 'Intercept',false );   % for psi 3 module model
                Kres_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false ); % this number works well for vanderpol model
            end
            obj.K_residual = Kres_trans';

       
            % DEBUG: Do PCA on data to reduce the number of basis functions
            diff_data = beta_dataonly_scaled - alpha_scaled;
            [PCmtx,score,latent,tsquared,explained,mu] = pca(diff_data);
            for P = 1 : size(PCmtx,2)   % take first P PCs that explain 99% of data
                if sum(explained(1:P)) > 99
                    break;
                end
            end

            % Identify data-driven only Koopman using Lasso, one column at a time
            N = obj.kphys.params.Nfull;
            A_data = alpha_scaled;
            Kdata_trans = zeros(P); % DEBUG: zeros(N);

            A_data = alpha_scaled * PCmtx(:,1:P);   % DEBUG
            beta_dataonly_scaled = beta_dataonly_scaled * PCmtx(:,1:P); % DEBUG

            for i = 1 : P   % DEBUG: was N before
                b_data = beta_dataonly_scaled(:,i);
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false ); % No Lasso penalty
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 2e-3, 'Intercept',false ); % this number works well
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 1e-4, 'Intercept',false );
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0.5e-4, 'Intercept',false ); % used for real arm system ****
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 1e-3, 'Intercept',false ); % this number works well for vanderpol model with limited data
                Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 0, 'Intercept',false ); % this number works well for vanderpol model with 10x data
                % Kdata_trans(:,i) = lasso( A_data, b_data, 'Lambda', 5e-4, 'Intercept',false ); % this number works well for duffing model
            end
            obj.K_data = Kdata_trans';
            obj.K_data = PCmtx(:,1:P) * obj.K_data * PCmtx(:,1:P)'; % DEBUG: make it NxN again (temporary)

            % % DEBUG: scale down K_data by its largest eigenvalue
            % max_eig_kdata = max( abs( eig(obj.K_data) ) );
            % obj.K_data = obj.K_data ./ (max_eig_kdata + 1e-4);

%             % Only care about correcting value of x (not lifting functions)
%             N = obj.kphys.params.Nfull;
%             beta_scaled = beta_scaled(:,1:obj.kphys.params.n);  % just first n components
%             Kres_trans_rect = pinv(alpha_scaled) * ( beta_scaled );
%             Kres_rect = Kres_trans_rect';
%             obj.K_residual = [ Kres_rect ; zeros(N-obj.kphys.params.n , N) ];

            % model that combines the physics-based and residual models
            obj.K_combined = ( obj.scaleup.res * obj.K_residual * obj.scaledown.res + obj.scaleup.z * obj.K_phys * obj.scaledown.z );
        
            % % DEBUG: scale down K_combined by its largest eigenvalue
            % max_eig_kcombined = max( abs( eig(obj.K_combined) ) );
            % obj.K_combined = obj.K_combined ./ (max_eig_kcombined);
        end

        function obj = get_lti_models( obj )
            % get lti model matrices from the koopman matrix models

            % define LTI combined model
            obj.model.combined.A = obj.K_combined( 1 : end - obj.kphys.params.m , 1 : end - obj.kphys.params.m );
            obj.model.combined.B = obj.K_combined( 1 : end - obj.kphys.params.m , end - obj.kphys.params.m + 1 : end );

            % define LTI physics-based model
            K_phys_unscaled = obj.scaleup.z * obj.K_phys * obj.scaledown.z;
            obj.model.physics_based.A = K_phys_unscaled( 1 : end - obj.kphys.params.m , 1 : end - obj.kphys.params.m );
            obj.model.physics_based.B = K_phys_unscaled( 1 : end - obj.kphys.params.m , end - obj.kphys.params.m + 1 : end );

            % define LTI data-driven model
            K_data_unscaled = obj.scaleup.res * obj.K_data * obj.scaledown.res;
            obj.model.data_driven.A = K_data_unscaled( 1 : end - obj.kphys.params.m , 1 : end - obj.kphys.params.m );
            obj.model.data_driven.B = K_data_unscaled( 1 : end - obj.kphys.params.m , end - obj.kphys.params.m + 1 : end );
        end
        
        function [ S_out , i ] = cull_basis( obj, diff_data , epsilon )
            % cull_basis: Outputs a selection matrix that eliminates basis
            % functions above a certain index. The cutoff is computed by
            % setting an error threshold for the reduced set of basis
            % functions relative to points lifted into the space spanned by
            % the complete set of basis functions.
            
            m = obj.kphys.params.m; % number of inputs to the system

            N = size(diff_data,2);  % dimension of lifted states
            num_pts = size(diff_data,1);  % number of data points
            for i = 1 : N-m
                S = blkdiag( eye(i) , zeros(N-i-m,N-i-m) , eye(m) );
                projection = diff_data * S;
                projection_error = vecnorm( (diff_data - projection)' )';
                % score = sum( projection_error ) / num_pts;
                % score = mean( projection_error ./ vecnorm(diff_data')' );   % different way of scoring
                score = var(projection_error) / sum(var(projection_error)) * 100;   % score using variance, similar to PCA "explained"
                if i==9 % score < epsilon % for debugging, set i == 9
                    break;
                end
            end
            S_out = [ S(1:i , :) ; S(end-m+1:end,:) ]';
            i = i + m;
        end

        function lambda = get_residual_weight( obj, Kp, Kr, alpha, beta )
            % Computes the optimal weighting function for the residual
            % Koopman matrix Kr given a set of snapshots [alpha, beta]
            
            % lambda_vec = diag( ( ( Kr*alpha' )' * ( beta' - Kp*alpha' ) ) ./ ( ( Kr*alpha' )'*( Kr*alpha' ) ) );
            % lambda = mean(lambda_vec);
            
            lambda = sum( diag( ( Kr*alpha' )' * ( beta' - Kp*alpha' ) ) ) ./ sum( diag( ( Kr*alpha' )'*( Kr*alpha' ) ) ) ;
            
            % bound value to be between 0 and 1
            lambda = max(0,lambda);
            lambda = min(1,lambda);
        end

        %% Simulation
        
        function [ data_hyb , data_phys , data_data ] = compare_models( obj , real_data_in , ploton )
            % compare_models For given data from the real system, simulate
            % the physics-based Koopman model and the "hybrid" koopman
            % model, and plot the results (optional)

            if nargin < 3
                ploton = 0; % default is not to plot
            end

            t_in = real_data_in.t;
            u_in = real_data_in.u;
            x0 = real_data_in.x(1,:)';
            num_state = length(x0);%/2;

            data_phys = obj.simulate_phys( t_in , u_in , x0 );
            data_data = obj.simulate_datadriven( t_in , u_in , x0 );
            data_hyb = obj.simulate_hybrid( t_in , u_in , x0 );

            if ploton
                figure; % plot just the first state (for now)
                for i = 1 : num_state
                    subplot(num_state,1,i);
                    hold on;
                    plot( t_in , real_data_in.x(:,i) ); % real system
                    plot( data_phys.t , data_phys.x(:,i) ); % physics-based Koopman model
                    plot( data_data.t , data_data.x(:,i) ); % data-driven Koopman model
                    plot( data_hyb.t , data_hyb.x(:,i) ); % "hybrid" Koopman model
                    if isfield( real_data_in , 'x_clean' )
                        plot( t_in , real_data_in.x_clean(:,i) ); % DEBUG: real system without measurement noise
                    end
                    hold off;
                    grid on; box on;
                    ylabel(['y',num2str(i)]);
                    xlabel('t (seconds)');
                    % ylim([-0.50,0.50]); % NOTE: Set heuristically based on Arm system bounds
                    % ylim([-1.5,1.5]); % NOTE: Set heuristically based on simple nonlinear system bounds
                end
                legend({'Real','Physics-based Koopman', 'Data-driven Koopman', 'Combined Koopman'});
                %             legend({'Real','Data-driven'});
            end
        end
        
        function [ data_hyb_out , data_phys_out , data_data_out ] = compare_models_output( obj , real_data_in , data_phys, data_data, data_hyb, Model_obj, which_output_idx)
            % compare_models_output For given data from the real system, simulate
            %  the physics-based Koopman model and the "hybrid" koopman
            %  model, and plot the results
            %
            %  Shows comparison in terms of model output rather than state.
            %  Model_obj is an instance of the class that contains
            %  functions to compute the output from the state.
            %
            %  which_output_idx specifies the indices of the specific
            %  outputs you want to be plotted

            t_in = real_data_in.t;
            u_in = real_data_in.u;
            x0 = real_data_in.x(1,:)';
            num_state = length(x0)/2;

            data_phys_out = data_phys;
            data_data_out = data_data;
            data_hyb_out = data_hyb;

            % compute output
            for i = 1 : length( real_data_in.t )
                data_phys_out.y(i,:) = Model_obj.get_y( data_phys.x(i,:) );
                data_data_out.y(i,:) = Model_obj.get_y( data_data.x(i,:) );
                data_hyb_out.y(i,:) = Model_obj.get_y( data_hyb.x(i,:) );
            end

            figure; % plot just the outputs specified by user
            for i = 1 : num_state
                subplot(num_state,1,i);
                hold on;
                plot( t_in , real_data_in.y(:,which_output_idx(i)) ); % real system
                plot( data_phys_out.t , data_phys_out.y(:,which_output_idx(i)) ); % physics-based Koopman model
                plot( data_data_out.t , data_data_out.y(:,which_output_idx(i)) ); % data-driven Koopman model
                plot( data_hyb_out.t , data_hyb_out.y(:,which_output_idx(i)) ); % "hybrid" Koopman model
                hold off;
                grid on; box on;
                ylabel(['y',num2str(i)]);
                xlabel('t (seconds)');
%                 ylim([-0.25,0.25]); % NOTE: Set heuristically based on Arm system bounds
            end
            legend({'Real','Physics-based Koopman', 'Data-driven Koopman', 'Combined Koopman'});
%             legend({'Real','Data-driven'});
        end

        function data = simulate_hybrid( obj , t_in , u_in , x0 )
            %simulate_physics Simulate using the "hybrid" koopman model
            %that combines the physics-based model with the data-driven
            %residual model.
            
            % resample if input data doesn't have same timestep as the class
            % t = ( t_in(1) : obj.timestep : t_in(end) )';
            t = t_in; % NOTE: to remove discrpency between real and simulated points
            u = interp1( t_in , u_in , t );

%             y0 = obj.kphys.sys.get_y( x0' )';
%             x0_ik = obj.kphys.sys.est_x_ik( y0 ); % estimate x from y using simplified dynamics (i/o are column vecs)
%             z0 = obj.kphys.lift.basis.x( x0_ik );

            z0 = obj.kphys.lift.basis.x( x0 );

            data.t = t;
            data.u = u;
            data.z = zeros( length(t) , obj.kphys.params.N );
            data.z(1,:) = z0';
%             dt = mean( t(2:end) - t(1:end-1) ); % time step
            iso_z = [ eye(obj.kphys.params.N) , zeros(obj.kphys.params.N , obj.kphys.params.Nfull - obj.kphys.params.N) ];
            for i = 2 : length( t )
                xi = data.z(i-1,1:obj.kphys.params.n)';
%                 data.z(i,:) = ( iso_z * (obj.K_residual + obj.K_phys) * obj.kphys.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % lift at each timestep
%                 data.z(i,:) = ( iso_z * obj.scaleup.z * (obj.K_residual + obj.K_phys) * obj.scaledown.z * obj.kphys.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % lift at each timestep
                % data.z(i,:) = ( iso_z * ( obj.scaleup.res * obj.K_residual * obj.scaledown.res + obj.scaleup.z * obj.K_phys * obj.scaledown.z ) * obj.kphys.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % different scaling for phys and residual models ****
%                 data.z(i,:) = ( iso_z * ( obj.scaleup.res * obj.K_residual * obj.scaledown.res ) * obj.kphys.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % data-driven model only
                % data.z(i,:) = ( iso_z * ( obj.K_combined ) * [ data.z(i-1,:)' ; u(i-1,:)'] )';  % DEBUG: Don't lift at each timestep
                data.z(i,:) = ( iso_z * ( obj.K_combined ) * obj.kphys.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % don't do any scaling here, just use K_combined mtx
                % data.z(i,:) = ( iso_z * obj.PCmtx * ( obj.K_combined ) * obj.PCmtx' * obj.kphys.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % flow it in the space spanned by principle components
            end
            data.x = data.z( : , 1:obj.kphys.params.n );
        end

        function data = simulate_datadriven( obj , t_in , u_in , x0 )
            %simulate_physics Simulate using the "hybrid" koopman model
            %that combines the physics-based model with the data-driven
            %residual model.
            
            % resample if input data doesn't have same timestep as the class
            % t = ( t_in(1) : obj.timestep : t_in(end) )';
            t = t_in; % NOTE: to remove discrpency between real and simulated points
            u = interp1( t_in , u_in , t );

            z0 = obj.kphys.lift.basis.x( x0 );

            data.t = t;
            data.u = u;
            data.z = zeros( length(t) , obj.kphys.params.N );
            data.z(1,:) = z0';
%             dt = mean( t(2:end) - t(1:end-1) ); % time step
            iso_z = [ eye(obj.kphys.params.N) , zeros(obj.kphys.params.N , obj.kphys.params.Nfull - obj.kphys.params.N) ];
            for i = 2 : length( t )
                xi = data.z(i-1,1:obj.kphys.params.n)';
                % data.z(i,:) = ( iso_z * ( obj.scaleup.res * obj.K_data * obj.scaledown.res ) * [ data.z(i-1,:)' ; u(i-1,:)'] )';  % DEBUG: Don't lift at each timestep
                data.z(i,:) = ( iso_z * ( obj.scaleup.res * obj.K_data * obj.scaledown.res ) * obj.kphys.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % data-driven model only
                % data.z(i,:) = ( iso_z * obj.PCmtx * ( obj.scaleup.res * obj.K_data * obj.scaledown.res ) * obj.PCmtx' * obj.kphys.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % flow in the space spanned by principle components
            end
            data.x = data.z( : , 1:obj.kphys.params.n );
        end


        function data = simulate_phys( obj , t_in , u_in , x0 )
            %simulate_physics Simulate using the physics-based koopman model
            
            % % Use simulation function from the Klift class
            % data = obj.kphys.sim_koop_y2x(t_in,u_in,x0);
            
            % resample if input data doesn't have same timestep as the class
            % t = ( t_in(1) : obj.timestep : t_in(end) )';
            t = t_in; % NOTE: to remove discrpency between real and simulated points
            u = interp1( t_in , u_in , t );

            z0 = obj.kphys.lift.basis.x( x0 );

            data.t = t;
            data.u = u;
            data.z = zeros( length(t) , obj.kphys.params.N );
            data.z(1,:) = z0';
%             dt = mean( t(2:end) - t(1:end-1) ); % time step
            iso_z = [ eye(obj.kphys.params.N) , zeros(obj.kphys.params.N , obj.kphys.params.Nfull - obj.kphys.params.N) ];
            for i = 2 : length( t )
                xi = data.z(i-1,1:obj.kphys.params.n)';
                % data.z(i,:) = ( iso_z * obj.K_phys * [ data.z(i-1,:)' ; u(i-1,:)'] )';  % DEBUG: Don't lift at each timestep
                data.z(i,:) = ( iso_z * obj.K_phys * obj.kphys.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % physics-based model only
            end
            data.x = data.z( : , 1:obj.kphys.params.n );
        end


    end
end

























