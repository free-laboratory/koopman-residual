classdef Klift
    %Klift Creates a lifted Koopman representation of a dynamical system
    %   Detailed explanation goes here

    properties
        params struct;
        sys struct;
        integral_res;   % resultion of numerically computed integrals
%         scale_factor;   % scaled data to be leq 1
        
        basis;
        fullbasis;
        basisdot;
        fullbasisdot;
        Kmtx;
        K_dis;
        scaleup;    % scaling up matrix for .x and .u
        scaledown;  % scaling doen matrix for .x and .u
        K_prods;    % running total of sum of basis functions products at sample points
        K_norms;    % running totle of sum of function norms at sample points
        grid_lines; % samples along each dimension of x and u

        % functions
        get_xdot;
        lift;
        get_massmtx;
        get_rhs;

        % properties that can be set using Name,Value pair input to constructor
        isupdate;   % true if this should overwrite an existing model when saving, false otherwise
        basis_type;   % cell array of the types of observables to include in model
        basis_degree; % array of the degree/complexity of each type of observable
        model_type; % 'linear' or 'bilinear'
        has_massmtx; % boolean
        num_samples; % number of sample points to take over state/inputs space for building Koopman matrix
        has_output; % boolean. True of the model has an output that is different than the state
        timestep;
        integration_type;   % 'analytic' or 'montecarlo'
    end

    methods
        function obj = Klift( sys_in , varargin )
            %Klift Construct an instance of this class
            %   Takes a sys struct as input, consising of fields:
            %       sys.x - symbolic state vector
            %       sys.u - symbolic input vector
            %       sys.xdot - symbolic expression for xdot
            
            obj.sys = sys_in;

            % system parameters
            obj.params = struct;
            obj.params.n = length( obj.sys.x );
            obj.params.m = length( obj.sys.u );
            obj.params.ny = length( obj.sys.y );

            % set default values of parameters
%             obj.scaleup.x = diag( sys.x_domain(:,2) );    % mult/divide by ub (assumes lb = -ub)
            obj.scaleup.u = diag( obj.sys.u_domain(:,2) );    % mult/divide by ub (assumes lb = -ub)
%             obj.scaledown.x = obj.scaleup.x^(-1);    
            obj.scaledown.u = obj.scaleup.u^(-1);
            obj.integral_res = 10;     % increase for better accuracy
            obj.basis_degree = 1;
            obj.basis_type = 'hermite';
            obj.model_type = 'linear';
            obj.has_massmtx = false;
            obj.num_samples = 1e3;
            obj.timestep = 1e-3;    % default timestep for discrete model
%             obj.sys.x_pos = obj.sys.x( 1 : end/2 );
%             obj.sys.Jy = jacobian( obj.sys.output( 1 : end/2 ) , obj.sys.x_pos );    % dy_pos/dx_pos
%             obj.sys.get_Jy = matlabFunction( obj.sys.Jy , 'Vars' , {obj.sys.x_pos} );
            % ---------------------------------
            % -------------TBD-----------------
            % ---------------------------------
            obj.integration_type = 'montecarlo';    % default method for computing integrals

            % replace default values with user input values
            obj = obj.parse_args( varargin{:} );

            % define function for evaluating system dynamics
            syms t; % dummy variable so it can work with ode45
            if obj.has_massmtx
                obj.get_massmtx = matlabFunction( obj.sys.massmtx , 'Vars' , {t , obj.sys.x} );
                obj.get_rhs = matlabFunction( obj.sys.rhs , 'Vars' , {t , obj.sys.x , obj.sys.u} );
            else
                obj.get_xdot = matlabFunction( obj.sys.xdot , 'Vars' , {t , obj.sys.x , obj.sys.u} );
            end

            % define basis functions
            obj = obj.def_basis;

            % identify Koopman operator over basis functions
            if obj.has_massmtx
                obj = obj.get_koopman_mc;   % computes via approx projection using Monte Carlo integration
            else
                if strcmp( obj.integration_type, 'analytic')
                    obj = obj.get_koopman;      % computes via analytic projection
                elseif strcmp( obj.integration_type, 'montecarlo')
                    obj = obj.get_koopman_mc;   % computes via approx projection using Monte Carlo integration
                else
                    error('Please choose a valide type of integration.');
                end
            end
        end

        function obj = parse_args( obj , varargin )
            %parse_args Parses the Name, Value pairs in varargin of the
            % constructor, and assigns property values
            for idx = 1:2:length(varargin)
                obj.(varargin{idx}) = varargin{idx+1} ;
            end
        end

        function vars = parse_args_local( obj , vars , varargin )
            %parse_args Parses the Name, Value pairs in varargin of a function,
            % and assigns variable values accordingly
            for idx = 1:2:length(varargin)
                vars.(varargin{idx}) = varargin{idx+1} ;
            end
        end

        %% Basis functions
        
        function [ obj , basis , basisdot ] = def_basis( obj )
            %def_basis Defines set of basis functions
            switch obj.basis_type
                case 'hermite'
%                     [ obj , basis , basisdot ] = obj.def_basis_hermite( obj.basis_degree );
                    [ obj , basis , basisdot ] = obj.def_basis_hermite_x( obj.basis_degree );   % Lifting state instead of output (again)
                otherwise
                    error('Please choose a valid basis type');
            end
%             obj.params.N = length(basis.y);
            obj.params.N = length(basis.x); % Lifting state instead of output (again)

            % Add control input to basis functions
            switch obj.model_type
                case 'linear'
                    obj.fullbasis.x = [ basis.x ; obj.sys.u ];
                    obj.fullbasis.y = [ basis.y ; obj.sys.u ];
                    obj.fullbasisdot.x = [ basisdot.x ; zeros( obj.params.m , 1 ) ]; % u assumed constant
                    obj.fullbasisdot.y = [ basisdot.y ; zeros( obj.params.m , 1 ) ]; % u assumed constant
                case 'bilinear'
                    obj.fullbasis.x = basis.x;
                    obj.fullbasis.y = basis.y;
                    obj.fullbasisdot.x = basisdot.x;
                    obj.fullbasisdot.y = basisdot.y;
                    for i = 1 : obj.params.m
                        obj.fullbasis.x = [ obj.fullbasis.x ; basis.x * obj.sys.u(i) ];
                        obj.fullbasis.y = [ obj.fullbasis.y ; basis.y * obj.sys.u(i) ];
                        obj.fullbasisdot.x = [ obj.fullbasisdot.x ; basisdot.x * obj.sys.u(i) ];
                        obj.fullbasisdot.y = [ obj.fullbasisdot.y ; basisdot.y * obj.sys.u(i) ];
                    end
                    obj.fullbasis.x = [ obj.fullbasis.x ; obj.sys.u ];
                    obj.fullbasis.y = [ obj.fullbasis.y ; obj.sys.u ];
                    obj.fullbasisdot.x = [ obj.fullbasisdot.x ; zeros(size(obj.sys.u)) ]; % u assumed constant
                    obj.fullbasisdot.y = [ obj.fullbasisdot.y ; zeros(size(obj.sys.u)) ]; % u assumed constant
                otherwise
                    error('Please choose a valide model type: linear or bilinear');
            end

%             obj.params.Nfull = length(obj.fullbasis.y);
            obj.params.Nfull = length(obj.fullbasis.x);     % Lifting state instead of output (again)

            % Create lifting functions
            obj.lift.basis.x = matlabFunction(basis.x , 'Vars', {obj.sys.x} );
            obj.lift.basis.y = matlabFunction(basis.y , 'Vars', {obj.sys.y} );
            obj.lift.fullbasis.x = matlabFunction(obj.fullbasis.x , 'Vars', {obj.sys.x , obj.sys.u} );
            obj.lift.fullbasis.y = matlabFunction(obj.fullbasis.y , 'Vars', {obj.sys.y , obj.sys.u} );
            if obj.has_massmtx
                obj.lift.basisdot.x = matlabFunction(basisdot.x , 'Vars', {obj.sys.x , obj.sys.u , obj.sys.dx} );
                obj.lift.basisdot.y = matlabFunction(basisdot.y , 'Vars', {obj.sys.y , obj.sys.u , obj.sys.dy} );
                obj.lift.fullbasisdot.x = matlabFunction(obj.fullbasisdot.x , 'Vars', {obj.sys.x , obj.sys.u , obj.sys.dx} );
                obj.lift.fullbasisdot.y = matlabFunction(obj.fullbasisdot.y , 'Vars', {obj.sys.y , obj.sys.u , obj.sys.dy} );
            else
                obj.lift.basisdot.x = matlabFunction(basisdot.x , 'Vars', {obj.sys.x , obj.sys.u} );
                obj.lift.basisdot.y = matlabFunction(basisdot.y , 'Vars', {obj.sys.y , obj.sys.u} );
                obj.lift.fullbasisdot.x = matlabFunction(obj.fullbasisdot.x , 'Vars', {obj.sys.x , obj.sys.u} );
                obj.lift.fullbasisdot.y = matlabFunction(obj.fullbasisdot.y , 'Vars', {obj.sys.y , obj.sys.u} );
            end
        end

        function [ obj , basis , basisdot ] = def_basis_hermite( obj , degree )
            %def_basis_hermite Defines hermite polynomial basis
            
            zeta = obj.sys.y;   % basis functions lift the output, not state    
            dzeta = obj.sys.dy;
            nzeta = length(zeta);
            maxDegree = degree;

            % Number of mononials, i.e. dimenstion of p(x)
            N = factorial(nzeta + maxDegree) / ( factorial(nzeta) * factorial(maxDegree) );

            % matrix of exponents (N x naug). Each row gives exponents for 1 monomial
            exponents = [];
            for i = 1:maxDegree
                exponents = [exponents; partitions(i, ones(1,nzeta))];
            end

            % create vector of orderd monomials (column vector)
            for i = 1:N-1
                hermiteBasis(i,1) = obj.get_hermite(zeta, exponents(i,:));
            end

%             % TRYOUT: add constant to the end of the basis functions
%             % UPDATE: doesn't seem to help with DC error
%             hermiteBasis = [ hermiteBasis ; 2 ];

            % output variables
            basis.y = hermiteBasis / 2;    % symbolic vector of basis monomials, expressed in terms of output, y
            basis.x = subs( basis.y , obj.sys.y , obj.sys.output ); % basis expressed in terms of state, x
            basisdot = obj.def_basisdot( basis );

            % I THINK THIS IS REDUNDANT, ALREADY DONE IN OUTER FUNCTION
%             % create the lifting function: zeta -> p(zeta)
%             obj.lift.basis = matlabFunction(basis , 'Vars', {zeta} );
%             if obj.has_massmtx
%                 obj.lift.basisdot = matlabFunction(basisdot , 'Vars', {zeta , obj.sys.u , dzeta} );
%             else
%                 obj.lift.basisdot = matlabFunction(basisdot , 'Vars', {zeta , obj.sys.u} );
%             end

            % save basis to class
            obj.basis.x = basis.x;
            obj.basisdot.x = basisdot.x;
            obj.basis.y = basis.y;
            obj.basisdot.y = basisdot.y;
        end

        function hermite = get_hermite( obj , x , orders )
            %get_monomial: builds a monomial from symbolic vector x and a vector of
            %exponents. Auxiliary function to def_basis_hermite
            %   e.g. x = [x1 x2]; exponents = [1 2]; =>  monomial = hermiteH(1,x1) * hermiteH(2,x2)
            
            n = length(x);

            hermite = hermiteH( orders(1) , x(1) );
            for i = 2:n
                hermite = hermite * hermiteH( orders(i) , x(i) );
            end
        end

        function [ basisdot ] = def_basisdot( obj , basis )
            %get_basisdot: Define the symbolic derivative of basis
            % functions in terms of the output (y) and state (x)
            %
            % If has_output == false, y = x

            x = obj.sys.x;
            y = obj.sys.y;
%             output = obj.sys.output;
%             dydt = jacobian( output , obj.sys.x ) * obj.sys.dx;
            dx = obj.sys.dx;
            dy = obj.sys.dy;

            dPsidy = jacobian( basis.y , y );
            dPsidx = jacobian( basis.x , x );

            basisdot.y = dPsidy * dy;
            basisdot.x = dPsidx * dx;
        end

        %% Koopman matrix identification
        
        function [ obj , K ] = get_koopman( obj )
            %get_koopman Identify the Koopman operator projected onto basis
            
            % local shorthand
            n = obj.params.n;
            m = obj.params.m;
            N = obj.params.N;
            basis = obj.fullbasis.x;
            basisdot = obj.fullbasisdot.x;
            x = obj.sys.x;
            u = obj.sys.u;
            dx = obj.sys.dx;
            xdot = obj.sys.rhs;
            basisdot = subs( basisdot, dx, xdot );
            x_domain = obj.sys.x_domain .* ones(n,1);
            u_domain = obj.sys.u_domain .* ones(m,1);

            % compute each entry in koopman matrix via projection
            K = zeros(N+m,N+m); % analytic koopman matrix
            tic
            basis_norm = zeros( N+m , 1 );
            for i = 1 : N+m    % compute the norm of each basis function
                norm_i = basis(i)^2 * exp( -sum(x.^2) ); % initialized value
                for k = 1 : n
                    % norm_i = int( norm_i , x(k) , x_domain(k,1) , x_domain(k,2) );
                    norm_i = int( norm_i , x(k) , -Inf , Inf );
                end
                for k = 1 : m
                    norm_i = int( norm_i , u(k) , u_domain(k,1) , u_domain(k,2) );
                    % norm_i = int( norm_i , u(k) , -Inf , Inf );
                end
                basis_norm(i) = norm_i;
            end
            for i = 1 : N+m    % compute the inner product of each basisdot elem. with each basis elem.
                
                for j = 1 : N+m
                    innerprod = basisdot(j) * basis(i) * exp( -sum(x.^2) ); % initialized value
                    for k = 1 : n
                        innerprod = int( innerprod , x(k) , x_domain(k,1) , x_domain(k,2) );
                        if innerprod == 0   % if term is zero don't waste time on useless integration
                            break;
                        end                        
                    end
                    for k = 1 : m
                        if innerprod == 0   % if term is zero don't waste time on useless integration
                            break;
                        end
                        innerprod = int( innerprod , u(k) , u_domain(k,1) , u_domain(k,2) );
                    end

                    K(j,i) = double( innerprod ) / basis_norm(i);
                end
            end
            toc
            
            obj.Kmtx = K;  % store as class object

            % for backwards compatibility
            obj.scaleup.z = eye(obj.params.Nfull);
            obj.scaledown.z =  obj.scaleup.z^(-1);
        end

        % get_koopman_mc: Identify elements of Koopman matrix using Monte Carlo integration
        function [ obj , K ] = get_koopman_mc( obj )
            %get_koopman: Identify the Koopman operator projected onto basis
            %   using Monte Carlo integration
            
            % local shorthand
            n = obj.params.n;
            m = obj.params.m;
            N = obj.params.N;
            basis = obj.fullbasis.x;
            basisdot = obj.fullbasisdot.x;
            x = obj.sys.x;
            u = obj.sys.u;
            dx = obj.sys.dx;
            xdot = obj.sys.rhs;
            basisdot = subs( basisdot, dx, xdot );
            x_domain = obj.sys.x_domain .* ones(n,1);
            u_domain = obj.sys.u_domain .* ones(m,1);
            integral_domain = [x_domain; u_domain]; % since we integrate over state and input space
            weight_func = @(x) exp( sum( -x.^2 ) ); % hermite polynomial ip weight function

            % break of total sample points into sets
            if obj.num_samples <= 1e6
                num_sets = 1;
                pts_per_set = obj.num_samples;
            else
                num_sets = ceil( obj.num_samples / 1e6 );
                pts_per_set = zeros(num_sets,1);
                pts_per_set(1:end-1) = 1e6;
                if rem( obj.num_samples , 1e6 ) == 0
                    pts_per_set(end) = 1e6;
                else
                    pts_per_set(end) = rem( obj.num_samples , 1e6 );
                end
            end

            % state and input space bounds
            ub = integral_domain(:,2);
            lb = integral_domain(:,1);

            % initializations
            norm_integrand_sum = zeros(1,N+m);  % running sum of norm integrand
            ip_integrand_sum = zeros(N+m,N+m);  % running sum of ip integrand
            K = zeros(N+m,N+m); % analytic koopman matrix

            for ii = 1 : num_sets
                tic
                sample_pts = ((ub - lb) .* rand( n+m, pts_per_set(ii) ) + lb)';
                basis_points = zeros( size(sample_pts,1) , obj.params.Nfull );
                basisdot_points = zeros( size(sample_pts,1) , obj.params.Nfull );
                weight_points = zeros( size(sample_pts,1) , 1 );
                
                for jj = 1 : pts_per_set(ii)
                    % evaluate basis and basisdot at all sample points
                    x_k = sample_pts(jj,1:n)';
                    u_k = sample_pts(jj,n+1:end)';
                    xdot_k = pinv( obj.get_massmtx( 0 , x_k ) ) * obj.get_rhs( 0 , x_k , u_k );
                    % xdot_k = obj.get_rhs( 0 , x_k , u_k );

                    basis_points(jj,:) = obj.lift.fullbasis.x( x_k , u_k )';
                    basisdot_points(jj,:) = obj.lift.fullbasisdot.x( x_k , u_k , xdot_k )';
                    % weight_points(jj,:) = weight_func( [x_k;u_k] ); % DEBUG: Should u be included? hermite polynomial ip weight function
                    weight_points(jj,:) = weight_func( x_k ); % hermite polynomial ip weight function
                end
                
                % update running sums of integrands evaluated at sample points
                for i = 1 : N+m    % compute the inner product of each basisdot elem. with each basis elem.
                    norm_integrand_pts = basis_points(:,i).^2 .* weight_points;
                    norm_integrand_sum(i) = norm_integrand_sum(i) + sum( norm_integrand_pts ) / obj.num_samples;
                    for j = 1 : N+m
                        ip_integrand_pts = basisdot_points(:,j) .* basis_points(:,i) .* weight_points;
                        ip_integrand_sum(j,i) = ip_integrand_sum(j,i) + sum( ip_integrand_pts ) / obj.num_samples;
                    end
                end
                toc
            end

            % assign values to elements of Koopman matrix by dividing integrand sums
            for i = 1 : N+m
                for j = 1 : N+m
                    K(j,i) = ip_integrand_sum(j,i) / norm_integrand_sum(i);
                end
            end

            obj.Kmtx = K;  % store as class object

            % for backwards compatibility
            obj.scaleup.z = eye(obj.params.Nfull);
            obj.scaledown.z =  obj.scaleup.z^(-1);
        end

        function result = num_sym_int( obj , fun , var , lb , ub , res )
            %num_sym_int Numerical integration on a symbolic expression
            %   Uses fixed-step to approximate integral over one (of
            %   possibly several) symbolic variables
            %       fun - symbolic expression
            %       var - symbolic variable to integrate over
            %       lb - lower integration bound
            %       ub - upper integration bound
            %       res - resolution (# of rectangles)
            digits(2);  % set precision used by vpa

            xstep = linspace(lb,ub,res)';
            dx = (ub-lb) / res;
            height = subs( fun , var , xstep );

            dx_vec = dx * ones( size(xstep) );
%             result = dx_vec' * vpa( [ height(1)/2 ; height(2:end-1) ; height(end)/2 ] );    % trapezoid rule
            result = dx_vec' * vpa( height );    % rectangle rule
        end
                      
        function sample_points = get_sample_points( obj , varargin )
            %get_sample_points: Sample over state/input space
            %   Name/value pairs
            %       'SampleType' - 'random' (default) or 'grid'
            %       'NumPts' - any natural number (only affects random sampling)
            %       'GridResolution' - any natural number (only affects grid sampling)
            %       'GridScale' - Number between 0 and 1 (default). Scales all the grid points

            % set property values
            properties.SampleType = 'random';   % default
            properties.NumPts = obj.num_samples;
            properties.GridResolution = obj.integral_res;
            properties.GridScale = 1;
            properties = obj.parse_args_local( properties, varargin{:} ); % override defaults

            x_domain = obj.sys.x_domain;
            u_domain = obj.sys.u_domain;

            if strcmp( properties.SampleType, 'random' )
                % random sample over state and input space
                num_pts = properties.NumPts;   % number of points to sample in state/input space
                sample_points = [ (x_domain(1,2)-x_domain(1,1)) * rand(num_pts,obj.params.n) + x_domain(1,1) , (u_domain(1,2)-u_domain(1,1)) * rand(num_pts,obj.params.m) + u_domain(1,1) ];    % one input for all joints
            elseif strcmp( properties.SampleType, 'grid')
                % discretize over state and input space
                grid_lines = cell(obj.params.n+obj.params.m,1);
                grid_mesh = cell(numel(grid_lines),1);
                for i = 1 : obj.params.n
                    grid_lines{i} = linspace( x_domain(i,1) , x_domain(i,2) , properties.GridResolution );
                end
                for i = 1 : obj.params.m
                    grid_lines{i+obj.params.n} = linspace( u_domain(i,1) , u_domain(i,2) , properties.GridResolution );
                end
                [grid_mesh{1:numel(grid_lines)}] = ndgrid(grid_lines{:}); %use comma-separated list expansion on both sides

                % list out all points in grid mesh in one data matrix
                sample_points = zeros( numel(grid_mesh{1}) , obj.params.n + obj.params.m );
                for i = 1 : obj.params.n + obj.params.m
                    sample_points(:,i) = grid_mesh{i}(:) * properties.GridScale;
                end
            end
        end

        function [ lifted_points, lifted_points_dot ] = lift_points( obj , sample_points )
            %lift_points: Evaluates basis and basisdot on a set of sample points

            % evaluate basisdot at all grid points
            lifted_points = zeros( size(sample_points,1) , obj.params.Nfull );
            lifted_points_dot = zeros( size(sample_points,1) , obj.params.Nfull );
%             lifted_points_plus = zeros( size(grid_points,1) , obj.params.Nfull );
            parfor i = 1 : size(sample_points,1)
                x_k = sample_points(i,1:obj.params.n)';
                u_k = sample_points(i,obj.params.n+1:end)';

                xdot_k = pinv( obj.get_massmtx( 0 , x_k ) ) * obj.get_rhs( 0 , x_k , u_k ); % note that there is a pinv here...
                x_kp1 = x_k + xdot_k * obj.timestep;

                lifted_points(i,:) = obj.lift.fullbasis.x( x_k , u_k )';
                lifted_points_dot(i,:) = obj.lift.fullbasisdot.x( x_k , u_k , xdot_k )';
%                 lifted_points_plus(i,:) = obj.lift.fullbasis.x( x_kp1 , u_k )';
            end
        end

        function error = estimate_error( obj , varargin )
            %estimate_error: Quantifies model error by comparing real
            %basisdot with those computed using Kmtx
            %   Name/Value pair arguments:
            %       'NumPts' - any natural number (defaul: 1e3)

            properties.NumPts = 1e3;
            properties = obj.parse_args_local( properties, varargin{:} ); % override defaults

            sample_points = obj.get_sample_points('NumPts', properties.NumPts, 'SampleType', 'random');
            [ lifted_points, lifted_points_dot ] = obj.lift_points( sample_points );

            % compute basisdot at each sample using Koopman operator (assumes no scaling)
            estimate_lifted_points_dot = lifted_points * obj.Kmtx';

            % compute error over all samples
            error = sum( vecnorm(lifted_points_dot(:,1:obj.params.n) - estimate_lifted_points_dot(:,1:obj.params.n), 2, 2) ) / length(lifted_points_dot);
        end

        % Monte Carlo Integrations (samples and lifts points)
        function result = intN_mc( obj , integrand , var , lb , ub , nsamples )
            % Uses monte carlo integration to approximate an integral
            %   integrand - a symbolic function
            %   var - Nx1 symbolic variable of integration
            %   lb - Nx1 vector of lower bounds of each integration variable
            %   ub - Nx1 vector of upper bounds of each integration variable
            
            N = length(var);

            volume = prod(ub-lb); %( ub - lb )^N;   % volume of integration region
            evaluated_points = zeros(1,nsamples);
            sample_points = (ub - lb) .* rand(N,nsamples) + lb; % uniformly sampled points
            for i = 1 : nsamples
                % sample_points(:,i) = (ub - lb) .* rand(N,1) + lb; % uniformly sampled points
                evaluated_points(:,i) = double( subs( integrand, [obj.sys.x ; obj.sys.u], sample_points(:,i) ) );
            end

            result = volume * ( sum( evaluated_points ) / nsamples );
        end

        % Monte Carlo Integrations (takes lifted points as argument)
        function result = intN_mc_points( obj , integrand_pts , lb , ub )
            % Uses monte carlo integration to approximate an integral
            %   integrand_pts - integrand evaluated at a bunch of sample points
            %   lb - Nx1 vector of lower bounds of each integration variable
            %   ub - Nx1 vector of upper bounds of each integration variable
            
            nsamples = length(integrand_pts);

            volume = prod(ub-lb); %( ub - lb )^N;   % volume of integration region

            result = volume * ( sum( integrand_pts ) / nsamples );
        end

        %% Simulation

        function real = sim_real( obj , t , u , x0 )
            %sim_real Simulates the real system model
            
%             [ real.t , real.x ] = ode45( @(tin,xin) obj.get_xdot(tin,xin,uin) , t , x0 );
            
            % simulate one timestep at a time
            real.t = 0;
            real.u = u(1,:);
            real.x = x0';
            real.y = obj.sys.get_y( x0' );
            if obj.has_massmtx
                for i = 2 : length(t)
                    options = odeset( 'Mass' , @(t,x) obj.get_massmtx( t , x ) );
                    [ tout , xout ] = ode45( @(tin,xin) obj.get_rhs(tin,xin,u(i-1,:)') , [t(i-1) , t(i)] , real.x(end,:)' , options);
                    real.t = [ real.t ; tout(end) ];
                    real.x = [ real.x ; xout(end,:) ];
                    real.y = [ real.y ; obj.sys.get_y( xout(end,:) ) ];
%                     real.u = [ real.u ; kron( u(i-1,:) , ones(size(xout,1),1) ) ];
                    real.u = [ real.u ; u(i-1,:) ];
                end
            else
                for i = 2 : length(t)
                    [ tout , xout ] = ode45( @(tin,xin) obj.get_xdot(tin,xin,u(i-1,:)') , [t(i-1) , t(i)] , real.x(end,:)' );
                    real.t = [ real.t ; tout(end) ];
                    real.x = [ real.x ; xout(end,:) ];
                    real.y = [ real.x ; obj.sys.get_y( xout(end,:) ) ];
%                     real.u = [ real.u ; kron( u(i-1,:) , ones(size(xout,1),1) ) ];
                    real.u = [ real.u ; u(i-1,:)];
                end
            end
        end

        function koop = sim_koop( obj , t , u , x0 )
            %sim_koop Simulates the koopman model
            
%             % Atonomous system
%             z0 = obj.lift.basis( x0 );
%             koop.t = t;
%             koop.z = zeros( length(t) , obj.params.N );
%             koop.z(1,:) = z0';
%             for i = 2 : length(t)
%                 koop.z(i,:) = ( expm(obj.Kmtx*t(i)) * z0 )';
%                 %     zkoop_mod(i,:) = ( expm(K*0.001) * zkoop_mod(i-1,:)' )';
%             end
%             koop.x = koop.z(:,1:obj.params.n);

            % System with input
            z0 = obj.lift.basis.x( x0 );
            koop.t = t;
            koop.u = u;
            koop.z = zeros( length(t) , obj.params.N );
            koop.z(1,:) = z0';
            dt = mean( t(2:end) - t(1:end-1) ); % time step
            iso_z = [ eye(obj.params.N) , zeros(obj.params.N , obj.params.Nfull - obj.params.N) ];
            for i = 2 : length(t)

                % Use continuous Koopman matrix
%                 koop.z(i,:) = ( iso_z * expm(obj.Kmtx*dt) * [ koop.z(i-1,:)' ; u(i-1,:)' ] )'; % don't lift at each timestep
                koop.z(i,:) = ( iso_z * expm(obj.Kmtx*dt) * obj.lift.fullbasis.y( koop.z(i-1,1:obj.params.ny)' , u(i-1,:)' ) )';  % lift at each timestep

                % Use discrete Koopman matrix
%                 koop.z(i,:) = ( iso_z * obj.K_dis * obj.lift.fullbasis.y( koop.z(i-1,1:obj.params.ny)' , u(i-1,:)' ) )';  % lift at each timestep

%                 fullbasis_scaled = obj.scale_factor^(-1) * obj.lift.fullbasis.y( koop.z(i-1,1:obj.params.ny)' , u(i-1,:)' );
%                 z_kp1_scaled = ( iso_z * expm(obj.Kmtx*dt) * fullbasis_scaled )';
%                 koop.z(i,:) = z_kp1_scaled * obj.scale_factor;
            end
%             koop.x = koop.z(:,1:obj.params.n);
            koop.y = koop.z(:,1:obj.params.ny);
        end

        %% Test functions for trying out new features

        function x = estimate_y2x_local( obj , y , LB , UB , mod_index )
            %estimate_y2x_local Estimates x from y using approximation of inverse kinematics
            % Assumes y will be a column vector
            
            % WORK IN PROGRESS--IK COMPUTATION HAS BEEN MOVED TO ARM CLASS
            % SO THIS FUNCTION HAS BEEN ABANDONED
%             local_index = 
%             y_local = y(  :  );
            local_end_coords = y_local( end-1 : end ) - y_local( 1 : 2 );
            beta = -atan2( y(5) , y(6) );
            L = norm( y(5:6) );   % distance from base to end effector
            b = 0.25; % obj.sys.params.l;

            cost = @(a) obj.cost_local_ik( a , mod_index , local_end_coords );
            sol = fmincon(@(a) cost , beta , [] , [] , [] , [] , LB , UB );

            % choose solution that doesn't violate bounds
            alpha_valid = sol( sol < UB & sol > LB );
            alpha = alpha_valid(1); % if more than one valid solution, pick first one
            
            x_pos = [ alpha ; alpha ; alpha ];    % first joint angle is different
            x_vel = pinv( obj.sys.get_Jy( x_pos ) ) * y(7:end);
            x = [ x_pos ; x_vel ];
        end

        function cost = cost_local_ik( obj , a , mod_index , local_end_coords )
            %cost_local_ik Penalizes the deviation of the end effector
            %location given joint angle a and the measured end effector
            %location

            x_star = local_end_coords(1);
            y_star = local_end_coords(2);
            beta = -atan2( x_star , y_star );
            L = norm( [ x_star ; y_star ] );   % distance from base to end effector

            dist = L;
            for i = 1 : obj.params.nlinks(mod_index)
                b = obj.params.l{mod_index}(i);
                dist = dist - ( b*cos(beta-a) + b*cos(beta-2*a) + b*cos(beta-3*a) - L );
            end
            cost = dist.^2;
        end


        function [ c , ceq ] = nonlcon_arm3link( obj, a , y )
            x_star = y(end/2-1) - 0;
            y_star = y(end/2) - obj.sys.params.l;  % assumes all links are the same length
            beta = -atan2( x_star , y_star );
            L = norm( [ x_star ; y_star ] );   % distance from base to end effector
            b = obj.sys.params.l;

            ceq(1) = b*cos(beta-a) + b*cos(beta-2*a) + b*cos(beta-3*a) - L;    % distance from origin should match
            ceq(2) = b*cos(a) + b*cos(2*a) + b*cos(3*a) - y_star;    % y coordinate should match

            c = 0;
        end

        function koop = sim_koop_y2x( obj , t , u , x0 )
            %sim_koop_y2x Simulates the koopman model

            % System with input
%             y0 = double( obj.sys.get_y( x0' )' );    % compute output y from x (output is column vec)
% %             x0_ik = obj.estimate_y2x( y0 , -pi/2 , pi/2 ); % estimate x from y using simplified dynamics (i/o are column vecs)
%             x0_ik = obj.sys.est_x_ik( y0 ); % estimate x from y using simplified dynamics (i/o are column vecs)
%             z0 = obj.lift.basis.x( x0_ik );
            z0 = obj.lift.basis.x( x0 );    % use the actual initial condition provided
            koop.t = t;
            koop.u = u;
            koop.z = zeros( length(t) , obj.params.N );
            koop.z(1,:) = z0';
            dt = mean( t(2:end) - t(1:end-1) ); % time step
            iso_z = [ eye(obj.params.N) , zeros(obj.params.N , obj.params.Nfull - obj.params.N) ];
            for i = 2 : length(t)

                % Use continuous Koopman matrix
%                 koop.z(i,:) = ( iso_z * expm(obj.Kmtx*dt) * [ koop.z(i-1,:)' ; u(i-1,:)' ] )'; % don't lift at each timestep

%                 yi = double( obj.sys.get_y( koop.z(i-1,1:obj.params.n) )' );  % compute output y from x (output is column vec)
%                 xi = obj.sys.est_x_ik( yi ); % estimate x from y using simplified dynamics (i/o are column vecs)
                xi = koop.z(i-1,1:obj.params.n)';
%                 koop.z(i,:) = ( iso_z * expm(obj.Kmtx*dt) * obj.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % lift at each timestep
%                 koop.z(i,:) = ( iso_z * obj.scaleup.z * expm(obj.Kmtx*dt) * obj.scaledown.z * obj.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % lift at each timestep, with scaling *****
%                 koop.z(i,:) = ( iso_z * obj.scaleup.z * expm(obj.Kmtx*dt) * obj.scaledown.z * obj.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % different scaling for z and zdot
                koop.z(i,:) = ( iso_z * expm(obj.Kmtx*dt) * obj.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % different scaling for z and zdot (already baked into Kmtx) ************************
%                 koop.z(i,:) = koop.z(i-1,:) + ( iso_z * obj.scaleup.z * ( dt * obj.Kmtx * obj.scaledown.z * obj.lift.fullbasis.x( xi , u(i-1,:)' ) ) )'; % lift at each timestep, with scaling, euler step
%                 koop.z(i,:) = ( iso_z * obj.scaleup.z * obj.Kmtx * obj.scaledown.z * obj.lift.fullbasis.x( xi , u(i-1,:)' ) )'; % Kmtx is discrete matrix in this case
%                 koop.z(i,:) = ( iso_z * expm(obj.Kmtx*dt) * obj.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % scaling is baked into K already

                % koop.z(i,:) = ( iso_z * expm(obj.Kmtx*dt) * [ koop.z(i-1,:)' ; u(i-1,:)' ] )';  % don't lift at each point (linear models only)

                % Use discrete Koopman matrix
%                 koop.z(i,:) = ( iso_z * obj.K_dis * obj.lift.fullbasis.y( koop.z(i-1,1:obj.params.ny)' , u(i-1,:)' ) )';  % lift at each timestep
%                 koop.z(i,:) = ( iso_z * obj.K_dis * obj.lift.fullbasis.x( xi , u(i-1,:)' ) )';  % discrete koopman matrix, lift at each timestep

%                 fullbasis_scaled = obj.scale_factor^(-1) * obj.lift.fullbasis.y( koop.z(i-1,1:obj.params.ny)' , u(i-1,:)' );
%                 z_kp1_scaled = ( iso_z * expm(obj.Kmtx*dt) * fullbasis_scaled )';
%                 koop.z(i,:) = z_kp1_scaled * obj.scale_factor;
            end
            koop.x = koop.z(:,1:obj.params.n);
%             koop.y = koop.z(:,1:obj.params.ny);
        end

        function [ obj , basis , basisdot ] = def_basis_hermite_x( obj , degree )
            %def_basis_hermite Defines hermite polynomial basis
            
            zeta = obj.sys.x;   % basis functions lift the output, not state    
            dzeta = obj.sys.dx;
            nzeta = length(zeta);
            maxDegree = degree;

            % Number of mononials, i.e. dimenstion of p(x)
            N = factorial(nzeta + maxDegree) / ( factorial(nzeta) * factorial(maxDegree) );

            % matrix of exponents (N x naug). Each row gives exponents for 1 monomial
            exponents = [];
            for i = 1:maxDegree
                exponents = [exponents; partitions(i, ones(1,nzeta))];
            end

            % create vector of orderd monomials (column vector)
            for i = 1:N-1
                hermiteBasis(i,1) = obj.get_hermite(zeta, exponents(i,:));
            end

%             % TRYOUT: add constant to the end of the basis functions
%             % UPDATE: doesn't seem to help with DC error
%             hermiteBasis = [ hermiteBasis ; 2 ];

            % output variables
            basis.x = [ hermiteBasis / 2 ];    % symbolic vector of basis monomials, expressed in terms of state, y
            basis.y = obj.sys.y; % basis expressed in terms of output, y (can't write without inverse kinematics)
            basisdot = obj.def_basisdot( basis );

            % save basis to class
            obj.basis.x = basis.x;
            obj.basisdot.x = basisdot.x;
            obj.basis.y = basis.y;
            obj.basisdot.y = basisdot.y;
        end

    end
end