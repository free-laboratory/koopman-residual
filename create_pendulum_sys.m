% create_new_duffing_sys
%
% Creates a nonlinear
% sys object that can be used to initialize an instance of the Klift class

x = sym('x',[2,1]);
u = sym('u',[1,1]);
dx = sym('dx', size(x) );
y = sym('y',[1,1]);
dy = sym('dy',[1,1]);
syms t;

massmtx = sym( eye(2) );

% % system parameters (real system)
% m = 1;  % [kg]
% c = 0.2;    % damping
% l = 1; % [m]
% M = 1; % [N/m]
% g = 9.81;   % [m/s^2]

% system parameters (template model)
m = 0.9;  % [kg]
c = 1.0;    % damping
l = 1; % [m]
M = 1.1; % [N/m]
g = 9.81;   % [m/s^2]

% dynamics
rhs = [x(2);...
       (-g*sin(x(1))/l - c*x(2)/(m*l^2) - cos(x(1))/(l*(M+m)) * (u + m*l*sin(x(1))*x(2)^2) ) / (1 - (m*cos(x(1))^2 / (M+m))) ];

% state and input bounds
x_domain = [ -10, 10;...
             -10, 10 ];
u_domain = [ -10 , 10 ]; 

%% save to a struct
sys.get_y = @(x) x(1); % function to get output from the state
sys.est_x_ik = @(y) [y;0];  % function to estimate state from output
sys.x_domain = x_domain;
sys.u_domain = u_domain;
sys.massmtx = massmtx;
sys.rhs = rhs;
sys.vf_massMtx = matlabFunction( massmtx, "Vars", {t,x,u} );
sys.vf_RHS = matlabFunction( rhs, "Vars", {t,x,u} );
sys.output = x(1); % y expressed as a function of x 
sys.x = x;
sys.dx = dx;
sys.dy = dy;
sys.u = u;
sys.t = t;
sys.y = y; 
% sys.params.n = length(x);
% sys.params.m = length(u);
% sys.params.Nmods = length(x)/2;
% sys.params.ny = length(y);

sys.params.m = m;
sys.params.c = c;
sys.params.l = l;
sys.params.M = M;
sys.params.g = g;

unique_fname = auto_rename(['systems' , filesep , 'cargo_crane_temp' , '.mat' ], '(0)');
save(unique_fname, 'sys');