% create_vanderpol_sys
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
mu = rand(1,1) * 1;
rhs = [x(2);...
       mu * ( 1 - x(1)^2 ) * x(2) - x(1) ];  
x_domain = [ -2, 2 ];
u_domain = [ 0, 1 ];

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

unique_fname = auto_rename(['systems' , filesep , 'vanderpol' , '.mat' ], '(0)');
save(unique_fname, 'sys');