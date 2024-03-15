%Klift_setup
%   Instantiate an instance of the Klift class

%% sample model

% % random dynamics for debugging
% x = sym('x',[3,1]);
% u = sym('u',[2,1]);
% xdot = [x(1) + x(2)*u(1) ; x(2)^2 ; u(1)*u(2) + x(3)];
% x_domain = [ -1 , 1 ; -1 , 1 ; -1 , 1 ];    % first col min, second col max
% u_domain = [ -1 , 1 ; -1 , 1 ];     % first col min, second col max

% % single pendulum
% x = sym('x',[2,1]);
% u = sym('u',[1,1]);
% damp = 1;
% stiff = 2;
% xdot = [ x(2);...
%          -(9.81/0.5) * sin( x(1) ) + 0.5*( u(1) - x(1) ) - damp*x(2)];
% x_domain = [ -1 , 1 ; -1 , 1 ];    % first col min, second col max
% u_domain = [ -1 , 1 ];     % first col min, second col max

% % Duffing oscillator
% x = sym('x',[2,1]);
% u = sym('u',[1,1]);
% delta = 1;
% alpha = 1;
% beta = 0.02;
% xdot = [ x(2);...
%          -delta*x(2) - alpha*x(1) - beta*x(1)^3 + u(1)];
% x_domain = [ -1 , 1 ; -1 , 1 ];    % first col min, second col max
% u_domain = [ -1 , 1 ];     % first col min, second col max
% y = x;

% 2D spring thing (see notebook)
x = sym('x',[4,1]);
u = sym('u',[2,1]);
k = [1,1]';
d = [1,1]';
L = [1,1]';
m = 1;
xdot = [ x(3);...
         x(4);...
         -(k(1)/m)*( sqrt( x(2)^2 + (L(1)+x(1))^2 ) - L(1) ) - (d(1)/m)*x(3) + (1/m)*u(1);...
         -(k(2)/m)*( sqrt( x(1)^2 + (L(2)+x(2))^2 ) - L(2) ) - (d(2)/m)*x(4) + (1/m)*u(2)... 
        ];
x_domain = [ -ones(4,1) , ones(4,1) ];    % first col min, second col max
u_domain = [ -ones(2,1) , ones(2,1) ];     % first col min, second col max
massmtx = sym( eye( length(x) ) );
rhs = xdot;
y = x;

% % Double pendulum with input
% x = sym('x',[4,1]);
% u = sym('u',[2,1]);
% Dinv = [ [1.0,   0,                                                                                                                                                                                                                                                                                                                                                                                 0,                                                                                                                                                                                                                                                                                                                                                                                                                    0];...
%          [  0, 1.0,                                                                                                                                                                                                                                                                                                                                                                                 0,                                                                                                                                                                                                                                                                                                                                                                                                                    0];...
%          [  0,   0,                                                        (43.0*(3.0*cos(x(1) + x(2))^2 + 3.0*sin(x(1) + x(2))^2 + 4.0))/(12.0*cos(x(1) + x(2))^2 + 12.0*sin(x(1) + x(2))^2 + 60.0*cos(x(1))^2 + 60.0*sin(x(1))^2 + 9.0*cos(x(1) + x(2))^2*cos(x(1))^2 + 45.0*cos(x(1) + x(2))^2*sin(x(1))^2 + 45.0*sin(x(1) + x(2))^2*cos(x(1))^2 + 9.0*sin(x(1) + x(2))^2*sin(x(1))^2 - 72.0*cos(x(1) + x(2))*sin(x(1) + x(2))*cos(x(1))*sin(x(1)) + 16.0),                                    -(43.0*(6.0*sin(x(1) + x(2))*sin(x(1)) + 3.0*cos(x(1) + x(2))^2 + 3.0*sin(x(1) + x(2))^2 + 6.0*cos(x(1) + x(2))*cos(x(1)) + 4.0))/(12.0*cos(x(1) + x(2))^2 + 12.0*sin(x(1) + x(2))^2 + 60.0*cos(x(1))^2 + 60.0*sin(x(1))^2 + 9.0*cos(x(1) + x(2))^2*cos(x(1))^2 + 45.0*cos(x(1) + x(2))^2*sin(x(1))^2 + 45.0*sin(x(1) + x(2))^2*cos(x(1))^2 + 9.0*sin(x(1) + x(2))^2*sin(x(1))^2 - 72.0*cos(x(1) + x(2))*sin(x(1) + x(2))*cos(x(1))*sin(x(1)) + 16.0)];...
%          [  0,   0, -(43.0*(6.0*sin(x(1) + x(2))*sin(x(1)) + 3.0*cos(x(1) + x(2))^2 + 3.0*sin(x(1) + x(2))^2 + 6.0*cos(x(1) + x(2))*cos(x(1)) + 4.0))/(12.0*cos(x(1) + x(2))^2 + 12.0*sin(x(1) + x(2))^2 + 60.0*cos(x(1))^2 + 60.0*sin(x(1))^2 + 9.0*cos(x(1) + x(2))^2*cos(x(1))^2 + 45.0*cos(x(1) + x(2))^2*sin(x(1))^2 + 45.0*sin(x(1) + x(2))^2*cos(x(1))^2 + 9.0*sin(x(1) + x(2))^2*sin(x(1))^2 - 72.0*cos(x(1) + x(2))*sin(x(1) + x(2))*cos(x(1))*sin(x(1)) + 16.0), (43.0*(12.0*sin(x(1) + x(2))*sin(x(1)) + 3.0*cos(x(1) + x(2))^2 + 3.0*sin(x(1) + x(2))^2 + 15.0*cos(x(1))^2 + 15.0*sin(x(1))^2 + 12.0*cos(x(1) + x(2))*cos(x(1)) + 8.0))/(12.0*cos(x(1) + x(2))^2 + 12.0*sin(x(1) + x(2))^2 + 60.0*cos(x(1))^2 + 60.0*sin(x(1))^2 + 9.0*cos(x(1) + x(2))^2*cos(x(1))^2 + 45.0*cos(x(1) + x(2))^2*sin(x(1))^2 + 45.0*sin(x(1) + x(2))^2*cos(x(1))^2 + 9.0*sin(x(1) + x(2))^2*sin(x(1))^2 - 72.0*cos(x(1) + x(2))*sin(x(1) + x(2))*cos(x(1))*sin(x(1)) + 16.0)]...
%        ];
% RHS = [ x(3);...
%         x(4);...
%         10.0*u(1) - 10.0*x(1) - 10.0*x(3) - 3.7*sin(x(1) + x(2)) - 11.0*sin(x(1)) - 1.0*x(3)*((0.19*cos(x(1) + x(2))*(x(3) + x(4)) + 0.38*x(3)*cos(x(1)))*(0.38*sin(x(1) + x(2)) + 0.75*sin(x(1))) + (0.38*cos(x(1) + x(2))*(x(3) + x(4)) + 0.75*x(3)*cos(x(1)))*(0.19*sin(x(1) + x(2)) + 0.38*sin(x(1))) - 1.0*(0.19*sin(x(1) + x(2))*(x(3) + x(4)) + 0.38*x(3)*sin(x(1)))*(0.38*cos(x(1) + x(2)) + 0.75*cos(x(1))) - 1.0*(0.38*sin(x(1) + x(2))*(x(3) + x(4)) + 0.75*x(3)*sin(x(1)))*(0.19*cos(x(1) + x(2)) + 0.38*cos(x(1)))) - 1.0*x(4)*(0.19*sin(x(1) + x(2))*(0.38*cos(x(1) + x(2))*(x(3) + x(4)) + 0.75*x(3)*cos(x(1))) - 0.19*cos(x(1) + x(2))*(0.38*sin(x(1) + x(2))*(x(3) + x(4)) + 0.75*x(3)*sin(x(1))) - 0.19*sin(x(1) + x(2))*(x(3) + x(4))*(0.38*cos(x(1) + x(2)) + 0.75*cos(x(1))) + 0.19*cos(x(1) + x(2))*(x(3) + x(4))*(0.38*sin(x(1) + x(2)) + 0.75*sin(x(1))));...
%         10.0*u(2) - 10.0*x(2) - 10.0*x(4) - 3.7*sin(x(1) + x(2)) - 1.0*x(3)*(0.38*sin(x(1) + x(2))*(0.19*cos(x(1) + x(2))*(x(3) + x(4)) + 0.38*x(3)*cos(x(1))) - 0.38*cos(x(1) + x(2))*(0.19*sin(x(1) + x(2))*(x(3) + x(4)) + 0.38*x(3)*sin(x(1))) - 0.38*sin(x(1) + x(2))*(x(3) + x(4))*(0.19*cos(x(1) + x(2)) + 0.38*cos(x(1))) + 0.38*cos(x(1) + x(2))*(x(3) + x(4))*(0.19*sin(x(1) + x(2)) + 0.38*sin(x(1)))) - 1.0*x(3)*(0.5*x(3)*(0.38*sin(x(1) + x(2))*(0.19*cos(x(1) + x(2)) + 0.38*cos(x(1))) + 0.19*sin(x(1) + x(2))*(0.38*cos(x(1) + x(2)) + 0.75*cos(x(1))) - 0.38*cos(x(1) + x(2))*(0.19*sin(x(1) + x(2)) + 0.38*sin(x(1))) - 0.19*cos(x(1) + x(2))*(0.38*sin(x(1) + x(2)) + 0.75*sin(x(1)))) + 0.5*x(4)*(0.38*sin(x(1) + x(2))*(0.19*cos(x(1) + x(2)) + 0.38*cos(x(1))) - 0.38*cos(x(1) + x(2))*(0.19*sin(x(1) + x(2)) + 0.38*sin(x(1))))) - 0.5*x(3)*x(4)*(0.19*sin(x(1) + x(2))*(0.38*cos(x(1) + x(2)) + 0.75*cos(x(1))) - 0.19*cos(x(1) + x(2))*(0.38*sin(x(1) + x(2)) + 0.75*sin(x(1))));...
%       ];
% xdot = Dinv*RHS;
% x_domain = [ -ones(4,1) , ones(4,1) ];    % first col min, second col max
% u_domain = [ -ones(2,1) , ones(2,1) ];     % first col min, second col max

% % Double pendulum arm with input, with mass matrix
% load([ 'systems' , filesep , 'arm_2-mods_1-links' , '.mat' ] );
% sys_in = sys;
% sys_in.dx = sym('dx', size(sys_in.x) );

% % Triple pendulum arm with input, with mass matrix
% load([ 'systems' , filesep , 'arm_3-mods_1-links' , '.mat' ] );
% sys_in = sys;
% sys_in.dx = sym('dx', size(sys_in.x) );
% sys_in.y = sym('y', size(sys_in.x) );

% % Triple pendulum template model (kinematics mostly) arm with input, with mass matrix
% load([ 'systems' , filesep , 'arm_3-mods_1-links_kinematics' , '.mat' ] );
% sys_in = sys;
% sys_in.dx = sym('dx', size(sys_in.x) );

% % Nine segment arm with input, with mass matrix
% load([ 'systems' , filesep , 'arm_3-mods_3-links' , '.mat' ] );
% sys_in = sys;
% sys_in.dx = sym('dx', size(sys_in.x) );

% save in single struct
sys_in.x = x;
sys_in.u = u;
sys_in.xdot = xdot;
sys_in.x_domain = x_domain;
sys_in.u_domain = u_domain;
sys_in.y = y;

% for mechanical systems of the form M(x)*xdot = f(x,u)
sys_in.massmtx = massmtx;
sys_in.rhs = rhs;
sys_in.dx = sym('dx', size(sys_in.x) );
sys_in.dy = sys_in.dx;

% % 9 link arm with three modules, with marker locations as the output
% load([ 'systems' , filesep , 'Arm_arm_3-mods_3-links_markers' , '.mat' ] );
% sys_in = sys;

% % 6 link arm with three modules, with marker locations as the output
% load([ 'systems' , filesep , 'arm_3-mods_2-links_markers' , '.mat' ] );
% sys_in = sys;

% % 3 link arm with three modules, with marker locations as the output
% load([ 'systems' , filesep , 'arm_3-mods_1-links_markers' , '.mat' ] );
% sys_in = sys;

% % 1 link arm with three modules, with marker locations as the output
% load([ 'systems' , filesep , 'arm_1-mods_1-links_markers' , '.mat' ] );
% sys_in = sys;

% % 1 link arm with three modules, with angles as the output
% load([ 'systems' , filesep , 'arm_1-mods_1-links_angles' , '.mat' ] );
% sys_in = sys;

%------------- Outputs with velocities below this line ---------------

% % 1 link arm with 1 modules, with angles as the output, with velocities
% % in output
% load([ 'systems' , filesep , 'arm_1-mods_1-links_angles_vel' , '.mat' ] );
% sys_in = sys;

% % 3 link arm with 3 modules, with angles as the output, with velocities
% % in output
% load([ 'systems' , filesep , 'arm_3-mods_1-links_angles_vel' , '.mat' ] );
% sys_in = sys;

% % 1 link arm with one modules, with markers as the output, with velocities
% % in output
% load([ 'systems' , filesep , 'arm_1-mods_1-links_markers_vel' , '.mat' ] );
% sys_in = sys;

% % 2 link arm with 2 modules, with markers as the output, with velocities
% % in output
% load([ 'systems' , filesep , 'arm_2-mods_1-links_markers_vel' , '.mat' ] );
% sys_in = sys;

% % Arm, 2 modules, 1 link per module, end effector as output, with velocity
% load([ 'systems' , filesep , 'arm_2-mods_1-links_endeff_vel' , '.mat' ] );
% sys_in = sys;

% % 3 link arm with 3 modules, with markers as the output, with velocities
% % in output
% load([ 'systems' , filesep , 'arm_3-mods_1-links_markers_vel(2)' , '.mat' ] );
% sys_in = sys;

% % Full arm, 4 modules with 15 links total
% load([ 'systems' , filesep , 'arm_4-mods_15-links_markers_vel' , '.mat' ] );
% sys_in = sys;

% % Full arm, 3 modules with 14 links total
% load([ 'systems' , filesep , 'arm_3-mods_14-links_markers_vel' , '.mat' ] );
% sys_in = sys;

% % Part of arm, 1 modules with 5 links total
% % load([ 'systems' , filesep , 'arm_1-mod_5-links_markers_vel' , '.mat' ] );
% load([ 'systems' , filesep , 'tau-arm_1-mod_5-links_mass-50g' , '.mat' ] );
% sys_in = sys;

% % Part of arm, 2 modules with 10 links total
% load([ 'systems' , filesep , 'arm_2-mods_10-links_markers_vel' , '.mat' ] );
% sys_in = sys;

% % Arm, 3 modules with 14 links total
% load([ 'systems' , filesep , 'tau-arm_3-mods_14-links_mass-50g' , '.mat' ] );
% sys_in = sys;

% Arm, 3 modules with 14 links total
% load([ 'systems' , filesep , 'tau-arm_3-mods_14-links_mass-50g_kjoints-50_damp-05_ufixed' , '.mat' ] );
% load([ 'systems' , filesep , 'tau-arm_3-mods_14-links_mass-50g_kjoints-20_damp-05_ufixed_Bf-79' , '.mat' ] ); %*******
% sys_in = sys;

% Systems with pressure as input--------------------------------

% % Part of arm, 1 modules with 5 links total
% load([ 'systems' , filesep , 'parm_1-mod_5-links_mass-50g' , '.mat' ] );
% sys_in = sys;

% Systems with pressure (in psi) as input--------------------------------

% % Part of arm, 1 modules with 5 links total
% % load([ 'systems' , filesep , 'psi-arm_1-mod_5-links_mass-50g' , '.mat' ] );
% load([ 'systems' , filesep , 'psi-arm_1-mods_5-links_mass-50g_kjoints-20_damp-05_Bf-79' , '.mat' ] );
% sys_in = sys;

% % Arm, 3 modules with 14 links total
% load([ 'systems' , filesep , 'psi-arm_3-mods_14-links_mass-50g' , '.mat' ] );
% sys_in = sys;


% Systems to help with debugging %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load([ 'systems' , filesep , 'simple_linear' , '.mat' ] );
% sys_in = sys;

% load([ 'systems' , filesep , 'simple_nonlinear' , '.mat' ] );
% sys_in = sys;

% Arms with the end effector position embedded into the state %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load( ['systems', filesep, 'tau-arm_3-mods_14-links_mass-50g_kjoints-20_damp-05_ufixed_Bf-79xyz_embed' , '.mat'] );
% sys_in = sys;

% load( ['systems', filesep, 'arm_3-mods_14-links_markers_vel_xyz_embed' , '.mat'] );
% sys_in = sys;

% load( ['systems', filesep, 'tau-arm_1-mod_5-links_mass-50g_xyz_embed' , '.mat'] );
% sys_in = sys;


%% Set time constant
dt = (1/50); % 1e-3;

%% construct class
% Klift = Klift( sys_in ,...
%                'model_type' , 'linear' ,...
%                'basis_degree' , 3 ,...
%                'basis_type' , 'hermite' ,...
%                'has_massmtx' , true ,...
%                'num_samples' , 1e3 ,...
%                'has_output' , false ... % I don't think this does anything anymore
%              );

Klift = Klift( sys_in ,...
               'model_type' , 'linear' ,...
               'basis_degree' , 2 ,...
               'basis_type' , 'hermite' ,...
               'has_massmtx' , true ,...
               'num_samples' , 3 ,...
               'integral_res' , 2 ,... % how finely to discretize each dimension when approximating IPs
               'has_output' , false ,... % I don't think this does anything anymore
               'timestep' , dt ...
             );

%% simulated real and identified models (SKIP FOR NOW SO WE DON'T HAVE TO SIM EACH TIME)

tin = ( 0 : dt : 100 )';

% uin = zeros( length(tin) ,  Klift.params.m );
% uin = 0.7 * rand( size(tin,1) , Klift.params.m );
% uin = [cos(tin),-sin(tin)];
% uin = sin(2*tin) .* ones( length(tin) ,  Klift.params.m ) .* 20;
uin = ones( length(tin) ,  Klift.params.m ) * (0.050);   % 3*cos(tin);
% uin = ones( length(tin) ,  Klift.params.m ) * 0.55 .* (1 - exp(-tin*1e0));

% uin = fliplr([ 0.5e3 * ones(length(tin),1) , zeros(length(tin),1) ]);   % for 2d input
% uin = fliplr([ 5 * ones(length(tin),1) , zeros(length(tin),1) ]);   % for 2d input (psi)
% uin = fliplr([ 10 * sin(2.*tin) + 10 , zeros(length(tin),1) ]);   % for 2d input (psi)
% uin = [ zeros(length(tin),1) , 10 * sin(2.*tin) + 10 , zeros(length(tin),1) , 10 * sin(2.*tin) + 10 , zeros(length(tin),1) , 10 * sin(2.*tin) + 10 ];   % for 6d input (psi)

% x0 = 1 * (2*rand( Klift.params.n , 1 ) - 1);
% x0 = (2*rand(1,1)-1) .* [ ones(Klift.params.n/2,1) ; zeros(Klift.params.n/2,1) ];   % all joint angles are the same, no joint velocity
x0 = zeros(Klift.params.n,1);   % all joint angles and velocites are zero
% x0 = ones(Klift.params.n,1);   % all joint angles and velocites are one

real = Klift.sim_real(tin,uin,x0);
% 
% koop = Klift.sim_koop(tin,uin,x0);
% 
% % plot the difference in the first state
% figure;
% hold on;
% plot(real.t,real.y(:,1));
% plot(koop.t,koop.y(:,1));
% hold off;
% legend('Real','Koopman');

%% load data rather than having to simulate real system each time we model it
% 
% % % 3mod arm, joint torque input
% % real_sys_name = 'tau-arm_3-mods_14-links_mass-50g_kjoints-20_damp-05_ufixed_Bf-79';
% % datafile_name = '2023-05-19_14-59';
% % load([ 'Arm_objects' , filesep , real_sys_name , filesep , 'simulations' , filesep , datafile_name , '.mat' ] );
% % real = data{1};
% 
% % % 1mod arm, joint torque input
% % real_sys_name = 'tau-arm_1-mod_5-links_mass-50g';
% % datafile_name = '2023-06-16_14-50';
% % load([ 'Arm_objects' , filesep , real_sys_name , filesep , 'simulations' , filesep , datafile_name , '.mat' ] );
% % real = data{1};
% 
% % 1mod arm, psi input
% real_sys_name = 'psi-arm_1-mods_5-links_mass-50g_kjoints-20_damp-05_Bf-79';
% datafile_name = '2023-05-24_16-17';
% load([ 'Arm_objects' , filesep , real_sys_name , filesep , 'simulations' , filesep , datafile_name , '.mat' ] );
% real = data{1};
% 
% 
% real.x = data{1}.Q; %_ik;
% % real.x = [ real.y(:, end/2-2 : end/2) , real.Q ];% DEBUG**** for systems with the end effector position embedded
% 
% % % 1mod arm, psi input
% % real_sys_name = 'psi-arm_1-mods_5-links_mass-50g_kjoints-10_damp-05_Bf-79';
% % datafile_name = '2023-05-24_16-20';
% % load([ 'Arm_systems' , filesep , real_sys_name , filesep , 'simulations' , filesep , datafile_name , '.mat' ] );
% % real = data{1};
% % real.x = data{1}.Q; %_ik;
% 
% tin = real.t;
% % uin = [ real.u(:,1) , real.u(:,2)*(5/3) , real.u(:,3) ];  % debug, trying something out
% uin = real.u;
% x0 = real.x(1,:)';


%% Beta testing features: Simulate the version with IK

koop_ik = Klift.sim_koop_y2x(tin,uin,x0);

% % plot the difference in the first state
% figure;
% hold on;
% plot(real.t,real.x(:,1));
% plot(koop_ik.t,koop_ik.x(:,1));
% hold off;
% legend('Real','Koopman');

% plot the difference in all the states
figure;
num_plots = size( real.x, 2 )/2;
% num_plots = 3;  % DEBUG**** for systems with the end effector position embedded
for i = 1 : num_plots
    subplot(num_plots, 1, i);
    hold on;
    plot(real.t,real.x(:,i));
    plot(koop_ik.t,koop_ik.x(:,i));
    hold off;
%     ylim([-0.5,0.5]);   % for the arm example
end
legend('Real','Koopman');

total_error = sum( vecnorm(real.x - koop_ik.x, 2, 2) ) / length(tin)
total_sampled_error = Klift.estimate_error

%% Improve model over and over and see what happens to the error

num_improvements = 10;
nrmse = zeros(num_improvements,1);
sampled_error = zeros(num_improvements,1);
for i = 1:num_improvements
%     Klift = Klift.improve_koopman_massmtx( 'NumPts', 1e3 );
    Klift = Klift.improve_koopman_massmtx( 'SampleType', 'grid', 'GridResolution', 2, 'GridScale', i/21 );
%     Klift = Klift.improve_koopman_massmtx( 'SampleType', 'grid', 'GridResolution', 2+i );
    koop_ik = Klift.sim_koop_y2x(tin,uin,x0);
    nrmse(i) = sum( vecnorm(real.x - koop_ik.x, 2, 2) ) / length(tin);
    sampled_error(i) = Klift.estimate_error('NumPts', 1e4);
%     nrmse(i)
    sampled_error(i)
end

% plot the difference in all the states
figure;
num_plots = size( real.x, 2 )/2;
for i = 1 : num_plots
    subplot(num_plots, 1, i);
    hold on;
    plot(real.t,real.x(:,i));
    plot(koop_ik.t,koop_ik.x(:,i));
    hold off;
%     ylim([-0.5,0.5]);   % for the arm example
end
legend('Real','Koopman');

%% Improve model over and over and see what happens to the error using grid shift/refinement method
% 
% state_index_offset = [0, 1];
% all_offset_combos = permn( state_index_offset, Klift.params.n+Klift.params.m );
% % all_offset_combos = [1 1 1 1 1 1 1 1 1; 2 2 2 2 2 2 2 2 2; 3 3 3 3 3 3 3 3 3];
% 
% num_improvements = size(all_offset_combos, 1);
% nrmse = zeros(num_improvements,1);
% sampled_error = zeros(num_improvements,1);
% for i = 1:num_improvements
%     Klift = Klift.improve_koopman_massmtx_grid_subset( all_offset_combos(i,:) );
%     koop_ik = Klift.sim_koop_y2x(tin,uin,x0);
% %     nrmse(i) = sum( vecnorm(real.x - koop_ik.x, 2, 2) ) / length(tin);
%     sampled_error(i) = Klift.estimate_error('NumPts', 1e3);
% %     nrmse(i)
%     sampled_error(i)
% end
% 
% % plot the difference in all the states
% figure;
% num_plots = size( real.x, 2 )/2;
% for i = 1 : num_plots
%     subplot(num_plots, 1, i);
%     hold on;
%     plot(real.t,real.x(:,i));
%     plot(koop_ik.t,koop_ik.x(:,i));
%     hold off;
% %     ylim([-0.5,0.5]);   % for the arm example
% end
% legend('Real','Koopman');








