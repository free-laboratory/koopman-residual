# README

## How to define a nonlinear system model

A physics-based Koopman model is based off of a nonlinear dynamical system model of the form, $M(x)\dot{x} = f(x,u)$. We define such a model as a `sys` object, which is a We store the information that is needed to construct a Koopman model as a struct with the following fields:

```apache
sys = struct;

sys.x = sym('x',[n,1]);          % nx1 symbolic state variable
sys.dx = sym('dx', size(x) );    % nx1 symbolic variable representing dx/dt
sys.u = sym('u',[m,1]);;         % mx1 symbolic input varible 
sys.t = syms t;                  % symbolic time variable
sys.y = sym('y',[1,1]);          % symbolic output variable
sys.dy = sym('dy', size(y) );    % symbolic variable representing dy/dt
sys.output = <output(sys.x)>;    % output expressed as a funciton of sys.x
sys.params = <params>;           % scruct of any relevant system parameters (optional)

sys.get_y = @(x) <y(x)>;          % function to get output from the state
sys.x_domain = <x_domain>;        % nx2 matrix defineing the upper/lower bounds for each state
sys.u_domain = <u_domain>;        % mx2 matrix defineing the upper/lower bounds for each input
sys.massmtx = <massmtx>;          % mass matrix, M(x)
sys.rhs = <rhs>;                  % right-hand side of dynamics, f(x,u)

sys.vf_massMtx = matlabFunction( massmtx, "Vars", {t,x,u} );    % function that evaluates the mass matrix
sys.vf_RHS = matlabFunction( rhs, "Vars", {t,x,u} );    % function that evaluates f(x,u) 
```

A `sys` object contains all of the information that is needed to construct a physics-based Koopman model. Therefore, a `sys` object must be loaded into the Matlab workspace prior to identifying a physics-based Koopman model. It is one of the arguments into the Klift class, which is described in the next section.

## How to identify a physics-based Koopman model

Klift.m defines a class that includes all of the functions needed to identify a physics-based Koopman model from a nonlinear ODE description of that system. An instance of the Klift class is created with the following command and arguments:

```apache
Klift_instance = Klift( sys ,... % object that describe nonlinear system dynamics (see description below)
    'basis_type', 'hermite' ,...   % type of function used for observables (this code only supports Hermite polynomials)
    'basis_degree' , 1 ,...        % maximum degree of the (hermite polynomial) basis functions
    'num_samples' , 1e6 ,...       % number of sample points to use for Monte Carlo integration
    'timestep' , dt ...            % length of one timestep for the discrete-time Koopman model
    );
```

Based on the arguments defined above, the class constructor function will define the following attributes:

* Kmtx - The best approximation of the continuous-time Koopman matrix
* K_dis - The corresponding discrete-time Koopman matrix for the specified timestep (dt), defined as K_dis = expm( Kmtx * dt )

## How to identify data-driven and combined Koopman models

Kres.m defines a class that includes all of the functions to identify a data-driven Koopman model, and a combined (physics-based+data-driven) Koopman model given a physics-based Koopman model and a set of experimental training data. An instance of the Kres class is created with the following command and arguments:

```apache
Kres_instance = Kres( Klift_instance , train_data,...
    'lasso' , 0 );    % specifies the L1 regularization parameter (lasso=0 corresponds to least-squares solution)
```

Based on the arguments defined above, the class constructor function will define the following attributes:

* K_phys - The discrete-time physics-based Koopman matrix (equivalent to Klift_instance.Kdis)
* K_data - The discrete-time data-driven Koopman matrix, approximated from the provided training data
* K_combined - The discrete-time combined Koopman matrix, approximated from the training data and K_phys
* K_res - The residual Koopman matrix, defined as
* lambda - The ... TODO, fill in based on changes to the manuscripte

## Examples

### Van der Pol Oscillator

The 'systems' folder contains sys objects for Van der Pol oscillator systems with the following dynamics:

$$
\begin{bmatrix} \dot{x}_1 (t) \\ \dot{x}_2 (t) \end{bmatrix} =
    \begin{bmatrix} x_2 (t) \\ \mu \left( 1 - x_1 (t)^2 \right) x_2 (t) - x_1 (t) \end{bmatrix}

$$

It also contains simulated data of the Van der Pol systems starting from various initial conditions. Running vanderpol_rmse_vs_traindata.m will go through the process of identifying physics-based, data-driven, and combined models for these systems, and compare the accuracy of their predictions for different amounts of training data.

### Pendulum on a cart

The 'systems' folder contains sys objects for pendulum on a cart systems with the following dynamics:

$$
\begin{bmatrix} \dot{x}_1 \\ \\ \dot{x}_2 \end{bmatrix}
    =
    \begin{bmatrix}
        x_2   \\ \\
        \frac{ 
            -\frac{g}{l}\sin{x_1  } - \frac{c_d}{ml^2} x_2   - \frac{\cos{x_1  }}{l(M+m)} \left( u + ml \sin{x_1  } x_2  ^2 \right)
        }{(1- \frac{m \cos^2{x_1  ^2}}{M+m})}
    \end{bmatrix}

$$

The folder also contains simulated data of the pendulum on a cart systems starting from various initial conditions. Running pendulum_rmse_vs_traindata.m will go through the process of identifying physics-based, data-driven, and combined models for the 'pendulum_cart_real' system, and compare the accuracy of their predictions for different amounts of training data.

### How to set up a new example system

-Create your own 'sys' object by modifying create_pendulum_sys.m to encode the dynamics of your system.

-Generate data for your system with simulate_sys.m. You can generate multiple trials within one file so that you can use some for training and some for validation in the next step.

-Edit pendulum_model_comparison.m to load in the system(s) you've defined. It will construct physics-based, data-driven, and combined models of your system, and calculate the prediction error of each for the validation trial you have specified. It can also generate plots
