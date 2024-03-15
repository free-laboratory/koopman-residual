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
-sys.output = @(x) <output(x)>;   % output expressed as a funciton of sys.x
sys.params = <params>;           % scruct of any relevant system parameters (optional)

-sys.get_y = @(x) <y(x)>;          % function to get output from the state
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


### Pendulum on a cart



### How to set up a new example system

Create a system object (need to explain all the parts of this)

Generate data with simulate_sys (note how you can add noise)

Run model_comparision, with the system you've defined. It will construction Klift, Kres, and plot some results
