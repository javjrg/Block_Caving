#----------------------------------------------------------------
#  IMPORT ALL PARAMETERS FROM parameters.py.
#----------------------------------------------------------------
from parameters import *
# read paramaters from command line
parameters.parse()   
# set some dolfin specific parameters
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["allow_extrapolation"] = True
# The minimization procedure requires parameters to get a suitable performance. The following is a suitable set of arrangements.
solver_minimization_parameters =  {"method" : "gpcg", 
                                    "linear_solver" : "gmres",
                            #--------------------------------
                            # These are parameters for optimization
                            #--------------------------------
                            "line_search": "armijo",
                            "preconditioner" : "hypre_euclid", 
                            "maximum_iterations" :200,  
                            "error_on_nonconvergence": False,
                            #--------------------------------
                            # These are parameters for linear solver
                            #--------------------------------
                            "krylov_solver" : {
                                "maximum_iterations" : 200,
                                "nonzero_initial_guess" : True,
                                "report" : True,
                                "monitor_convergence" : False,
                                "relative_tolerance" : 1e-8
                            }
                           }
# The linear solver requires parameters to get a suitable performance. The following is a suitable set of arrangements.
solver_LS_parameters =  {"linear_solver" : "cg",
                            "symmetric" : True,  
                            "preconditioner" : "jacobi", 
                            "krylov_solver" : {
                                "report" : True,
                                "monitor_convergence" : False,
                                "relative_tolerance" : 1e-8
                                }
                            }
#----------------------------------------------------------------
# DEFINE THE OPERATORS.
#----------------------------------------------------------------
# Constitutive functions of the damage model
def w(alpha):
    if model=="lin":
        return w11*alpha
    if model =="quad":
        return w11*alpha**2  
def A(alpha):
    return (1-alpha)**2 
# Define th Deviator and Spheric Tensors
def Dev(Tensor):
    return Tensor-Sph(Tensor)
def Sph(Tensor):
    Tensor2=as_matrix([[1,0],[0,1]])
    return (inner(Tensor,Tensor2)/inner(Tensor2,Tensor2))*Tensor2    
# Strain and stress in free damage regime.
def eps(v):
    return sym(grad(v))  
def sigma_0(eps):
    return 2.0*mu*(eps)+lmbda*tr(eps)*Identity(ndim)
# modification of the Young modulus. 
def sigma(eps,alpha):
    return (A(alpha)+k_ell)*sigma_0(eps)
#----------------------------------------------------------------
#  MESH WITHOUT CAVITY.
#----------------------------------------------------------------
# Define the function, test and trial fields.
# "u, du, v" are vectorial expressions.
u   =   Function(V_vector,name="u")
du  =   TrialFunction(V_vector)
v   =   TestFunction(V_vector)
# "alpha, dalpha, beta" are scalar.
alpha   =   Function(V_scalar,name="alpha")
alpha0  =   Function(V_scalar,name = "alpha")
dalpha  =   TrialFunction(V_scalar)
beta    =   TestFunction(V_scalar)
# Define the energies functions.
W_energy    =   Function(V_scalar,name="energy_w")
dev_energy  =   Function(V_scalar,name="energy_dev")
div_energy  =   Function(V_scalar,name="energy_sph")
E_energy    =   Function(V_scalar,name="energy_E")
p_energy    =   Function(V_scalar,name="energy_p")
dis_energy  =   Function(V_scalar,name="energy_dis")
# Define the stress and strain functions.
stressG =   Function(V_tensor,name="sigma")
strainG =   Function(V_tensor,name="epsilon")
# Define the alpha auxiliar function.
alphaAux    =   Function(V_scalar,name="alpha")
# Define the function "alpha_error" to measure relative error.
alpha_error =   Function(V_scalar)
# Interpolate the initial condition for the damage variable "alpha".
alpha_0     =   interpolate ( Expression("0.",degree = 2), V_scalar)
# Define ds and dx.
ds  =   Measure('ds',domain=mesh,subdomain_data=boundaries)
dx  =   Measure('dx',domain=mesh)
# Let us define the total energy of the system as the sum of elastic energy, dissipated energy due to the damage and external work due to body forces. 
# Elastic energy for differents cases.
if case=="Marigo":
    elastic_energy1 =   0.5*inner(sigma(eps(u),alpha),eps(u))*dx
    elastic_energy2 =   0.5*inner(sigma(eps(u),alpha),eps(u))*dx
if case=="DMSF":
    Id_Tensor       =   as_matrix([[1,0],[0,1]]) 
    elastic_energy1 =   (0.5*((lmbda+mu)*(tr(eps(u))**2)*inner(Id_Tensor,Id_Tensor)))*dx+((A(alpha)+k_ell)*mu*inner(Dev(eps(u)),Dev(eps(u))))*dx
    elastic_energy2 =   (0.5*((lmbda+mu)*(tr(eps(u))**2)*inner(Id_Tensor,Id_Tensor)))*dx+((A(alpha)+k_ell)*mu*inner(Dev(eps(u)),Dev(eps(u))))*dx 
if case=="ShearComp": 
    elastic_energy1 =   0.5*inner(sigma(eps(u),alpha),eps(u))*dx
    elastic_energy2 =   0.5/E*(inner(Dev(sigma(eps(u),alpha)),Dev(sigma(eps(u),alpha)))\
                            -kappa*inner(Sph(sigma(eps(u),alpha)),Sph(sigma(eps(u),alpha))))*dx  
# External work.
external_work   =   dot(body_force,u)*dx
# Dissipated energy.          
dissipated_energy   =   (w(alpha)+ellv**2*w1*dot(grad(alpha),grad(alpha)))*dx
# Definition of the total energy
total_energy1   =   elastic_energy1+dissipated_energy-external_work 
total_energy2   =   elastic_energy2+dissipated_energy-external_work
# Weak form of damage problem. This is the formal expression for the tangent problem which gives us the equilibrium equations.
E_u     =   derivative(total_energy1,u,v)
E_alpha =   derivative(total_energy2,alpha,beta)
# Hessian matrix.
E_alpha_alpha   =   derivative(E_alpha,alpha,dalpha)
# Writing tangent problems in term of test and trial functions for matrix assembly.
E_du     =  replace(E_u,{u:du})
E_dalpha =  replace(E_alpha,{alpha:dalpha})
# Once the tangent problems are formulated in terms of trial and text functions, we define the variatonal problems.
# Variational problem for the displacement.
problem_u   =   LinearVariationalProblem(lhs(E_du),rhs(E_du),u,bc_u)
# Define the classs Optimization Problem for then define the damage.
# Variational problem for the damage (non-linear to use variational inequality solvers of petsc).
class DamageProblem(OptimisationProblem):   
    def __init__(self):
        OptimisationProblem.__init__(self)  
    # Objective vector
    def f(self,x):
        alpha.vector()[:]=x
        return assemble(total_energy2) 
    # Gradient of the objective function
    def F(self,b,x):
        alpha.vector()[:]=x
        assemble(E_alpha,tensor=b)
    # Hessian of the objective function
    def J(self,A,x):
        alpha.vector()[:]=x
        assemble(E_alpha_alpha, tensor=A)
# Define the minimization problem using the class.
problem_alpha   =   DamageProblem()
# Set up the solvers. Define the object for solving the displacement problem, "solver_u".
solver_u    =   LinearVariationalSolver(problem_u)
# Get the set of paramters for the class "solver_u". This only requires the solution of linear system solver.
solver_u.parameters.update(solver_LS_parameters)
# Define the corresponding object, "solver_alpha".
# The object associated to minimization is created.
solver_alpha    =   PETScTAOSolver()
# Get the set of paramters for the class "solver_alpha". This requires the solution of a minimization problem.
solver_alpha.parameters.update(solver_minimization_parameters)
# As the optimization is a constrained type we need to provide the corresponding lower and upper  bounds.
lb = interpolate(Expression("0.0",degree = 0),V_scalar)
ub = interpolate(Expression ("0.95", degree = 0),V_scalar)
lb.vector()[:] = alpha.vector()
# Crete the files to store the solution of damage and displacements.
file_alpha  =   File(savedir+"/alpha.pvd")
file_u      =   File(savedir+"/u.pvd")
#----------------------------------------------------------------   
# ALTERNATE MINIIZATION.
#----------------------------------------------------------------
# Initialization
iter = 1; err_alpha = 1; err_alpha_aux=1

a0 = Vector(MPI.COMM_SELF)
a1 = Vector(MPI.COMM_SELF)
# Iterations of the alternate minimization stop if an error limit is reached or a maximim number of iterations have been done.
while True and err_alpha > toll and iter < maxiter :
    alpha.vector().gather(a0, np.array(range(V_scalar.dim()), "intc"))
    amin=alpha.vector().min()
    amax=alpha.vector().max()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Job %d: Iteration:  %2d, a0:[%.8g,%.8g]"%(MPI.COMM_WORLD.Get_rank(),iter,a0.min(),a0.max()))
    # solve elastic problem
    solver_u.solve()             
    # solve damage problem via a constrained minimization algorithm.
    solver_alpha.solve(problem_alpha,alpha.vector(),lb.vector(),ub.vector())
    alpha.vector().get_local()[alpha.vector().get_local()>0.95]=0.95
    alpha.vector().gather(a1,np.array(range(V_scalar.dim()),"intc"))   
    # Compute the norm of the the error vector.
    err_alpha = np.linalg.norm(a1-a0,ord=np.Inf) 
    # Numerical Improve.
    if C_L != 0.0:
        while err_alpha > err_alpha_aux:
            alpha_2 = C_L*alpha_0+(1.0-C_L)*alpha
            alpha.assign(alpha_2)  
            alpha.vector().gather(a1,np.array(range(V_scalar.dim()),"intc"))            
            err_alpha = np.linalg.norm(a1-a0,ord=np.Inf)  
    # Monitor the results
    if MPI.COMM_WORLD.Get_rank() >= 0:
        print ("Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" % (iter,err_alpha,alpha.vector().max()))
    # Update the solution for the current alternate minimization iteration.
    err_alpha_aux   =   err_alpha
    alpha_0.assign(alpha)
    iter = iter + 1
# updating the lower bound with the solution of the solution corresponding to the current global iteration, it is for accounting for the irreversibility.
lb.vector ( ) [ :] = alpha.vector ( )
print ("---------------------------------------------------")
print (" End of the alternate minimization without cavity. ")
print ("----------------------------------------------------")
#----------------------------------------------------------------
#  END ALTERNATE MINIMIZATION.
#----------------------------------------------------------------
# Store u,alpha.
if MPI.COMM_WORLD.Get_rank()>=0:
    file_u      <<  (u, 0.)
    file_alpha  <<  (alpha,0.)               
# Store the damage for this geometry
alphaAux.assign(alpha)
alpha0.assign(alpha)
print ("-------------------------------------------------------------------")
print ("              Geometry without cavity is finished.                 ")
print ("-------------------------------------------------------------------")
# Remove previous integrating factors "dx, ds"
del ds, dx

#----------------------------------------------------------------
#  MESH WITH CAVITY.
#----------------------------------------------------------------
# Start loop over new geometries. These are obtained from a sequence of geometries which are obtained from an external folder. The number of external files is "NstepW" and the call is driven by the counter "itmesh".
#  Starting the loop of the mesh sequence. It is driven by the index "itmesh".
while itmesh <= NstepW :
    a0 = Vector(MPI.COMM_SELF)
    a1 = Vector(MPI.COMM_SELF)
    # Define Subdomain Cavity.
    class Structure(SubDomain):
        def inside(self, x,on_boundary):
            return between(x[0],(-500.0,-500+40*itmesh))andbetween(x[1] ,(-20.0,20.0))
    # Create sub domain markers and mark everaything as 0
    sub_domains = MeshFunction("size_t",mesh,mesh.topology().dim())
    sub_domains.set_all(0)
    # Mark structure domain as 1
    structure = Structure()
    structure.mark(sub_domains,1)
    # Extract sub meshes
    domain_new = SubMesh(mesh,sub_domains,0)
    tunel_new = SubMesh(mesh, sub_domains,1)
    # Define boundary sets for boundary conditions
    class Left_new(SubDomain):
        def inside(self,x,on_boundary):
            return near(x[0],-1500.)   
    class Right_new(SubDomain):
        def inside(self,x,on_boundary):
            return near(x[0],1500.)
    class Top_new(SubDomain):
        def inside(self,x,on_boundary):
            return near(x[1],500.)   
    class Bottom_new(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1],-500.)
    # Initialize sub-domain instances
    left_new    =   Left_new() 
    right_new   =   Right_new()
    top_new     =   Top_new() 
    bottom_new  =   Bottom_new()    
    # define meshfunction to identify boundaries by numbers
    boundaries_new = MeshFunction("size_t",domain_new,domain_new.topology().dim()-1)
    boundaries_new.set_all(0)
    left_new.mark(boundaries_new,1) # mark left as 1
    right_new.mark(boundaries_new,2) # mark right as 2
    top_new.mark(boundaries_new,3) # mark top as 3
    bottom_new.mark(boundaries_new,4) # mark bottom as 4 
    # Define the new ds and dx.
    dsN = Measure('ds', domain=domain_new, subdomain_data=boundaries_new)
    dxN = Measure('dx', domain=domain_new)
    #normal vectors
    normal_v_new = FacetNormal ( domain_new)
    # Create new function space for elasticity + Damage
    V_vector_new   =    VectorFunctionSpace(domain_new,"CG",1)
    V_scalar_new   =    FunctionSpace( domain_new,"CG",1)
    V_tensor_new   =    TensorFunctionSpace( domain_new,"DG",0)
    #----------------------------------------------------------------
    # REMARK: To generate a sequence of plots in paraview the name
    # of the variable must be the same. It is achieved by including 
    # name="alpha" at the moment of the definition of the structure "alpha".
    #
    #  << alphaN =  Function ( V_scalar_new, name="alpha") >>
    #
    # The same definition needs to be done for displacement "u" and
    # other arrays as the difference of damage without cavity and 
    # damage with cavity, for example "alphaDiff".
    #----------------------------------------------------------------
    # Define the function, test and trial fields
    uN          =   Function(V_vector_new,name="u")
    duN         =   TrialFunction(V_vector_new)
    vN          =   TestFunction(V_vector_new)
    alphaN      =   Function(V_scalar_new,name="alpha")
    alphaN_2    =   Function ( V_scalar_new,name="alpha")                        
    dalphaN     =   TrialFunction ( V_scalar_new)
    betaN       =   TestFunction(V_scalar_new)                                                        
    # Project the rerence solution into the new mesh.
    alphaN_0    =   Function(V_scalar_new,name="alpha")
    # Define the initial damage for the new mesh.
    alphaN_0    =   interpolate(alphaAux,V_scalar_new)
    # Boudary conditions. 
    zero_v_new      =   Constant((0.,)*ndim)
    u_0_new         =   zero_v_new
    bc_boxbottomN   =   DirichletBC(V_vector_new,u_0_new,boundaries_new,4)
    bc_leftN        =   DirichletBC(V_vector_new.sub(0),0.0,boundaries_new,1)
    bc_rightN       =   DirichletBC(V_vector_new.sub(0),0.0,boundaries_new,2)
    bc_uN           =   [bc_boxbottomN,bc_leftN,bc_rightN]
    # Let us define the total energy of the system as the sum of elastic energy, dissipated energy due to the damage and external work due to body forces. 
    if case == "Marigo":
        elastic_energy1_new =   0.5*inner(sigma(eps(uN),alphaN),eps(uN))*dxN
        elastic_energy2_new =   0.5*inner(sigma(eps(uN),alphaN),eps(uN))*dxN
    if case=="DMSF":
        Id_Tensor           =   as_matrix([[1,0],[0,1]]) 
        elastic_energy1_new =   (0.5*((lmbda+mu)*(tr(eps(uN))**2)*inner(Id_Tensor,Id_Tensor)))*dxN+((A(alphaN)+k_ell)*mu*inner(Dev(eps(uN)),Dev(eps(uN))))*dxN
        elastic_energy2_new =   (0.5*((lmbda+mu)*(tr(eps(uN))**2)*inner(Id_Tensor,Id_Tensor)))*dxN+((A(alphaN)+k_ell)*mu*inner(Dev(eps(uN)),Dev(eps(uN))))*dxN      
    if case=="ShearComp":
        elastic_energy1_new =   0.5*inner(sigma(eps(uN),alphaN),eps(uN))*dxN
        elastic_energy2_new =   0.5/E*(inner(Dev(sigma(eps(uN),alphaN)),Dev( sigma(eps(uN),alphaN)))\
                                 - kappa*inner(Sph(sigma(eps(uN),alphaN)),Sph(sigma(eps(uN),alphaN))))*dxN  
    # Dissipated energy. 
    dissipated_energy_new = ( w ( alphaN) + ellv**2 *w1 * dot ( grad ( alphaN), grad ( alphaN) ) ) * dxN
    # External work.
    external_work_new     = dot ( body_force, uN) * dxN
    # Definition of the total energy
    total_energy1_new = elastic_energy1_new + dissipated_energy_new-external_work_new 
    total_energy2_new = elastic_energy2_new + dissipated_energy_new-external_work_new 
    # Weak form of elasticity problem. This is the formal expression for the tangent problem which gives us the equilibrium equations.
    E_uN     =  derivative(total_energy1_new,uN,vN)
    E_alphaN =  derivative(total_energy2_new,alphaN,betaN)
    # Hessian matrix
    E_alpha_alphaN  =   derivative(E_alphaN,alphaN,dalphaN)
    # Writing tangent problems in term of test and trial functions for matrix assembly
    E_duN     = replace(E_uN,{uN:duN})
    E_dalphaN = replace(E_alphaN,{alphaN:dalphaN})
    # Once the tangent problems are formulated in terms of trial and text functions, we define the variatonal problems.
    # Variational problem for the displacement.
    problem_uN  =   LinearVariationalProblem(lhs(E_duN),rhs(E_duN),uN,bc_uN)
    # Define the classs Optimization Problem for then define the damage.
    # Variational problem for the damage (non-linear to use variational inequality solvers of petsc). 
    class DamageProblemN(OptimisationProblem) :
        def __init__ (self): 
            OptimisationProblem.__init__(self) 
        # Objective vector    
        def f(self,x):
            alphaN.vector()[:]=x
            return assemble(total_energy2_new)
        # Gradient of the objective function
        def F(self,b,x):
            alphaN.vector()[:]=x
            assemble(E_alphaN,tensor=b)                       
        # Hessian of the objective function    
        def J(self,A,x):
            alphaN.vector()[:]=x
            assemble(E_alpha_alphaN,tensor=A)         
    # Define the minimization problem using the class.    
    problem_alphaN = DamageProblemN ( )
    # Set up the solvers        
    solver_uN = LinearVariationalSolver ( problem_uN)
    solver_uN.parameters.update ( solver_LS_parameters)     
    solver_alphaN = PETScTAOSolver ( )
    solver_alphaN.parameters.update ( solver_minimization_parameters)      
    #  For the constraint minimization problem we require the lower and upper bound, "lbN" and "ubN".  They are initialized though interpolations.
    lbN = alphaN_0
    ubN = interpolate ( Expression ( "9.5", degree = 0), V_scalar_new) 
    #----------------------------------------------------------------   
    # ALTERNATE MINIIZATION.
    #----------------------------------------------------------------
    iter = 1; err_alphaN = 1; err_alpha_aux=1

    while err_alphaN > toll and iter < maxiter :
        alphaN.vector().gather(a0,np.array(range(V_scalar_new.dim()),"intc"))
        amin=alphaN.vector().min()
        amax=alphaN.vector().max()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Job %d: itmesh=%-2d, Iteration:  %2d   , a0:[%.8g,%.8g], alphaN:[%.8g,%.8g]" \
                    %(MPI.COMM_WORLD.Get_rank(),itmesh,iter,a0.min(),a0.max(),amin,amax))
        # solve elastic problem
        solver_uN.solve ( )
        # solve damage problem via a constrained minimization algorithm.
        solver_alphaN.solve ( problem_alphaN, alphaN.vector ( ), lbN.vector ( ), ubN.vector ( ) )
        alphaN.vector().get_local( )[alphaN.vector().get_local( ) > 0.95] = 0.95
        alphaN.vector().gather(a1, np.array(range(V_scalar_new.dim()), "intc"))
        # Compute the norm of the the error vector.
        err_alphaN = np.linalg.norm(a1 - a0, ord = np.Inf)
        if C_L != 0.0:
            while err_alphaN>err_alpha_aux:
                alphaN_2 = C_L*alphaN_0+(1.0-C_L)*alphaN
                alphaN.assign(alphaN_2)  
                alphaN.vector().gather(a1,np.array(range(V_scalar_new.dim()), "intc"))            
                err_alphaN = np.linalg.norm(a1-a0,ord=np.Inf)  
        # Monitor the results for the new mesh.
        if MPI.COMM_WORLD.Get_rank() >= 0:
            print("Job %d: itmesh=%-2d, Iteration:  %2d   , Error: %2.8g, alpha_max: %.8g" \
                    %(MPI.COMM_WORLD.Get_rank(),itmesh,iter,err_alphaN,alphaN.vector().max()))        
        # update the solution for the current alternate minimization iteration.
        err_alpha_aux   =   err_alphaN
        alphaN_0.assign(alphaN)
        iter = iter + 1                                                                                                              
    if MPI.COMM_WORLD.Get_rank() >= 0:
        print ("------------------------------------------------------------")
        print (" End of the alternate minimization in Remesh: %d " %(itmesh) )
        print ("------------------------------------------------------------")
    # Once a new damage has been obtained, we store it into an auxiliary variable "alphaAux"
    alphaAux    =   Function(V_scalar_new,name="alpha")
    alphaAux.assign(alphaN)
    # Store u,alpha.
    if MPI.COMM_WORLD.Get_rank() >= 0:
        file_u         << ( uN, 1.0 * itmesh)
        file_alpha     << ( alphaN   , 1.0 * itmesh)
    itmesh = itmesh + 1
    # Free memory for lists depending on the current mesh iteration
    del duN
    del vN  
    del alphaN
    del dalphaN
    del betaN  
    del alphaN_0   
    del ubN
    del lbN
    del bc_uN
    del normal_v_new
    del boundaries_new
    del domain_new   
    del V_vector_new
    del V_scalar_new
    del total_energy1_new
    del elastic_energy1_new
    del total_energy2_new
    del elastic_energy2_new
    del external_work_new
    del dissipated_energy_new
    del E_uN
    del E_alphaN
    del E_alpha_alphaN
    del E_duN
    del E_dalphaN  
    del solver_uN
    del problem_uN
    del problem_alphaN
    del dsN, dxN
    del DamageProblemN
    del a0
    del a1
    del a1
#----------------------------------------------------------------
# THE MAIN LOOP FOR REMESHING GAS FINISHED.
#----------------------------------------------------------------
print("-------------------------------------------------------------------")
print("                Geometry with cavity is finished.                  ")
print("-------------------------------------------------------------------")
#----------------------------------------------------------------
# FREE MEMORY
#---------------------------------------------------------------- 
del u, du, v
del alpha, dalpha, beta, alpha_0
del alpha_error
del file_alpha
del file_u
del lb, ub
del solver_u
del solver_alpha
del alphaAux
del V_vector
del V_scalar
del normal_v
del mesh, boundaries
#----------------------------------------------------------------
#   End of the main program.
#----------------------------------------------------------------
