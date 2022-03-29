#----------------------------------------------------------------
# IMPORT ALL PARAMETERS FROM Parameters.py AND AuxFunctions.py.
#----------------------------------------------------------------
from parameters import *
from AuxFunctions import *
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
    if model==1:
        return w11*alpha
    if model==2:
        return w11*alpha**2
    if model==3:
        return w11*(1-(1-alpha)**2)
    if model==4:
        return w11*alpha 
def A(alpha):
    if model==1 or model ==2:
        return (1-alpha)**2
    if model==3:
        return (1-alpha)**4   
    if model==4:
        return (1-alpha)/(1+alpha)  
# Define th Deviator and Spheric Tensors
def Dev(Tensor):
    return Tensor-Sph(Tensor)
def Sph(Tensor):
    Tensor2=as_matrix([[1,0,0],[0,1,0],[0,0,1]])
    return (inner(Tensor,Tensor2)/inner(Tensor2,Tensor2))*Tensor2    
# Strain and stress in free damage regime.
def eps(v):
    return sym(grad(v))  
def sigma_0(eps):
    return 2.0*mu*(eps)+lmbda*tr(eps)*Identity(ndim)
# modification of the Young modulus. 
def sigma(eps,alpha):
    return (A(alpha)+k_ell)*sigma_0(eps)
#  Define the energies. 
def energy_w(u,alpha):
    Es=eps(u)-tr(eps(u))/3*Identity(ndim)
    Eploc=tr(eps(u))
    return ((lmbda/2+mu/3)*Eploc**2+mu*inner(Es,Es))
def energy_dev(u,alpha):
    Es=Dev(sigma(eps(u),alpha))
    return (0.5*inner(Es,eps(u)))
def energy_sph(u,alpha):
    Eploc=Sph(sigma(eps(u),alpha))
    return (0.5*inner(Eploc,eps(u)))
def energy_dis (alpha,ell,w1):
    w_a=w(alpha)
    grad_a=ell**2*w1*dot(grad(alpha),grad(alpha))
    return w_a+grad_a
def energy_p(u,body_force):
    return -inner(body_force,u)
def energy_E(u,alpha,body_force):
    return energy_dev(u,alpha)+\
           energy_sph(u,alpha)+\
           energy_p(u,body_force)
# Function for boundary condition
def k_r(u):
    k=1.0e9
    return k*u          
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
alpha0  =   Function(V_scalar,name="alpha")
dalpha  =   TrialFunction(V_scalar)
beta    =   TestFunction(V_scalar)
# Define energy functions.
W_energy    =   Function(V_scalar,name="energy_w")
dev_energy  =   Function(V_scalar,name="energy_dev")
div_energy  =   Function(V_scalar,name="energy_sph")
E_energy    =   Function(V_scalar,name="energy_E")
p_energy    =   Function(V_scalar,name="energy_p")
dis_energy  =   Function(V_scalar,name="energy_dis")
# Define stress and strain functions.
stressG =   Function(V_tensor,name="sigma")
strainG =   Function(V_tensor,name="epsilon")
# Define the alpha auxiliar function.
alphaAux    =   Function(V_scalar,name="alpha")
# Define the function "alpha_error" to measure relative error.
alpha_error =   Function(V_scalar)
# Interpolate the initial condition for the damage variable "alpha".
alpha_0 =   interpolate(Expression("0.",degree=2),V_scalar)
if FH=="Initial_Damage":
    a_1 =   Constant(900.0)
    a_2 =   Constant(900.0)
    a_3 =   Constant(121.0)
    for i in range(0,5):
        for j in range(0,5):
            alpha_0_aux=interpolate(Expression("(pow(cos(theta)*(x[0]-65*j)+sin(theta)*(x[2]-(30+30*i)),2)/a_1+pow((x[1]),2)/a_2+\
                                                    pow(-sin(theta)*(x[0]-65*j)+cos(theta)*(x[2]-(30+30*i)),2)/a_3)<=1?0.95:0.",
                                                    degree=2,a_1=a_1,a_2=a_2,a_3=a_3,theta=pi/4,j=j,i=i),V_scalar)
            alpha_0.vector()[:] =   alpha_0.vector()+alpha_0_aux.vector()
alpha.assign(alpha_0)
# Define ds and dx.
ds  =   Measure('ds',domain=mesh,subdomain_data=boundaries)
dx  =   Measure('dx',domain=mesh)
# Let us define the total energy of the system as the sum of elastic energy, dissipated energy due to the damage and external work due to body forces. 
# Elastic energy.
elastic_energy1 =   0.5*inner(sigma(eps(u),alpha),eps(u))*dx
elastic_energy2 =   0.5/E*(inner(Dev(sigma(eps(u),alpha)),Dev(sigma(eps(u),alpha)))\
                        -2.0/3.0*kappa2*inner(Sph(sigma(eps(u),alpha)),Sph(sigma(eps(u),alpha))))*dx
# External work.
external_work   =   dot(body_force,u)*dx
# Neumand BC.
external_bc =   0.5*((dot(k_r(u),normal_v)*dot(u,normal_v))*ds(BOXMIDX1) 
                +(dot(k_r(u),normal_v)*dot(u,normal_v))*ds(BOXMIDX2)
                +(dot(k_r(u),normal_v)*dot(u,normal_v))*ds(BOXMIDY1)
                +(dot(k_r(u),normal_v)*dot(u,normal_v))*ds(BOXMIDY2)
                -dot(g_bc_zz*normal_v,u)*ds(BOXMIDX1) 
                -dot(g_bc_zz*normal_v,u)*ds(BOXMIDX2) 
                -dot(g_bc_zz*normal_v,u)*ds(BOXMIDY1) 
                -dot(g_bc_zz*normal_v,u)*ds(BOXMIDY2))
# Dissipated energy.  
dissipated_energy   =   (w(alpha)+ell**2*w1*dot(grad(alpha),grad(alpha)))*dx
# Definition of the total energy
total_energy1   =   elastic_energy1+dissipated_energy-external_work+external_bc
total_energy2   =   elastic_energy2+dissipated_energy-external_work+external_bc
# Weak form of damage problem. This is the formal expression for the tangent problem which gives us the equilibrium equations.
E_u     =   derivative(total_energy1,u,v)
E_alpha =   derivative(total_energy2,alpha,beta)
# Hessian matrix
E_alpha_alpha   =   derivative(E_alpha,alpha,dalpha)
# Writing tangent problems in term of test and trial functions for matrix assembly.
E_du        =   replace(E_u,{u:du})
E_dalpha    =   replace(E_alpha,{alpha:dalpha})
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
        alpha.vector()[:]= x
        assemble(E_alpha_alpha,tensor=A)
# define the minimization problem using the class.
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
lb  =   interpolate(Expression("0.0",degree=0),V_scalar)
ub  =   interpolate(Expression("0.95",degree=0),V_scalar)
lb.vector()[:]  =   alpha.vector()
# Crete the files to store the solutions
file_alpha      =   File(savedir + "/alpha.pvd")
file_energW     =   File(savedir + "/energy_w.pvd")
file_energDev   =   File(savedir + "/energy_dev.pvd")
file_energDiv   =   File(savedir + "/energy_sph.pvd")
file_energDis	=   File(savedir + "/energy_dis.pvd")
file_energP	    =   File(savedir + "/energy_p.pvd")
file_energE	    =   File(savedir + "/energy_E.pvd")
file_u          =   File(savedir + "/u.pvd")
file_sigma      =   File(savedir+"/sigma.pvd") 
file_epsilon    =   File(savedir+"/epsilon.pvd") 
#----------------------------------------------------------------   
# ELASTICITY PROBLEM.
#---------------------------------------------------------------- 
if caso=="Elasticity":
        solver_u.solve()
        print ("-------------------------------------------------------------------")
        print ("    End of the Elasticity Problem in Remesh: %d    " %( 0 ))
        print ("-------------------------------------------------------------------")
#----------------------------------------------------------------   
# ALTERNATE MINIIZATION.
#----------------------------------------------------------------
# Initialization
if caso =="Damage":
    iter = 1; err_alpha = 1
    a0 = Vector(MPI.COMM_SELF)
    a1 = Vector(MPI.COMM_SELF)
    # Iterations of the alternate minimization stop if an error limit is reached or a maximim number of iterations have been done.
    while True and err_alpha > toll and iter < maxiter :
        alpha.vector().gather(a0,np.array(range(V_scalar.dim()),"intc"))
        if MPI.COMM_WORLD.Get_rank()==0:
            print("Job %d: Iteration:  %2d, a0:[%.8g,%.8g]"%(MPI.COMM_WORLD.Get_rank(),iter,a0.min(),a0.max()))
        # solve elastic problem
        solver_u.solve()
        # solve damage problem via a constrained minimization algorithm.
        solver_alpha.solve(problem_alpha,alpha.vector(),lb.vector(),ub.vector())
        alpha.vector().get_local()[alpha.vector().get_local()>0.95]=0.95
        alpha.vector().gather(a1,np.array(range(V_scalar.dim()),"intc"))              
        # Compute the norm of the the error vector.
        err_alpha = np.linalg.norm(a1 - a0, ord = np.Inf) #np.linalg.norm ( alpha_error.vector ( ).get_local ( ),ord = np.Inf)
        # Numerical Improve.
        if C_L != 0.0:
            while err_alpha>err_alpha_aux:
                alpha_2=C_L*alpha_0+(1.0-C_L)*alpha
                alpha.assign(alpha_2)  
                alpha.vector().gather(a1,np.array(range(V_scalar.dim()),"intc"))            
                err_alpha=np.linalg.norm(a1-a0,ord=np.Inf) 
        # monitor the results
        if MPI.COMM_WORLD.Get_rank()>=0:
            print ("Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" %(iter,err_alpha,alpha.vector().max()))
        # update the solution for the current alternate minimization iteration.
        err_alpha_aux   =   err_alpha
        alpha_0.assign(alpha)
        iter = iter + 1
    # updating the lower bound with the solution of the solution corresponding to the current global iteration, it is for accounting for the irreversibility.
    lb.vector()[:]=alpha.vector()
    print ("---------------------------------------------------")
    print (" End of the alternate minimization without cavity. ")
    print ("---------------------------------------------------")
#----------------------------------------------------------------
#  END ALTERNATE MINIMIZATION.
#----------------------------------------------------------------
# Store u,alpha.
if MPI.COMM_WORLD.Get_rank()>=0:
    file_u  <<  (u,0.)
    if caso=="Damage":
        file_alpha  <<  (alpha,0.)
# Store strain and stress.      
strain  =   eps(u)
stress  =   project(sigma(strain,alpha),V_tensor,solver_type="cg", preconditioner_type="petsc_amg")
stressG.assign(stress)
strainG.assign(project(strain,V_tensor,solver_type="cg", preconditioner_type="petsc_amg"))
if MPI.COMM_WORLD.Get_rank()>= 0:
    file_sigma      <<  (stressG,0.)
    file_epsilon    <<  (strainG,0.)
# Store energies.
W_energy.assign(project(energy_w(u,alpha),V_scalar,solver_type='cg',preconditioner_type="petsc_amg"))
dev_energy.assign(project(energy_dev(u,alpha),V_scalar,solver_type='cg',preconditioner_type="petsc_amg"))
div_energy.assign(project(energy_sph(u,alpha),V_scalar,solver_type='cg',preconditioner_type="petsc_amg"))
E_energy.assign(project(energy_E(u,alpha,body_force),V_scalar,solver_type='cg',preconditioner_type="petsc_amg"))
p_energy.assign(project(energy_p(u,body_force),V_scalar,solver_type='cg',preconditioner_type="petsc_amg"))
if caso=="Damage":
    dis_energy.assign(project(energy_dis(alpha,ell,w1),V_scalar,solver_type='cg',preconditioner_type="petsc_amg"))	
# Control if energies are too small
W_energy.vector().get_local()[W_energy.vector().get_local()<1e-12]=0.0
dev_energy.vector().get_local()[dev_energy.vector().get_local()<1e-12]=0.0
div_energy.vector().get_local()[div_energy.vector().get_local()<1e-12]=0.0
E_energy.vector().get_local()[E_energy.vector().get_local()<1e-12]=0.0
p_energy.vector().get_local()[p_energy.vector().get_local()<1e-12]=0.0
if caso=="Damage":
    dis_energy.vector().get_local()[dis_energy.vector().get_local()<1e-12]=0.0 
if MPI.COMM_WORLD.Get_rank()>=0:
    file_energW     <<  (W_energy,0.)
    file_energDev   <<  (dev_energy,0.)
    file_energDiv   <<  (div_energy,0.)
    file_energE     <<  (E_energy,0.)
    file_energP     <<  (p_energy,0.)
    if caso == "Damage":
        file_energDis   <<  (dis_energy,0.)          
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
    # Read mesh from a sequence of meshes generated externally.
    if malla=="v3":
        mesh_new=Mesh("Meshes/V3/"+meshname+str(itmesh)+".xml.gz")       
    if malla=="esmeralda":
        mesh_new=Mesh("Meshes/Esmeralda/"+meshname+str(itmesh)+".xml.gz") 
    if malla=="sub6":
        mesh_new=Mesh("Meshes/Sub6/"+meshname+str(itmesh)+".xml.gz")
    if malla=="nuevonivel":
        mesh_new=Mesh("Meshes/Nuevo_Nivel/"+meshname+str(itmesh)+".xml.gz") 
    # Read files for interpolation
    if malla=="v3":
        with open("Meshes/V3/"+meshname+str(itmesh)+"_"+str(0)+".txt","r") as file:
            GVertexI_2_GVertex0=eval(file.readline())
    if malla=="esmeralda":
        with open("Meshes/Esmeralda_2/"+meshname+str(itmesh)+"_"+str(0)+".txt","r") as file:
            GVertexI_2_GVertex0=eval(file.readline())            
    if malla=="sub6":
        with open("Meshes/Sub6/"+meshname+str(itmesh)+"_"+str(0)+".txt","r") as file:
            GVertexI_2_GVertex0=eval(file.readline())  
    if malla=="nuevonivel":
        with open("Meshes/Nuevo_Nivel/"+meshname+str(itmesh)+"_"+str(0)+".txt","r") as file:
            GVertexI_2_GVertex0=eval(file.readline()) 
    if malla=="v3":
        with open("Meshes/V3/"+meshname+str(0)+"_"+str(itmesh)+".txt","r") as file:
            GVertex0_2_GVertexI=eval(file.readline())   
    if malla=="esmeralda":
        with open("Meshes/Esmeralda_2/"+meshname+str(0)+"_"+str(itmesh)+".txt","r") as file:
            GVertex0_2_GVertexI=eval(file.readline()) 
    if malla=="sub6":
        with open("Meshes/Sub6/"+meshname+str(0)+"_"+str(itmesh)+".txt","r") as file:
            GVertex0_2_GVertexI=eval(file.readline()) 
    if malla=="nuevonivel":
        with open("Meshes/Nuevo_Nivel/"+meshname+str(0)+"_"+str(itmesh)+".txt","r") as file:
            GVertex0_2_GVertexI=eval(file.readline()) 
    mesh_new.init()
    # Read boundaries for new mesh              
    if malla=="v3":
        boundaries_new=MeshFunction('size_t',mesh_new,"Meshes/V3/"+meshname+str(itmesh)+"_faces.xml.gz")
    if malla=="esmeralda":
        boundaries_new=MeshFunction('size_t',mesh_new,"Meshes/Esmeralda_2/"+meshname+str(itmesh)+"_faces.xml.gz")
    if malla=="sub6":
        boundaries_new=MeshFunction('size_t', mesh_new,"Meshes/Sub6/"+meshname+str(itmesh)+"_faces.xml.gz")
    if malla=="nuevonivel":
        boundaries_new=MeshFunction('size_t',mesh_new,"Meshes/Nuevo_Nivel/"+meshname+str(itmesh)+"_faces.xml.gz")   
    MeshFunction("size_t", mesh, mesh.topology().dim()-1)     
    # Define the new ds and dx.
    dsN =   Measure('ds',domain=mesh_new,subdomain_data=boundaries_new)
    dxN =   Measure('dx',domain=mesh_new)
    #normal vectors
    normal_v_new = FacetNormal ( mesh_new)  
    # Create new function spaces
    V_vector_new    =   VectorFunctionSpace(mesh_new,"CG",1)
    V_scalar_new    =   FunctionSpace(mesh_new,"CG",1)
    V_tensor_new    =   TensorFunctionSpace(mesh_new,"DG",0)
    strainGN        =   Function(V_tensor_new,name="epsilon")
    stressGN        =   Function(V_tensor_new,name="sigma")   
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
    alphaN_2    =   Function(V_scalar_new,name="alpha")                        
    dalphaN     =   TrialFunction(V_scalar_new)
    betaN       =   TestFunction(V_scalar_new) 
    # Define energy functions.
    W_energyN    =  Function(V_scalar_new,name="energy_w")
    dev_energyN  =  Function(V_scalar_new,name="energy_dev")
    div_energyN  =  Function(V_scalar_new,name="energy_sph")
    E_energyN    =  Function(V_scalar_new,name="energy_E")
    p_energyN  	 =  Function(V_scalar_new,name="energy_p")
    dis_energyN  =  Function(V_scalar_new,name="energy_dis")
    # Define the initial damage for the new mesh.
    alphaN_0    =   interpolate(Expression("0.0",degree=1),V_scalar_new) 
    # Interpolate the previous damage.
    Interpola_In_Out(alphaAux,alphaN_0,GVertexI_2_GVertex0)    
    alphaN_0.vector().get_local()[alphaN_0.vector().get_local()<1e-12]=0.0
    alphaN_0.vector().get_local( )[alphaN_0.vector().get_local()>0.95]=0.95
    alphaN.assign(interpolate(alphaN_0,V_scalar_new))
    # Boudary conditions.
    faceCornerBottom.mark(boundaries_new, FACECORNERBOTTOM)
    bc_facecornerbottomN    =   DirichletBC(V_vector_new.sub(2),Constant(0.0),boundaries_new,FACECORNERBOTTOM)
    bc_boxmidx1N            =   DirichletBC(V_vector_new.sub(0),Constant(0.0),boundaries_new,BOXMIDX1)
    bc_boxmidx2N            =   DirichletBC(V_vector_new.sub(0),Constant(0.0),boundaries_new,BOXMIDX2)
    bc_boxmidy1N            =   DirichletBC(V_vector_new.sub(1),Constant(0.0),boundaries_new,BOXMIDY1)
    bc_boxmidy2N            =   DirichletBC(V_vector_new.sub(1),Constant(0.0),boundaries_new,BOXMIDY2)
    bc_boxbottomN           =   DirichletBC(V_vector_new.sub(2),Constant(0.0),boundaries_new,BOXBOTTOM)
    bc_uN                   =   [bc_boxbottomN,bc_facecornerbottomN]
    # Let us define the total energy of the system as the sum of elastic energy, dissipated energy due to the damage and external work due to body forces.
    elastic_energy1_new =   0.5*inner(sigma(eps(uN),alphaN),eps(uN))*dxN
    elastic_energy2_new =   0.5/E*(inner(Dev(sigma(eps(uN),alphaN)),Dev(sigma(eps(uN),alphaN)))\
                                -2.0/3.0*kappa2*inner(Sph(sigma(eps(uN),alphaN)),Sph(sigma(eps(uN),alphaN))))*dxN
    # External work.
    external_work_new   =   dot(body_force,uN)*dxN
    # Neuman BC.                  
    external_bc_new =   (0.5*((dot(k_r(uN),normal_v_new)*dot(uN,normal_v_new))*dsN(BOXMIDX1) 
                        +(dot(k_r(uN),normal_v_new)*dot(uN,normal_v_new))*dsN(BOXMIDX2)
                        +(dot(k_r(uN),normal_v_new)*dot(uN,normal_v_new))*dsN(BOXMIDY1)
                        +(dot(k_r(uN),normal_v_new)*dot(uN,normal_v_new))*dsN(BOXMIDY2))
                        -dot(g_bc_zz*normal_v_new,uN)*dsN(BOXMIDX1) 
                        -dot(g_bc_zz*normal_v_new,uN)*dsN(BOXMIDX2) 
                        -dot(g_bc_zz*normal_v_new,uN)*dsN(BOXMIDY1) 
                        -dot(g_bc_zz*normal_v_new,uN)*dsN(BOXMIDY2))
    # Dissipated energy.
    dissipated_energy_new   =   (w(alphaN)+ell**2*w1*dot(grad(alphaN),grad(alphaN)))*dxN
    # Definition of the total energy
    total_energy1_new   =   elastic_energy1_new+dissipated_energy_new-external_work_new+external_bc_new
    total_energy2_new   =   elastic_energy2_new+dissipated_energy_new-external_work_new+external_bc_new
    # Weak form of elasticity problem. This is the formal expression for the tangent problem which gives us the equilibrium equations.
    E_uN        =   derivative(total_energy1_new,uN,vN)
    E_alphaN    =   derivative(total_energy2_new,alphaN,betaN)
    # Hessian matrix
    E_alpha_alphaN  =   derivative(E_alphaN,alphaN,dalphaN)
    # Writing tangent problems in term of test and trial functions for matrix assembly
    E_duN       =   replace(E_uN,{uN:duN})
    E_dalphaN   =   replace(E_alphaN,{alphaN:dalphaN})
    # Once the tangent problems are formulated in terms of trial and text functions, we define the variatonal problems.
    # Variational problem for the displacement.
    problem_uN  =   LinearVariationalProblem(lhs(E_duN),rhs(E_duN),uN,bc_uN)
    # Define the classs Optimization Problem for then define the damage.
    # Variational problem for the damage (non-linear to use variational inequality solvers of petsc). 
    class DamageProblemN(OptimisationProblem):
        def __init__(self): 
            OptimisationProblem.__init__(self) 
        # Objective vector    
        def f(self,x):
            alphaN.vector()[:]=x
            return assemble(total_energy2_new)
        # Gradient of the objective function
        def F(self,b,x):
            alphaN.vector()[:]=x
            assemble(E_alphaN,tensor = b)                      
        # Hessian of the objective function    
        def J(self,A,x):
            alphaN.vector()[:]=x
            assemble(E_alpha_alphaN,tensor=A)
    # Define the minimization problem using the class.
    problem_alphaN  =   DamageProblemN()
    # Set up the solvers       
    solver_uN = LinearVariationalSolver(problem_uN)
    solver_uN.parameters.update(solver_LS_parameters)   
    solver_alphaN = PETScTAOSolver()
    solver_alphaN.parameters.update(solver_minimization_parameters)
    #  For the constraint minimization problem we require the lower and upper bound, "lbN" and "ubN".  They are initialized though interpolations.
    lbN = alphaN_0
    ubN = interpolate ( Expression ( "0.95", degree = 0), V_scalar_new) 
    #----------------------------------------------------------------   
    # ALTERNATE MINIIZATION.
    #----------------------------------------------------------------
    if caso == "Elasticity":
        solver_uN.solve()
        print ("-------------------------------------------------------------------")
        print ("    End of the Elasticity Problem in Remesh: %d    " %( itmesh ))
        print ("-------------------------------------------------------------------")     
    if caso == "Damage":    
        iter = 1; err_alphaN = 1; err_alpha_aux=1
        count_arreglo=0
        while err_alphaN>toll and iter<maxiter:
            alphaN.vector().gather(a0,np.array(range(V_scalar_new.dim()),"intc"))
            amin=alphaN.vector().min()
            amax=alphaN.vector().max()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("Job %d: itmesh=%-2d, Iteration:  %2d   , a0:[%.8g,%.8g], alphaN:[%.8g,%.8g]" \
                      %(MPI.COMM_WORLD.Get_rank(),itmesh,iter,a0.min(),a0.max(),amin,amax))            
            # solve elastic problem
            solver_uN.solve()
            # solve damage problem via a constrained minimization algorithm.
            solver_alphaN.solve(problem_alphaN,alphaN.vector(),lbN.vector(),ubN.vector())
            alphaN.vector().get_local()[alphaN.vector().get_local()>0.95]=0.95
            alphaN.vector().gather(a2,np.array(range(V_scalar_new.dim()),"intc"))
            # Compute the norm of the the error vector.
            err_alphaN = np.linalg.norm(a2 - a0, ord = np.Inf)
            if C_L != 0.0:
                while err_alphaN > err_alpha_aux:
                    alphaN_2 = C_L*alphaN_0+(1.0-C_L)*alphaN
                    alphaN.assign(alphaN_2)  
                    alphaN.vector().gather(a2, np.array(range(V_scalar_new.dim()), "intc"))            
                    err_alphaN = np.linalg.norm(a2 - a0, ord = np.Inf)
            # Monitor the results for the new mesh
            if MPI.COMM_WORLD.Get_rank() >= 0:
                print ("Job %d: itmesh=%-2d, Iteration:  %2d   , Error: %2.8g, alpha_max: %.8g" \
                        % ( MPI.COMM_WORLD.Get_rank(),itmesh, iter, err_alphaN, alphaN.vector ( ).max ( ) ))
            # update the solution for the current alternate minimization iteration.
            err_alpha_aux   =   err_alphaN
            alphaN_0.assign(alphaN)
            iter = iter + 1                                                                                  
        if MPI.COMM_WORLD.Get_rank() >= 0:
            print ("-------------------------------------------------------------------")
            print ("    End of the alternate minimization in Remesh: %d    " %( itmesh ))
            print ("-------------------------------------------------------------------")
    # Interpolation.
    Interpola_In_Out(alphaN, alphaAux, GVertex0_2_GVertexI) 
    # Store strain and stress.
    strainN =   eps(uN)
    stressN =   project(sigma(strainN,alpha),V_tensor_new,solver_type='cg',preconditioner_type="petsc_amg")
    stressGN.assign(stressN)
    strainGN.assign(project(strainN,V_tensor_new,solver_type='cg',preconditioner_type="petsc_amg"))
    if MPI.COMM_WORLD.Get_rank()>=0:
        file_u          <<  (uN,1.0*itmesh)
        file_sigma      <<  (stressGN,1.0*itmesh)
        file_epsilon    <<  (strainGN,1.0*itmesh)
        if caso=="Damage":
            file_alpha  <<  (alphaN,1.0*itmesh)
    # Eval the energy
    W_energyN.assign(project(energy_w(uN,alphaN),V_scalar_new,solver_type='cg',preconditioner_type="petsc_amg"))
    dev_energyN.assign(project(energy_dev(uN,alphaN),V_scalar_new,solver_type='cg',preconditioner_type="petsc_amg"))
    div_energyN.assign(project(energy_sph(uN,alphaN),V_scalar_new,solver_type='cg',preconditioner_type="petsc_amg"))
    E_energyN.assign(project(energy_E(uN,alphaN,body_force),V_scalar_new,solver_type='cg',preconditioner_type="petsc_amg"))
    p_energyN.assign(project(energy_p(uN,body_force),V_scalar_new,solver_type='cg',preconditioner_type="petsc_amg"))
    if caso=="Damage":
        dis_energyN.assign(project(energy_dis(alphaN,ell,w1),V_scalar_new,solver_type='cg',preconditioner_type="petsc_amg"))   
    # Control if energies are too small
    W_energyN.vector().get_local()[W_energyN.vector().get_local()<1e-12]=0.0
    dev_energyN.vector().get_local()[dev_energyN.vector().get_local()<1e-12]=0.0
    div_energyN.vector().get_local()[div_energyN.vector().get_local()<1e-12]=0.0
    E_energyN.vector().get_local()[E_energyN.vector().get_local()<1e-12]=0.0
    p_energyN.vector().get_local()[p_energyN.vector().get_local()<1e-12]=0.0
    if caso=="Damage":
        dis_energyN.vector().get_local()[dis_energyN.vector().get_local()<1e-12]=0.0
    # Store the energy
    if MPI.COMM_WORLD.Get_rank()>=0:
        file_energW     <<  (W_energyN,1.0*itmesh)
        file_energDev   <<  (dev_energyN,1.0*itmesh)
        file_energDiv   <<  (div_energyN,1.0*itmesh)
        file_energE     <<  (W_energyN,1.0*itmesh)
        file_energP     <<  (dev_energyN,1.0*itmesh)
        if caso == "Damage":
            file_energDis   <<  (dis_energyN,1.0*itmesh)    
    # Integral value of alpha
    int_alpha = assemble (alphaN*dxN)
    vol       = assemble (1.0*dxN)
    intalpha[itmesh] = ([1.0*itmesh, int_alpha, vol])
    np.savetxt(savedir + '/int_alpha.txt', intalpha)
    itmesh = itmesh + 1
    # Free memory for lists depending on the current mesh iteration
    del duN
    del vN 
    del alphaN
    del dalphaN
    del betaN 
    del alphaN_0
    del W_energyN
    del div_energyN
    del dev_energyN  
    del ubN
    del lbN
    del bc_uN
    del bc_boxmidx1N 
    del bc_boxmidx2N 
    del bc_boxmidy1N
    del bc_boxmidy2N
    del bc_boxbottomN
    del bc_alpha_upN
    del bc_alphaN
    del normal_v_new
    del boundaries_new
    del mesh_new   
    del V_vector_new
    del V_scalar_new
    del total_energy1_new
    del total_energy2_new
    del elastic_energy1_new
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
#----------------------------------------------------------------
# THE MAIN LOOP FOR REMESHING GAS FINISHED.
#----------------------------------------------------------------
print("-------------------------------------------------------------------")
print("                Geometry with cavity is finished.                  ")
print("-------------------------------------------------------------------")
# Plot integral of damage
def plot_int_alpha():
    p1, = plt.plot(intalpha[:,0], intalpha[:,1],'r-o',linewidth=2)
    plt.xlabel('Mesh Iterations')
    plt.ylabel('Integral of damage')
    plt.savefig(savedir + '/int_alpha.png')
    plt.grid(True)
plot_int_alpha()
#----------------------------------------------------------------
# FREE MEMORY
#---------------------------------------------------------------- 
del u, du, v
del alpha, dalpha, beta, alpha_0
del alpha_error
del W_energy, div_energy, dev_energy
del file_alpha
del file_u
del lb, ub
del solver_u
del solver_alpha
del alphaAux
del V_vector
del V_scalar
del bc_u
del bc_boxmidx1, bc_boxmidx2, bc_boxmidy1, bc_boxmidy2, bc_boxbottom
del normal_v
del mesh, boundaries
#----------------------------------------------------------------
#   End of the main program.
#----------------------------------------------------------------
