#----------------------------------------------------------------
# AUXILIARY FUNCTION FOR INTERPOLATION.
#----------------------------------------------------------------
from dolfin import dof_to_vertex_map,vertex_to_dof_map,Function,Vector
from numpy import array
from mpi4py import MPI
#----------------------------------------------------------------
# INTERPOLATION.
#----------------------------------------------------------------        
def Interpola_In_Out(Var_In,Var_Out,GVertex_Out_2_GVertex_In) :
    V_In        =   Var_In.function_space()
    mesh_In     =   Var_In.function_space().mesh()
    l2gvi_In    =   mesh_In.topology().global_indices(0)
    g2lvi_In    =   {}
    for j in range(mesh_In.num_vertices()):
        g2lvi_In[l2gvi_In[j]]=j 
    dof_2_vl_In             =   dof_to_vertex_map(V_In)
    vl_2_dof_In             =   vertex_to_dof_map(V_In)
    l2g_dof_vector_In       =   Function(V_In) 
    LDof_2_GVertex_arr_In   =   l2g_dof_vector_In.vector().get_local() 
    for i, v in enumerate(LDof_2_GVertex_arr_In):
        LDof_2_GVertex_arr_In[i]=l2gvi_In[dof_2_vl_In[i]]  
    l2g_dof_vector_In.vector().set_local(LDof_2_GVertex_arr_In)    
    Var_G_In                =   Vector(MPI.COMM_SELF)
    GDof_2_GVertex_Vself_In =   Vector(MPI.COMM_SELF)
    Var_In.vector().gather(Var_G_In,array(range(V_In.dim()),"intc"))
    l2g_dof_vector_In.vector().gather(GDof_2_GVertex_Vself_In,array(range(V_In.dim()),"intc"))
    Var_G_sorted_In =   [Var_G_In.get_local()[i[0]] for i in sorted(enumerate(GDof_2_GVertex_Vself_In.get_local()),key=lambda x:x[1])]
    V_Out           =   Var_Out.function_space()
    mesh_Out        =   Var_Out.function_space().mesh()
    l2gvi_Out       =   mesh_Out.topology().global_indices(0)
    g2lvi_Out       =   {}
    for j in range(mesh_Out.num_vertices()):
        g2lvi_Out[l2gvi_Out[j]]=j
    dof_2_vl_Out            =   dof_to_vertex_map(V_Out)
    vl_2_dof_Out            =   vertex_to_dof_map(V_Out)
    LDof_2_GVertex_Fn_Out   =   Function ( V_Out) 
    LDof_2_GVertex_arr_Out  =   LDof_2_GVertex_Fn_Out.vector().get_local() 
    for i, v in enumerate(LDof_2_GVertex_arr_Out):
        LDof_2_GVertex_arr_Out[i]=l2gvi_Out[dof_2_vl_Out[i]]
    LDof_2_GVertex_Fn_Out.vector().set_local(LDof_2_GVertex_arr_Out)        
    Var_G_Out                   =   Vector(MPI.COMM_SELF)
    GDof_2_GVertex_Vself_Out    =   Vector(MPI.COMM_SELF)
    Var_Out.vector().gather(Var_G_Out, array(range(V_Out.dim()),"intc"))
    LDof_2_GVertex_Fn_Out.vector().gather(GDof_2_GVertex_Vself_Out,array(range(V_Out.dim()),"intc"))
    Var_G_sorted_Out=[Var_G_Out.get_local()[i[0]] for i in sorted(enumerate(GDof_2_GVertex_Vself_Out.get_local()), key=lambda x:x[1])]
    for i in range(len(Var_G_sorted_In),len(Var_G_sorted_Out)) :
        Var_G_sorted_In.append(Var_G_sorted_Out[i])
    Var_array_Out   =   Var_Out.vector().get_local()
    rk              =   MPI.COMM_WORLD.Get_rank()
    for i,GV_Out in enumerate(LDof_2_GVertex_arr_Out) :
        try:      
            iGV_Out=int(GV_Out)     
            Var_array_Out[i]=Var_G_sorted_In[GVertex_Out_2_GVertex_In[iGV_Out]]
        except IOError:
            print ("iGV_Out=",iGV_Out)
    Var_Out.vector().set_local(Var_array_Out)    
    Var_Out.vector().apply("")     
