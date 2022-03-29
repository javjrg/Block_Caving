#----------------------------------------------------------------
# IMPORT LIBRARIES TO GET THE CODE WORKING.
#----------------------------------------------------------------
from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
import socket
import datetime
from ufl import replace
from mpi4py import MPI
from inspect import currentframe, getframeinfo,stack
#----------------------------------------------------------------
# PARAMETERS    
#----------------------------------------------------------------
itmesh  =   1
NstepW  =   40
Kw      =   Constant(100)
kappa  =   Constant(1.0)
# Diferents models for w(alpha)
model   =   1
w1      =   Constant(1.0e6)
w11     =   Constant(1000)
C_L     =   0.0 
# Diferent meshes to Damage problem, for example: "v3";"esmeralda"; "sub6"; "nuevonivel"
malla   =   "v3"
# Select cases: Damage or Elasticity
caso    =   "Damage"
# Consider Initial Damage for FH: Initial_Damage or No_Initial_Damage
FH      =   "NO_Initial_Damage" 
rk      =   MPI.COMM_WORLD.Get_rank()
#----------------------------------------------------------------
# MATERIAL CONSTANTS
#----------------------------------------------------------------
E           =   2.9e10
nu          =   0.3
mu          =   E/(2.0*(1.0+nu))
lmbda       =   E*nu/(1.0-nu**2)
ffD         =   lmbda/(lmbda+2*mu)
# In this case this quantity contains the density, so gravity =  rho * g, with g the gravity acceleration.
rho         =   2.7e3
g           =   9.8
gravity     =   rho*g
ell         =   1.0
k_ell       =   Constant(1.e-6) 
energies    =   np.zeros((NstepW,6))
#Body force
body_force  =   Constant((0.0,0.0,-gravity))
ndim        =   3 
#----------------------------------------------------------------
# NUMERICAL PARAMETERS OF THE ALTERNATE MINIMIZATION.
#----------------------------------------------------------------
maxiter =   1000    
toll    =   1e-5    
#----------------------------------------------------------------
# THE FILES ARE STORED IN A FOLDER NAMED "modelname".
#----------------------------------------------------------------
now =   datetime.datetime.now().strftime("%m-%d_%H.%M.%S")+'_'+socket.gethostname()
if FH=="Initial_Damage":
    if caso=="Elasticity":
        modelname="[FH][caso=%s]_[malla=%s]_[np=%d]_%s"%(caso,malla,MPI.COMM_WORLD.Get_size(),now)
    if caso=="Damage":
        modelname="[FH][caso=%s]_[malla=%s]_[model=%d]_[w11=%.0f]_[np=%d]_[k2=%.2f]_[C_L=%.2f]_%s"%(caso,malla,model,w11,MPI.COMM_WORLD.Get_size(),kappa,C_L,now)
else:
    if caso=="Elasticity":
        modelname="[caso=%s]_[malla=%s]_[np=%d]_%s"%(caso,malla,MPI.COMM_WORLD.Get_size(),now)
    if caso=="Damage":
        modelname="[caso=%s]_[malla=%s]_[model=%d]_[w11=%.0f]_[np=%d]_[k2=%.2f]_[C_L=%.2f]_%s"%(caso,malla,model,w11,MPI.COMM_WORLD.Get_size(),kappa,C_L,now)                          
if MPI.COMM_WORLD.Get_rank() == 0:
    print('modelname ='+modelname)
# others
regenerate_mesh =   True
savedir         =   "results/%s"%(modelname)
if MPI.COMM_WORLD.Get_rank() == 0:
    print('savedir='+savedir)
#----------------------------------------------------------------
# READ MESH AND BOUNDARIES FROM EXTERNAL FILES.
#----------------------------------------------------------------    
# In this block we define boundary sets for boundary conditions.
# This depends on particular tests. In this case the mesh is readed
# from external files.  
# boundaries labels. These are availables in the file "*_faces.xml.gz"
# Labels for cavities
CAVEUP      =   101
CAVEBOTTOM  =   102
CAVEMID     =   103
# Labels for external boundaries
BOXUP       =   201
BOXMIDX1    =   202
BOXMIDX2    =   203
BOXMIDY1    =   204
BOXMIDY2    =   205
BOXBOTTOM   =   206
# Meshname.
if malla=="v3":
    meshname="Socavacion_incrArea5000_maxArea200000_v3_" 
if malla=="esmeralda":
    meshname="esmeralda"
if malla=="sub6":
    meshname="sub6_"
if malla=="nuevonivel":
    meshname="nuevonivel_"
# Read mesh
if malla=="v3":
    mesh=Mesh("Meshes/V3/"+meshname+str(0)+".xml.gz")
if malla=="esmeralda":
    mesh=Mesh("Meshes/Esmeralda/"+meshname+str(0)+".xml.gz")
if malla=="sub6":
    mesh=Mesh("Meshes/Sub6/"+meshname+str(0)+".xml.gz")
if malla=="nuevonivel":
    mesh=Mesh("Meshes/Nuevo_Nivel/"+meshname+str(0)+".xml.gz")
mesh.init()
# Read boundaries
if malla=="v3":
    boundaries=MeshFunction('size_t',mesh,"Meshes/V3/"+meshname+str(0)+"_faces.xml.gz")
if malla=="esmeralda":
    boundaries=MeshFunction('size_t',mesh,"Meshes/Esmeralda/"+meshname+str(0)+"_faces.xml.gz")
if malla=="sub6":
    boundaries=MeshFunction('size_t',mesh,"Meshes/Sub6/"+meshname+str(0)+"_faces.xml.gz")
if malla=="nuevonivel":
    boundaries=MeshFunction('size_t',mesh,"Meshes/Nuevo_Nivel/"+meshname+str(0)+"_faces.xml.gz")
# Maximum values of mesh (local)
meshl_xmax,meshl_ymax,meshl_zmax = mesh.coordinates().max(axis=0)
# Maximum values of mesh (global)
mesh_xmax   =   MPI.COMM_WORLD.allreduce(meshl_xmax, op=MPI.MAX)
mesh_ymax   =   MPI.COMM_WORLD.allreduce(meshl_ymax, op=MPI.MAX)
mesh_zmax   =   MPI.COMM_WORLD.allreduce(meshl_zmax, op=MPI.MAX)
# Minimum values of mesh (local)
meshl_xmin,meshl_ymin,meshl_zmin = mesh.coordinates().min(axis=0)
# Minimum values of mesh (global)
mesh_xmin   =   MPI.COMM_WORLD.allreduce(meshl_xmin, op=MPI.MIN)
mesh_ymin   =   MPI.COMM_WORLD.allreduce(meshl_ymin, op=MPI.MIN)
mesh_zmin   =   MPI.COMM_WORLD.allreduce(meshl_zmin, op=MPI.MIN)
print ("ZMAX(G,L):",mesh_zmax,meshl_ymax)
print ("ZMIN(G,L):",mesh_zmin,meshl_ymin)
# normal vectors
normal_v    =   FacetNormal(mesh)
# Face Corner Bottom for the uniqueness
class FaceCornerBottom(SubDomain):
    def inside(self,x,on_boundary):
        px,py,pz=x
        return (on_boundary and near(pz,mesh_zmin)
                and (px<mesh_xmin+mesh.hmax())
                and (py<mesh_ymin+mesh.hmax())
                )
FACECORNERBOTTOM    =   400
faceCornerBottom    =   FaceCornerBottom()
faceCornerBottom.mark(boundaries, FACECORNERBOTTOM)
#----------------------------------------------------------------
# CREATE FUNCTION SPACE FOR 3D ELASTICITY AND DAMAGE
#----------------------------------------------------------------
V_vector    =   VectorFunctionSpace(mesh,"CG",1)
V_scalar    =   FunctionSpace(mesh,"CG",1)
V_tensor    =   TensorFunctionSpace(mesh,"DG",0)
#----------------------------------------------------------------
# BOUNDARY CONDITIONS.
#----------------------------------------------------------------
bc_facecornerbottom =   DirichletBC(V_vector.sub(2),Constant(0.0),boundaries,FACECORNERBOTTOM)
bc_boxmidx1         =   DirichletBC(V_vector.sub(1),Constant(0.0),boundaries,BOXMIDX1)
bc_boxmidx2         =   DirichletBC(V_vector.sub(1),Constant(0.0),boundaries,BOXMIDX2)
bc_boxmidy1         =   DirichletBC(V_vector.sub(0),Constant(0.0),boundaries,BOXMIDY1)
bc_boxmidy2         =   DirichletBC(V_vector.sub(0),Constant(0.0),boundaries,BOXMIDY2)
bc_boxbottom        =   DirichletBC(V_vector.sub(2),Constant(0.0),boundaries,BOXBOTTOM)
bc_u                =   [bc_boxbottom,bc_facecornerbottom]
# Newmann boundary condition
kx      =   gravity
g_bc_zz =   Expression('ffD*k*(x[2]-mesh_zmax)',degree=2,k=kx,mesh_zmax=mesh_zmax,ffD=ffD)
