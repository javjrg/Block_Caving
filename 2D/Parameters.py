#----------------------------------------------------------------
#  IMPORT LIBRARIES TO GET THE CODE WORKING.
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
# MESH  
#----------------------------------------------------------------
mesh    =   RectangleMesh(Point(-1500.0,-500.0),Point(1500.0,500.0),750,250,"crossed")
#----------------------------------------------------------------
# PARAMETERS    
#----------------------------------------------------------------
itmesh  =   1
NstepW  =   15 
model   =   "quad"
w1      =   1.0e5
w11     =   1.0e5
C_L     =   0.
kappa   =   1.0
#Diferent Models to Damage problem, for example: "Marigo";"ShearComp"; "DMSF"
case    =   "Marigo" 
#----------------------------------------------------------------
# MATERIAL CONSTANTS
#----------------------------------------------------------------
E       =   29e9
nu      =   0.3
mu      =   E/(2.0*(1.0+ nu))
lmbda   =   E*nu/(1.0-nu**2)
# In this case this quantity contains the density, so gravity =  rho * g, with g the gravity acceleration.
rho     =   2.7e3
g       =   9.8
gravity =   rho*g
k_ell   =   Constant(1.e-6) #residual stiffness
# Definition of \ell parameter
h       =   CellDiameter(mesh) # diameters of all elements
hmin    =   mesh.hmin() # minimum of all diameters
hmax    =   mesh.hmax() # maximun of all diameters
ellv    =   5.0*hmin    #\ell parameter
#Body force
body_force = Constant( (0.0, -gravity) )
#----------------------------------------------------------------
# NUMERICAL PARAMETERS OF THE ALTERNATE MINIMIZATION.
#----------------------------------------------------------------
maxiter = 1e3
toll    = 1e-5
#----------------------------------------------------------------
# THE FILES ARE STORED IN A FOLDER NAMED "modelname".
#----------------------------------------------------------------
date        =   datetime.datetime.now().strftime("%m-%d-%y_%H.%M.%S")
where       =   socket.gethostname()
modelname   =   "[C_L=0]_[case=%s]_[model=%s]_[w1=%g]_[w11=%g]_[ell=%.2g]_[kappa=%.2g]_[Date=%s]_[Where=%s]"%(case,model,w1,w11,ellv,kappa,date,where) 
print('modelname='+modelname)
# others
regenerate_mesh =   True
savedir         =   "results/%s"%(modelname)
print('savedir='+savedir)
#----------------------------------------------------------------
# PARAMETERS IN THE GEOMETRY
#----------------------------------------------------------------
ndim = mesh.geometry().dim() #get number of space dimensions
# Define boundary sets for boundary conditions
class Left(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[0],-1500.)        
class Right(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[0],1500.)
class Top(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[1],500.)    
class Bottom(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[1],-500.)
# Initialize sub-domain instances
left    =   Left() 
right   =   Right()
top     =   Top() 
bottom  =   Bottom()
# define meshfunction to identify boundaries by numbers
boundaries = MeshFunction("size_t",mesh,mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark  (boundaries,1) # mark left as 1
right.mark (boundaries,2) # mark right as 2
top.mark   (boundaries,3) # mark top as 3
bottom.mark(boundaries,4) # mark bottom as 4
# normal vectors
normal_v    =   FacetNormal(mesh)
#----------------------------------------------------------------
# CREATE FUNCTION SPACE FOR 2D ELASTICITY AND DAMAGE
#----------------------------------------------------------------
V_vector    =   VectorFunctionSpace(mesh,"CG",1)
V_scalar    =   FunctionSpace(mesh,"CG",1)
V_tensor    =   TensorFunctionSpace(mesh,"DG",0)
#----------------------------------------------------------------
# BOUNDARY CONDITIONS.
#----------------------------------------------------------------
zero_v          =   Constant((0.,)*ndim)
u_0             =   zero_v
bc_left         =   DirichletBC(V_vector.sub(0),0.0,boundaries,1)
bc_right        =   DirichletBC(V_vector.sub(0),0.0,boundaries,2)
bc_boxbottom    =   DirichletBC(V_vector,u_0,boundaries,4)
bc_u            =   [bc_boxbottom,bc_left,bc_right]
