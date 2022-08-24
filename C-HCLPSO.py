# ***************************************************************************************#
""" Author:  Ph.D Thu Huynh Van, Assoc. Prof. Sawekchai Tangaramvong 
#   Emails:  thuxd11@gmail.com, Sawekchai.T@chula.ac.th
#            Applied Mechanics and Structures Research Unit, Department of Civil Engineering, 
#            Chulalongkorn University 
#   https://scholar.google.com/citations?user=NysMfoAAAAAJ&hl=vi 
# Research paper: Chaotic Heterogeneous Comprehensive Learning Particle Swarm Optimization 
# for Simultaneous Sizing and Shape Design of Structures (2022) "Applied Soft Computing Journal"
""" 
# Reference: Thu Huynh Van, Sawekchai Tangaramvong (2022). Two-Phase ESO-CLPSO Method for the Optimal Design 
# of Structures with Discrete Steel Sections. "Advances in Engineering Software". https://doi.org/10.1016/j.advengsoft.2022.103102
# CLPSO code: https://github.com/thuchula6792/CLPSO

import random
import math
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh
import matplotlib.pyplot as plt 
import pandas as pd
import sys
import scipy
import itertools
from operator import itemgetter
from time import process_time
from timeit import default_timer as timer  
from mpl_toolkits import mplot3d 
from traits . etsconfig . api import ETSConfig
ETSConfig.toolkit = 'qt4' # Force PyQt4 utilization for the GUI of MayaVi
from mayavi import mlab

# Function meshtruss create of a uniform truss structure
def meshtruss(p1, p2, nx, ny, nz):
    nodes=[]
    bars=[]
    # ****
    nodes.append([0, 0, 275.59])
    nodes.append([273.26, 0, 196.85])
    nodes.append([236.65, 136.63, 196.85])
    nodes.append([136.63, 236.65, 196.85])
    nodes.append([0, 273.26, 196.85])
    nodes.append([-136.63, 236.65, 196.85])
    nodes.append([-236.65, 136.63, 196.85])
    nodes.append([-273.26, 0, 196.85])
    # ****
    nodes.append([-236.65, -136.63, 196.85])
    nodes.append([-136.63, -236.65, 196.85])
    nodes.append([0, -273.26, 196.85])
    nodes.append([136.63, -236.65, 196.85])
    nodes.append([236.65, -136.63, 196.85])
    # ****
    nodes.append([492.12, 0, 118.11])
    nodes.append([475.351, 127.37, 118.11])
    nodes.append([426.118, 246.06, 118.11])
    nodes.append([347.981, 347.981, 118.11])
    nodes.append([246.06, 426.188, 118.11])
    nodes.append([127.37, 475.351, 118.11])
    nodes.append([0, 492.12, 118.11])
    # ****
    nodes.append([-127.37, 475.351, 118.11])
    nodes.append([-246.06, 426.188, 118.11])
    nodes.append([-347.981, 347.981, 118.11])
    nodes.append([-426.118, 246.06, 118.11])
    nodes.append([-475.351, 127.37, 118.11])
    nodes.append([-492.12, 0, 118.11])
    # ****
    nodes.append([-475.351, -127.37, 118.11])
    nodes.append([-426.118, -246.06, 118.11])
    nodes.append([-347.981, -347.981, 118.11])
    nodes.append([-246.06, -426.188, 118.11])
    nodes.append([-127.37, -475.351, 118.11])
    nodes.append([0, -492.12, 118.11])
    # ****
    nodes.append([127.37, -475.351, 118.11])
    nodes.append([246.06, -426.188, 118.11])
    nodes.append([347.981, -347.981, 118.11])
    nodes.append([426.118, -246.06, 118.11])
    nodes.append([475.351, -127.37, 118.11])
    
    nodes.append([625.59, 0, 0])
    nodes.append([541.777, 312.795, 0])
    nodes.append([312.795, 541.777, 0])
    nodes.append([0, 625.59, 0])
    nodes.append([-312.795, 541.777, 0])
    nodes.append([-541.777, 312.795, 0])
    nodes.append([-625.59, 0, 0])
    nodes.append([-541.777, -312.795, 0])
    nodes.append([-312.795, -541.777, 0])
    nodes.append([0, -625.59, 0])
    nodes.append([312.795, -541.777, 0])
    nodes.append([541.777, -312.795, 0])

    # Layer 0          
    bars.append([0,1]) # A1-1
    bars.append([0,2]) # A1-2
    bars.append([0,3]) # A1-3
    
    bars.append([0,4]) # A1-4
    bars.append([0,5]) # A1-5
    bars.append([0,6]) # A1-6
    
    bars.append([0,7]) # A1-7
    bars.append([0,8]) # A1-8
    bars.append([0,9]) # A1-9
    
    bars.append([0,10]) # A1-10
    bars.append([0,11]) # A1-11
    bars.append([0,12]) # A1-12
    
    bars.append([1,2]) # A2-13
    bars.append([2,3]) # A2-14
    bars.append([3,4]) # A2-15
    
    bars.append([4,5]) # A2-16
    bars.append([5,6]) # A2-17
    bars.append([6,7]) # A2-18
    
    bars.append([7,8]) # A2-19
    bars.append([8,9]) # A2-20
    bars.append([9,10]) # A2-21
    
    bars.append([10,11]) # A2-22
    bars.append([11,12]) # A2-23
    bars.append([12,1]) # A2-24
    
    # Layer 1
    bars.append([1,13]) # A3-25
    bars.append([2,15]) # A3-26
    bars.append([3,17]) # A3-27
    bars.append([4,19]) # A3-28
    bars.append([5,21]) # A3-29
    bars.append([6,23]) # A3-30
    bars.append([7,25]) # A3-31
    bars.append([8,27]) # A3-32
    bars.append([9,29]) # A3-33
    bars.append([10,31]) # A3-34
    bars.append([11,33]) # A3-35
    bars.append([12,35]) # A3-36
    
    # Layer 2
    bars.append([1,14]) # A4-37
    bars.append([2,14]) # A4-38
    bars.append([2,16]) # A4-39
    bars.append([3,16]) # A4-40
    bars.append([3,18]) # A4-41
    bars.append([4,18]) # A4-42
    bars.append([4,20]) # A4-43
    bars.append([5,20]) # A4-44
    bars.append([5,22]) # A4-45
    bars.append([6,22]) # A4-46
    bars.append([6,24]) # A4-47
    bars.append([7,24]) # A4-48
    
    bars.append([7,26]) # A4-49
    bars.append([8,26]) # A4-50
    bars.append([8,28]) # A4-51
    bars.append([9,28]) # A4-52
    bars.append([9,30]) # A4-53
    bars.append([10,30]) # A4-53
    bars.append([10,32]) # A4-55
    bars.append([11,32]) # A4-56
    bars.append([11,34]) # A4-57
    bars.append([12,34]) # A4-58
    bars.append([12,36]) # A4-59
    bars.append([1,36]) # A4-60
    
    # Layer 3
    bars.append([13,14]) # A5-61
    bars.append([14,15]) # A5-62
    bars.append([15,16]) # A5-63
    bars.append([16,17]) # A5-64
    bars.append([17,18]) # A5-65
    bars.append([18,19]) # A5-66
    bars.append([19,20]) # A5-67
    bars.append([20,21]) # A5-68
    bars.append([21,22]) # A5-69
    bars.append([22,23]) # A5-70
    bars.append([23,24]) # A5-71
    bars.append([24,25]) # A5-72
    
    bars.append([25,26]) # A5-73
    bars.append([26,27]) # A5-74
    bars.append([27,28]) # A5-75
    bars.append([28,29]) # A5-76
    bars.append([29,30]) # A5-77
    bars.append([30,31]) # A5-78
    bars.append([31,32]) # A5-79
    bars.append([32,33]) # A5-80
    bars.append([33,34]) # A5-81
    bars.append([34,35]) # A5-82
    bars.append([35,36]) # A5-83
    bars.append([36,13]) # A5-84
    
    # Layer 4
    bars.append([13,37]) # A6-85
    bars.append([15,38]) # A6-86
    bars.append([17,39]) # A6-87
    bars.append([19,40]) # A6-88
    bars.append([21,41]) # A6-89
    bars.append([23,42]) # A6-90
    bars.append([25,43]) # A6-91
    bars.append([27,44]) # A6-92
    bars.append([29,45]) # A6-93
    bars.append([31,46]) # A6-94
    bars.append([33,47]) # A6-95
    bars.append([35,48]) # A6-96
    
    # Layer 5
    bars.append([14,37]) # A7-97
    bars.append([14,38]) # A7-98
    bars.append([16,38]) # A7-99
    bars.append([16,39]) # A7-100
    bars.append([18,39]) # A7-101
    bars.append([18,40]) # A7-102
    bars.append([20,40]) # A7-103
    bars.append([20,41]) # A7-104
    bars.append([22,41]) # A7-105
    bars.append([22,42]) # A7-106
    bars.append([24,42]) # A7-107
    bars.append([24,43]) # A7-108
    
    bars.append([26,43]) # A7-109
    bars.append([26,44]) # A7-110
    bars.append([28,44]) # A7-111
    bars.append([28,45]) # A7-112
    bars.append([30,45]) # A7-113
    bars.append([30,46]) # A7-114
    bars.append([32,46]) # A7-115
    bars.append([32,47]) # A7-116
    bars.append([34,47]) # A7-117
    bars.append([34,48]) # A7-118
    bars.append([36,48]) # A7-119
    bars.append([36,37]) # A7-120
    
    thu=np.arange(0,121)
    gr1=thu[0:12]
    gr2=thu[12:24]
    gr3=thu[24:36]
    gr4=thu[36:60]
    gr5=thu[60:84]
    gr6=thu[84:96]
    gr7=thu[96:120]

    Groups=[]
    Groups.append(gr1)
    Groups.append(gr2)
    Groups.append(gr3)
    Groups.append(gr4)
    Groups.append(gr5)
    Groups.append(gr6)
    Groups.append(gr7)
    
    return np.array(nodes), np.array(bars), Groups

def remove_bar(connec, n1, n2):
    bars = connec.tolist()
    for bar in bars[:]:
        if (bar[0] == n1 and bar[1] == 2) or (bar[0] == 2 and bar[1] == n1):
            bars.remove(bar)
            return np.array(bars)
        else:
            print ('There is no such bar')
            return connec
        
def remove_node(connec, n1):
    bars = connec.tolist()
    for bar in bars[:]:
        if bar[0] == n1 or bar[1] == n1:
            bars.remove
    return np.array(bars)

# ASSEMBLED GLOBAL STIFFNESS MATRIX AND PERFORM OPTIMIZATION USING CHAOS-CLPSO
def opttruss(coord, connec, E, F, freenode, No_groups, Fy, C_c, Mater_Dens, plotdisp=False):
    n = connec.shape[0]  
    m = coord.shape[0]
    vectors = coord[connec[:,1],:] - coord[connec[:,0],:]
    l = np.sqrt((vectors**2).sum(axis=1))  
    e = vectors.T/l
    B = (e[np.newaxis] * e[:,np.newaxis]).T
    
    # Finite Element Analysis
    def fobj(A):
        A1 = [0]*n
        for i_2 in range(len(A)):
            for j_2 in No_groups[i_2]:
                A1[j_2] = A[i_2]
        A2 = np.copy(A1)
        Allowable_stress = np.copy(A1)
        stress_ratio = np.copy(A1)
        D = E * A2 / l
        radius = 0.4993*A2**0.6777
        lamda = 1*l/radius
        kx = e * D
        K = np.zeros((3*m, 3*m))
        for i in range(n):  
             aux = 3*connec[i,:]
             # The indies of DOEs
             index = np.r_[aux[0]:aux[0] + 3, aux[1]:aux[1] + 3]
             k0 = np.concatenate((np.concatenate((B[i], -B[i]), axis=1),np.concatenate((-B[i],B[i]), axis=1)), axis=0)
             K[np.ix_(index,index)] = K[np.ix_(index,index)] + D[i] * k0
        
        # Solve KU=F 
        block = freenode.flatten().nonzero()[0]
        matrix = K[np.ix_(block,block)]
        rhs = F.flatten()[block]
        solution = np.linalg.inv(matrix).dot(rhs)
        u = freenode.astype(float).flatten()
        u[block] = solution
        U = u.reshape(m,3)
        axial = ((U[connec[:,1],:]-U[connec[:,0],:])*kx.T).sum(axis=1)
        stress = axial / A2
        cost = (U*F).sum()
        dcost = -stress**2 / E *l
        volume = (A2 * l * Mater_Dens).sum()
        
        # Determine Maximum displacement
        Max_U = abs(U).max(axis=0)

        # Calculate the stress ratio for each element
        for k_k, j_j in enumerate(stress):
            if j_j >= 0:
                Allowable_stress[k_k] = 0.6*Fy
                stress_ratio[k_k] = abs(stress[k_k]/Allowable_stress[k_k])
            else:
                if lamda[k_k] < C_c:
                    Allowable_stress[k_k] = ((1-lamda[k_k]**2/(2*C_c**2))*Fy)/(5/3+3*lamda[k_k]/(8*C_c)-lamda[k_k]**3/(8*C_c**3))
                    stress_ratio[k_k] = abs(stress[k_k] / Allowable_stress[k_k])
                else:
                    Allowable_stress[k_k] = 12*math.pi**2*E[0]/(23*lamda[k_k]**2)
                    stress_ratio[k_k] = abs(stress[k_k] / Allowable_stress[k_k])
        Max_stress = np.max(abs(stress_ratio))        
        return cost, dcost, U, stress, Max_U, Max_stress, volume
    
        return cost, dcost, U, stress, Max_U, Max_stress, volume
    
    # Plot Results of Truss Structures
    def drawtruss(A, iteration, global_best_all_iteration, XPL, XPT, factor=70, wdt=1e-10):
 
         [U, stress, Max_U, Max_stress, volume] = fobj(A) [2:]
         A1 = [0]*n
         for i_2 in range(len(A)):
             for j_2 in No_groups[i_2]:
                 A1[j_2] = A[i_2]

         print('Displacement of each node: ', np.round(U,5))
         print('Stress Values of each element: ',np.round(stress,5))
         print('Max_Displacement: ', np.round(Max_U, 5))
         print('Max_Stress: ', np.round(Max_stress,5))
         print('Total Weight: ', np.round(volume,5))
         
         print('Exploration: ', XPL[-1], 'Exploitation: ', XPT[-1])
         plt.plot(iteration, XPL,'b', label = "Exploration (%)", linewidth=1)
         plt.plot(iteration, XPT,'g', label = "Exploitation (%)", linewidth=1)
         plt.xlabel('Iteration', fontsize=14)
         plt.ylabel('Percentage (%)', fontsize=14) 
         plt.title("C-HCLPSO", fontsize=14)   
         plt.xticks(np.arange(0, 1501, 300))
         plt.legend()
         plt.tick_params(labelsize=14)
         plt.grid(False)
         plt.legend(shadow=True)
         plt.savefig(r'C:\Users\ADMIN\Desktop\C-HCLPSO.png', dpi=800) 


         percent = np.row_stack((XPL, XPT))
         fig = plt.figure()
         bx = fig.add_subplot(111)
         col = ['skyblue', 'lightpink']
         bx.stackplot(iteration, percent, colors=col)
         bx.plot(iteration, XPL,'r', linewidth=1)
         bx.set_ylabel('Percentage (%)')
         bx.margins(0, 0)
         plt.show()
         
         fig = plt.figure()
         dx = fig.add_subplot(111)
         dx.plot(iteration, global_best_all_iteration,'r', linewidth=1)
         dx.set(xlabel='Numbers of Iteration')
         dx.grid(False)
         dx.set_title('Total Weight', fontsize=10)
         plt.show()

         # Export Data to excel file
         Final_result = {'Numbers of Iteration': iteration, 'Total Weight':  global_best_all_iteration, 
         'Area of each element':  A, 'Stress Values':  stress, 'Displacement_X': U[:,0], 'Displacement_Y': U[:,1]}
         df = pd.DataFrame.from_dict(Final_result, orient='index') 
         df1_transposed = df.T
         # df1_transposed.to_excel (r'C:\Users\ADMIN\Desktop\Data_120_Bars.xlsx', index = False, header = True)

    #--- MAIN       
    class Particle:
        def __init__(self):
            self.position_i = []          # particle position
            self.velocity_i = []          # particle velocity
            self.pos_best_i = []          # best position individual
            self.pos_best_i_record = []   # particles x dimensions NP*D
            self.err_best_i = 1000000     # best error individual
            self.err_best_i_record = []
            self.err_i = -1               # error individual
            self.U_CHECK = []
            self.stress_check = []
            for i in range(0,num_dimensions):
                 self.velocity_i.append(random.uniform(0,1)*(bounds[i][1]-bounds[i][0])+bounds[i][0])
                 self.position_i.append(random.uniform(0,1)*(bounds[i][1]-bounds[i][0])+bounds[i][0])
            self.pos_best_i_record.append(self.position_i)
        
        # Evaluate current fitness
        def evaluate(self,costFunc, Stress_Limit, Disp_Limit):
            
            # Perform FE Analysis
            self.U_CHECK, self.stress_check, self.err_i = costFunc(self.position_i)[4:]  
            
            # Check to see if the current position is an individual best
            if self.err_i < self.err_best_i:
                 # Check stress and displacement constraints 
                 if self.stress_check < Stress_Limit and abs(self.U_CHECK[0]) < Disp_Limit and abs(self.U_CHECK[1]) < Disp_Limit and abs(self.U_CHECK[2]) < Disp_Limit:
                     self.pos_best_i=self.position_i  # position
                     self.err_best_i=self.err_i       # fitness value
                 else:
                     self.err_i = self.err_i + 50000
                     self.pos_best_i=self.pos_best_i   # position
                     self.err_best_i=self.err_best_i   # fitness value
            else:
                 self.err_i = self.err_best_i

        # Update new particle velocity
        def update_velocity(self, pbest_f, pos_best_g, bounds, mdblI_1, c1, c2):
            # c1 = 3.5          # Cognative constant
            # c2 = 0.5          # Cocial constant
            for i in range(0,num_dimensions):
                 r1=random.random()
                 r2=random.random()
                 vel_cognitive = c1*r1*(pbest_f[i] - self.position_i[i])
                 vel_social = c2*r2*(pos_best_g[i]-self.position_i[i])
                 self.velocity_i[i] = mdblI_1*self.velocity_i[i] + vel_cognitive + vel_social
                 self.velocity_i[i] = np.where(self.velocity_i[i] >= 0.2*(bounds[i][1]-bounds[i][0]), 0.2*(bounds[i][1]-bounds[i][0]), self.velocity_i[i]).tolist()
        
        def Hetero_update_velocity(self, pbest_f, pos_best_g, bounds, mdblI_1, c1, c2):
            # Exploitation strategy--------- 
            for i in range(0,num_dimensions):
                 r1=random.random()
                 r2=random.random()
                 vel_cognitive = c1*r1*(pbest_f[i] - self.position_i[i])
                 vel_social = c2*r2*(pos_best_g[i]-self.position_i[i])
                 self.velocity_i[i] = mdblI_1*self.velocity_i[i] + vel_cognitive + vel_social
                 self.velocity_i[i] = np.where(self.velocity_i[i] >= 0.2*(bounds[i][1]-bounds[i][0]), 0.2*(bounds[i][1]-bounds[i][0]), self.velocity_i[i]).tolist()
        
        # Update the particle position based off new velocity updates
        def update_position(self, bounds):
            
            for i in range(0,num_dimensions):
                 self.position_i[i] = self.position_i[i] + self.velocity_i[i]

                 # Adjust maximum position if necessary
                 if self.position_i[i] > bounds[i][1]:
                     self.position_i[i] = bounds[i][1]

                 # Adjust minimum position if neseccary
                 if self.position_i[i] < bounds[i][0]:
                     self.position_i[i] = bounds[i][0]

    # PERFORM CHAOS-CLPSO*****************************************************
    class CHCLPSO():
        def __init__(self, costFunc, No_variable, bounds, num_particles, Stress_Limit, Disp_Limit):
            global num_dimensions
            num_dimensions = No_variable
            self.err_best_g = 1000000              # Best error for group
            self.pos_best_g = []                   # Best position for group
            self.pos_best_g_record = []
            self.pos_best_g_record_1 = []
            self.err_best_g_record = []
            self.global_best_all_iteration = []
            self.iteration = []
            self.f_pbest = []
            self.pbest_f = []               
            self.pbest_f_1 = []
            self.Pc = []
            self.fi1 = [0]*num_dimensions
            self.fi2 = [0]*num_dimensions
            self.fi = [0]*num_dimensions
            self.bi1 = 0
            self.bi = [0]*num_dimensions
            self.mintSinceLastChange = [0]*num_particles
            self.w_min = 0.2
            self.w_max = 0.9
            self.mintNuC = 5
            self.index_1 = 0
            self.c = [3.5, 1]
            self.c_cal = 0
            self.c1 = [2.5, 0.5]
            self.c2 = [2.5, 0.5]
            self.c1_cal = 0
            self.c1_cal = 0
            self.i_record = 0
            self.mdblI = [0.9]*num_particles
            self.Measurement = np.zeros((1, num_dimensions))
            self.Div = []  

            # SET UP INITIAL PAPRAMETERS
            t=np.linspace(0,5,num_particles)
            self.swarm = []
            for i in range(0,num_particles):
                
                 self.swarm.append(Particle())   
                 self.Pc.append(0 + 0.5*(np.exp(t[i]) - np.exp(t[0]))/(np.exp(5) - np.exp(t[0])))
                 self.f_pbest.append([i]*num_dimensions) 
 
            # The Initial Evaluation
            for k in range(0,num_particles):
                self.swarm[k].evaluate(costFunc, Stress_Limit, Disp_Limit)
                
                if self.swarm[k].err_i < self.err_best_g:   
                     self.pos_best_g = list(self.swarm[k].position_i)
                     self.err_best_g = float(self.swarm[k].err_i)
                self.pos_best_g_record.append(self.swarm[k].position_i)
                self.err_best_g_record.append(self.swarm[k].err_i)
                self.pbest_f.append(self.swarm[k].position_i)
            
            # Determine Exemplars
            for v in range(0, num_particles):
                for z in range(0,num_dimensions): 
                    
                     self.fi1[z] = math.ceil(random.uniform(0,1)*(num_particles-1))
                     self.fi2[z] = math.ceil(random.uniform(0,1)*(num_particles-1))
                     self.fi[z] = np.where(self.err_best_g_record[self.fi1[z]] < self.err_best_g_record[self.fi2[z]], self.fi1[z], self.fi2[z]).tolist()
                     self.bi1 = random.random() - 1 + self.Pc[v]
                     self.bi[z] = np.where(self.bi1 >= 0, 1, 0).tolist()
                     
                if np.sum(self.bi) == 0:
                     rc = round(random.uniform(0,1)*(num_dimensions-1))
                     self.bi[rc] = 1

                for m in range(0,num_dimensions):
                     self.f_pbest[v][m] = self.bi[m]*self.fi[m] + (1-self.bi[m])*self.f_pbest[v][m]

            self.pbest_f_1=np.copy(self.pos_best_g_record)
        
        # Adaptive inertia weight factor (AIWF)
        def aiwf(self, scores):
             """ the Adaptive inertia weight factor """
             mean_score = np.mean(scores)
             min_score = np.min(scores)

             return [self.new_w(mean_score, score, min_score) for score in scores]
        

        def new_w(self, mean_score, score, min_score):
            if score <= mean_score and min_score < mean_score:
                return self.w_min + (((self.w_max - self.w_min) * (score - min_score)) /
                    (mean_score - min_score))
            else:
                return self.w_max    
        
        # Evaluate current fitness
        def evaluate_2(self, chaotic_particle_2, err_best_i_2, Stress_Limit, Disp_Limit, costFunc):
            
            # Perform FE Analysis
            U_CHECK_2, stress_check_2, err_i_2 = costFunc(chaotic_particle_2)[4:]  
            
            # Check to see if the current position is an individual best
            if err_i_2 < err_best_i_2: 
                 # Check stress and displacement constraints 
                 if stress_check_2 < Stress_Limit and abs(U_CHECK_2[0]) < Disp_Limit and abs(U_CHECK_2[1]) < Disp_Limit and abs(U_CHECK_2[2]) < Disp_Limit:
                     err_best_i_2 = err_i_2
            else:  
                 err_best_i_2 = err_best_i_2
             
            return err_best_i_2
             
        # Chaotic Local Search by using Logistic Map              
        def Chaotic_Local_Search(self, particle, score, num_CLS, Stress_Limit, Disp_Limit, costFunc):
            
            # Apply the chaotic function to the particle.
            mins = [var[0] for var in bounds]
            maxs = [var[1] for var in bounds]
            particle_1 = particle
            
            for i in range(num_CLS):
               chaotic_particle = self.logistic_function(particle_1[:], mins, maxs)
               new_score = self.evaluate_2(chaotic_particle, score, Stress_Limit, Disp_Limit, costFunc)

               # Updated the new best solution by randomly using Logistic map
               if new_score < score:
                   return new_score, chaotic_particle
               particle_1 = chaotic_particle
            return score, particle
        
        def logistic_function(self, particle, mins, maxs):
            cxs = self.part_to_cx(particle, mins, maxs)
            logistic = [4 * cx * (1 - cx) for cx in cxs]

            return self.cx_to_part(logistic, mins, maxs)

        def part_to_cx(self, particle, lows, highs):
            return [(x - low) / (high - low)
                for x, (low, high) in zip(particle, zip(lows, highs))]

        def cx_to_part(self, cxs, lows, highs):
            return [low + cx * (high - low)
                for cx, (low, high) in zip(cxs, zip(lows, highs))]
        
        def decrease_search_space(self, particle, bounds, r):
            mins = [var[0] for var in bounds]
            maxs = [var[1] for var in bounds]

            xmins = [max(mins[i], particle[i] - (r * (maxs[i] - mins[i]))) for i in
                 range(num_dimensions)]
            xmaxs = [min(maxs[i], particle[i] + (r * (maxs[i] - mins[i]))) for i in
                 range(num_dimensions)]
            return [(xmins[k], xmaxs[k]) for k in range(num_dimensions)]

        # Start Iteration Process of HCLPSO*************************************
        def run(self, costFunc, maxiter):   
            i = 0
            while i < maxiter:
                self.iteration.append(self.i_record)
                self.pos_best_g_record_1 = np.copy(self.pos_best_g_record)

                # Updated Exemplars Through using Comprehensive learning strategy
                for j in range(0,num_particles):
                    if self.mintSinceLastChange[j] > self.mintNuC:
                        self.mintSinceLastChange[j] = 0

                        for k in range(0,num_dimensions):
                             self.fi1[k] = math.ceil(random.uniform(0,1)*(num_particles-1))
                             self.fi2[k] = math.ceil(random.uniform(0,1)*(num_particles-1))
                             self.fi[k] = np.where(self.err_best_g_record[self.fi1[k]] < self.err_best_g_record[self.fi2[k]], self.fi1[k], self.fi2[k]).tolist()
                             self.bi1 = random.uniform(0,1) - 1 + self.Pc[j]
                             self.bi[k] = np.where(self.bi1>=0, 1, 0).tolist()

                        if np.sum(self.bi) == 0:
                             rc = round(random.uniform(0,1)*(num_dimensions-1))
                             self.bi[rc] = 1

                        for m in range(0,num_dimensions):
                             self.f_pbest[j][m] = self.bi[m]*self.fi[m] + (1-self.bi[m])*self.f_pbest[j][m]      
            
                for j in range(0,num_particles):
                    for k in range(0,num_dimensions):
                         self.index_1 = self.f_pbest[j][k]
                         self.pbest_f_1[j,k] = self.pos_best_g_record_1[self.index_1, k] 
            
                self.pbest_f = self.pbest_f_1.tolist()

                # Update the velocity and position of all particles - cycle through swarm and update velocities and position
                self.mdblI = self.aiwf(self.err_best_g_record)     

                # Heterogeneous CLPSO with enhanced exploration and exploitation through update the velocity and position of all particles 
                for j in range(0, num_particles):

                    # The first subpopulation for exploration 
                    if j <= 10: 
                         self.c1_cal = self.c[0]-(self.c[0]-self.c[1]) * i / maxiter
                         self.c2_cal = 1
                         self.swarm[j].update_velocity(self.pbest_f[j], self.pos_best_g, bounds, self.mdblI[j], self.c1_cal, self.c2_cal)
                         self.swarm[j].update_position(bounds)

                    # The second subpopulation for exploitation
                    else:
                         self.c1_cal = self.c1[0] - (self.c1[0]-self.c1[1]) * i / maxiter
                         self.c2_cal = self.c2[0] - (self.c2[0]-self.c2[1]) * i / maxiter
                         self.swarm[j].Hetero_update_velocity(self.pbest_f[j], self.pos_best_g, bounds, self.mdblI[j], self.c1_cal, self.c2_cal)
                         self.swarm[j].update_position(bounds)
               
                # Cycle through particles in swarm and evaluate fitness
                for j in range(0,num_particles):
                    self.swarm[j].evaluate(costFunc, Stress_Limit ,Disp_Limit)

                    # Update the personal best position and fitness values for population
                    if self.swarm[j].err_i < self.err_best_g_record[j]:
                         self.pos_best_g_record[j] = list(self.swarm[j].position_i)
                         self.err_best_g_record[j] = float(self.swarm[j].err_i)
                    else:
                         self.mintSinceLastChange[j] += 1
                    # determine if current particle is the best (globally)
                    if self.swarm[j].err_i < self.err_best_g: 
                         self.pos_best_g=list(self.swarm[j].position_i)
                         self.err_best_g=float(self.swarm[j].err_i)
                
                # Measure the percentage of exploration and exploitation
                self.Measurement = np.zeros((1, num_dimensions))
                for z in range(0,num_particles):
                    self.Measurement +=  abs(self.pos_best_g_record_1.mean() - self.pos_best_g_record_1[z,:])

                self.Div.append(np.sum(self.Measurement/num_particles)/num_dimensions)
                
                self.global_best_all_iteration.append(self.err_best_g)
                self.i_record += 1
                i += 1

    #--- EXECUTE---------------------------------------------------------------
    No_variable = len(No_groups)
    bounds=[(0.775, 20)*1]*No_variable 
    num_particles = 20  
    Stress_Limit = 1.0
    Disp_Limit = 0.1969
    num_CLS = 5
    No_run = 0
    scale_bounds = 0.3
    t = CHCLPSO(fobj, No_variable, bounds, num_particles, Stress_Limit, Disp_Limit)
    
    while No_run < 15: 
        
       # Perform Comprehensive Learning-PSO
       t.run(fobj, maxiter = 100) 
       
       # Top is a list with tuples (score, particle)
       index, g_best_sort = zip(*sorted(enumerate(t.err_best_g_record), key=itemgetter(1)))
       thu = np.copy(t.pos_best_g_record)
       top = [thu[index[t]].tolist() for t in range(int(num_particles / 5))]
       g_best_sort_list = list(g_best_sort)
       
       # Perform Chaotic Local Search
       for l_1 in range(int(num_particles / 5)):
           g_best_sort_list[l_1], top[l_1] = t.Chaotic_Local_Search(top[l_1], g_best_sort_list[l_1], num_CLS, Stress_Limit, Disp_Limit, fobj)
           if g_best_sort_list[l_1] < t.err_best_g:
                 t.err_best_g = g_best_sort_list[l_1]
                 t.pos_best_g =  top[l_1]
                 
       # Decrease search space
       if No_run % 3 == 0 and No_run != 0:
            bounds = t.decrease_search_space(t.pos_best_g, bounds, scale_bounds)
       
       # Stored The best solution for the next run
       Iteration_out =  t.iteration
       pos_best_g_out = t.pos_best_g
       err_best_g_out = t.err_best_g
       global_best_all_iteration_out = t.global_best_all_iteration
       i_record_out = t.i_record
       Div_out = t.Div
       w_out = [t.mdblI[index[k]] for k in range(int(num_particles / 5))]
       
       # Perform CLPSO
       t = CHCLPSO(fobj, No_variable, bounds, num_particles, Stress_Limit, Disp_Limit)
       t.iteration = Iteration_out
       t.i_record = i_record_out
       t.global_best_all_iteration = global_best_all_iteration_out
       t.Div = Div_out
       
       if err_best_g_out < t.err_best_g:
             t.pos_best_g = pos_best_g_out
             t.err_best_g = err_best_g_out

       for z_1 in range(int(num_particles / 5)):
           if g_best_sort[z_1] < t.err_best_g_record[z_1]:
                 t.swarm[z_1].position_i = top[z_1]
                 t.pos_best_g_record[z_1] = top[z_1]
                 t.err_best_g_record[z_1] = g_best_sort[z_1]
                 t.mdblI[z_1] = w_out[z_1]
                 t.pbest_f[z_1] = top[z_1]
   
       No_run += 1
       print('*****Global optimal******: ', np.round(g_best_sort[0],3), ' ****** ',t.pos_best_g) 
       print('------Iteration-------{}: '.format(t.i_record), bounds)
    
    # Diversity measurement
    Div_final = np.asarray(t.Div)
    XPL = Div_final/np.max(Div_final)*100
    XPT = abs(Div_final-np.max(Div_final))/np.max(Div_final)*100
     
    # Print final results
    print('**************************FINAL**********************************:')
    print('Area of each element: ', t.pos_best_g)
    print('Lowest Weight: ',  t.err_best_g)
    
    # Draw truss after optimizing
    drawtruss(t.pos_best_g, t.iteration, t.global_best_all_iteration, XPL, XPT)
    
# *******************************************************************    
# Initial paramater of Truss-Layout Topology Optimization------------
start = timer()
coord, connec, No_groups = meshtruss((0,0,0), (120,240,120), 1, 4, 1)
E0 = 30450e+3
Fy = 58e+3
E = E0 * np.ones(connec.shape[0])
loads = np.zeros_like(coord)

loads[:,2] = -2248
loads[0,2] = -13490
loads[1:14,2] = -6744

free = np.ones_like(coord).astype('int')
free[48,:] = 0
free[47,:] = 0
free[46,:] = 0
free[45,:] = 0
free[44,:] = 0
free[43,:] = 0
free[42,:] = 0
free[41,:] = 0
free[40,:] = 0
free[39,:] = 0
free[38,:] = 0
free[37,:] = 0

C_0=(2*E0*math.pi**2/Fy)**0.5
opttruss(coord, connec, E, loads, free, No_groups, Fy, C_0, 0.288, True)
elapsed = process_time() 
print ("Running Time: ", timer()-start,'s') 

def drawtruss_3D(coord, connec):
         n = connec.shape[0]
         mlab.figure(size=(1000, 600), bgcolor = (1,1,1))  
         for i in range(n):
                 bar1 = np.concatenate((coord[connec[i,0],:][np.newaxis], coord[connec[i,1],:][np.newaxis]), axis=0)
                 mlab.plot3d(bar1[:,1], bar1[:,2], bar1[:,0], tube_radius=7, color=(0.8, 0.1, 0.5))   # color=(0.8, 0.1, 0.5) (0.4, 0.2, 0.8) (0.4, 0.3, 0.9) (0.5, 0.3, 0.9)
         
         mlab.points3d(coord[:, 1], coord[:, 2], coord[:, 0],  mode='sphere', color = (0, 0.4, 0.9) , scale_factor=30)
         mlab.orientation_axes()
         print ('MayaVi finished!!!')
node_coords, connectivity = meshtruss((0,0,0), (120,240,120), 1, 4, 1)[:2]
drawtruss_3D(node_coords, connectivity)
