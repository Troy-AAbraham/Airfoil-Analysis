from airfoil_functions import *
from complex_airfoil_functions import *
import time
import json
import matplotlib.pyplot as plt

# generate array of desired node count
# node_range = np.hstack((np.arange(10,114,4), np.arange(100,1100,20)))
# node_range = np.hstack((np.arange(10,110,10), np.arange(100,1100,100)))

node_range = [200]
n = 1

n = len(node_range)
CL_comp = np.zeros(n)
Cm4_comp = np.zeros(n)

CL_vort = np.zeros(n)
Cm4_vort = np.zeros(n)

CL_diff = np.zeros(n)
Cm4_diff = np.zeros(n)

for i in range(n):

    node_num = node_range[i]
    comp_airfoil = complex_airfoil("complex_airfoil_specs")
    # comp_airfoil.plot()
    CL_comp[i] = comp_airfoil.CL
    Cm4_comp[i] = comp_airfoil.cm4
    
    comp_airfoil.write_geometry(node_num)

    airfoil = airfoil_potential("airfoil_input")
    airfoil.alpha_d = 0.0
    airfoil.run()
    CL_vort[i], _, Cm4_vort[i], _ = airfoil.solve_coefficients(0.0)
    
    CL_diff[i] = CL_vort[i] - CL_comp[i]
    Cm4_diff[i] = Cm4_vort[i] - Cm4_comp[i]
    
CL_error = abs((CL_diff)/CL_comp)*100
Cm4_error = abs((Cm4_diff)/Cm4_comp)*100

plt.figure(1)
plt.plot(node_range,CL_comp, linestyle = '--', color = 'r')
plt.scatter(node_range,CL_vort, color = 'b')
plt.xlabel('N-Nodes')
plt.ylabel('CL')
plt.xscale('log')
plt.xlim(8,1100)
plt.ylim(0.1,0.3)
plt.legend(['Conformal Mapping', 'Vortex Panel'])
plt.tight_layout()
plt.show()

plt.figure(2)
plt.plot(node_range,Cm4_comp, linestyle = '--', color = 'r')
plt.scatter(node_range,Cm4_vort, color = 'b')
plt.xlabel('N-Nodes')
plt.ylabel('Cm4')
plt.xscale('log')
plt.xlim(8,1100)
plt.ylim(-0.1,0.0)
plt.legend(['Conformal Mapping', 'Vortex Panel'])
plt.tight_layout()
plt.show()

plt.figure(3)
plt.plot(node_range,CL_diff, linestyle = '-', color = 'k')
plt.xlabel('N-Nodes')
plt.ylabel('Difference in CL')
plt.xscale('log')
plt.xlim(8,1100)
plt.ylim(-0.1,0.1)
plt.tight_layout()
plt.show()

plt.figure(4)
plt.plot(node_range,Cm4_diff, linestyle = '-', color = 'k')
plt.xlabel('N-Nodes')
plt.ylabel('Difference in Cm4')
plt.xscale('log')
plt.xlim(8,1100)
plt.ylim(-0.15,0.15)
plt.tight_layout()
plt.show()

plt.figure(5)
plt.scatter(node_range,CL_error, linestyle = '-', color = 'k')
plt.xlabel('N-Nodes')
plt.ylabel('CL % Error ')
plt.xscale('log')
plt.xlim(8,1100)
plt.ylim(0,50)
plt.tight_layout()
plt.show()

plt.figure(6)
plt.scatter(node_range,Cm4_error, linestyle = '-', color = 'k')
plt.xlabel('N-Nodes')
plt.ylabel('Cm4 % Error ')
plt.xscale('log')
plt.xlim(8,1100)
plt.ylim(0,50)
plt.tight_layout()
plt.show()