from complex_airfoil_functions import *
import time


t1 = time.time()

airfoil = complex_airfoil("complex_airfoil_specs")
airfoil.plot()

t2 = time.time()
tf = t2 - t1
print('Total Runtime: ', tf)