from airfoil_functions import *
import time

t1 = time.time()

'''This will run the airfoil specified by "airfoil" if there is no file that matches
"filename" provided in the airfoil_input.json file'''

airfoil = airfoil_potential("airfoil_input")
airfoil.run()

t2 = time.time()
tf = t2 - t1
print('Total Runtime: ', tf)