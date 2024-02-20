from airfoil_functions import *
from complex_airfoil_functions import *
import time
import json
import matplotlib.pyplot as plt


airfoil = airfoil_potential("airfoil_roncz")

airfoil.run()

CL = airfoil.CL_range
CM = airfoil.CM_range
CM4 = airfoil.CM4_range