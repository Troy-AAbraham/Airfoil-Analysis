import numpy as np
import matplotlib.pyplot as plt
import json

class airfoil_potential:
    
    def __init__(self, filename):
        
        '''Reads inputs from json file specified by filename'''
        
        json_string=open(filename + ".json").read()
        json_vals = json.loads(json_string)
        
        # airfoil geometry inputs
        self.NACA_number = json_vals["geometry"]["airfoil"]

        if self.NACA_number[:2] != 'UL':
            self.m = int(self.NACA_number[0])/100.
            self.p = int(self.NACA_number[1])/10.

        self.point_filename = json_vals["geometry"]["filename"]

        self.t = int(self.NACA_number[2:])
        self.max_t = self.t/100.

        self.CL_d = json_vals["geometry"]["CL_design"]
        self.naca4_TE = json_vals["geometry"]["trailing_edge"]
        self.n = json_vals["geometry"]["n_points"]
        self.alpha_zero = -2.077
        self.chord = 1.

        # operating inputs
        self.Vinf = json_vals["operating"]["freestream_velocity"]
        self.alpha_d = json_vals["operating"]["alpha[deg]"]
        
        # alpha sweep inputs
        self.alpha_start = json_vals["alpha_sweep"]["start[deg]"]
        self.alpha_end = json_vals["alpha_sweep"]["end[deg]"]
        self.alpha_increment = json_vals["alpha_sweep"]["increment[deg]"]

        # plotting inputs
        self.x_start = json_vals["plot_options"]["x_start"]
        self.x_lower_limit = json_vals["plot_options"]["x_lower_limit"]
        self.x_upper_limit = json_vals["plot_options"]["x_upper_limit"]
        self.delta_s = json_vals["plot_options"]["delta_s"]
        self.n_lines = json_vals["plot_options"]["n_lines"]
        self.delta_y = json_vals["plot_options"]["delta_y"]

        # run command inputs
        self.plot_streamlines = json_vals["run_commands"]["plot_streamlines"]
        self.plot_pressure = json_vals["run_commands"]["plot_pressure"]
        self.alpha_sweep = json_vals["run_commands"]["alpha_sweep"]
        self.export_geometry = json_vals["run_commands"]["export_geometry"]
        
    def run(self):
        
        '''Runs potential flow pressure and streamline functions for the user
        specified operating conditions'''

        try:
            # attempts to open filename of airfoil points, if exists,
            # generates geometry
            self.airfoil_points = np.genfromtxt(self.point_filename)
            self.NACA_number = 'Custom - ' + self.point_filename
            self.n = len(self.airfoil_points[:,0])
            self.xn, self.yn, self.xc, self.yc = self.geometry_nodes(self.n)
            print('Reading geometry from file...')
        except:
            # genereates airfoil geometry if no input file exists
            print('Generating geometry from inputs...')
            self.xn, self.yn, self.xc, self.yc, self.x_cam, self.y_cam = self.geometry_nodes(self.n)

        if self.export_geometry == True and hasattr(self, 'airfoil_points') == False:
            # exports geometry nodes to txt file
            print('Writing geometry to file...')
            np.savetxt('airfoil_' + self.NACA_number + '.txt', np.asarray([self.xn, self.yn]).T)

        # A matrix for gamma algorithm
        self.A = self.generate_A(self.xn, self.yn, self.xc, self.yc, self.n)
        
        if self.alpha_sweep == True:
            # creates range of alpha values for alpha sweep
            self.alpha_d_range = np.arange(self.alpha_start, self.alpha_end + self.alpha_increment, self.alpha_increment)
        else:
            # if no alpha sweep, creates list using input alpha
            self.alpha_d_range = [self.alpha_d]
            
        for i in range(len(self.alpha_d_range)):
            # runs alpha sweep
            self.alpha_d = self.alpha_d_range[i]
            self.alpha_rad = self.alpha_d*(np.pi/180.)
            # solves vortex strengths at current angle of attack
            self.gamma = self.solve_gamma(self.n, self.alpha_rad, self.A, self.len_panel, self.xn, self.yn)
            
            print('\nRunning Alpha = ', self.alpha_d)
            
            # result plotting conditions
            if self.plot_pressure == True:
                self.plot_pressures(i)
            if self.plot_streamlines == True:
                self.plot_airfoil(10 + i)

    def generate_P(self, len_p, x_1, x_2, y_1, y_2, xc, yc):
        
        '''Generates the influence matrix from panel at x_ and y_ on the point
        xc and yc'''
        
        xi = (1/len_p)*(((x_2-x_1)*(xc-x_1))+((y_2-y_1)*(yc-y_1))) #1.6.20
        eta = (1/len_p)*((-(y_2-y_1)*(xc-x_1))+((x_2-x_1)*(yc-y_1)))
        phi = np.arctan2((eta*len_p),((eta**2)+(xi**2)-(xi*len_p)))       #1.6.21
        psi = (0.5)*np.log(((xi**2)+(eta**2))/(((xi-len_p)**2)+(eta**2))) #1.6.22

        M11 = (x_2 - x_1)
        M12 = -(y_2 - y_1)
        M21 = (y_2 - y_1)
        M22 = (x_2 - x_1)
        K11 = (((len_p-xi)*phi) + (eta*psi))
        K12 = (xi*phi)-(eta*psi)
        K21 = (eta*phi)-((len_p-xi)*psi) - len_p
        K22 = (-eta*phi)-(xi*psi)+len_p
        
        # eq. 4.2.14
        p11 = (1/(2*np.pi*(len_p**2)))*((M11*K11)+(M12*K21))
        p12 = (1/(2*np.pi*(len_p**2)))*((M11*K12)+(M12*K22))      
        p21 = (1/(2*np.pi*(len_p**2)))*((M21*K11)+(M22*K21))
        p22 = (1/(2*np.pi*(len_p**2)))*((M21*K12)+(M22*K22))
        
        return p11, p12, p21, p22
        
    def generate_A(self, x_n, y_n, x_c, y_c, n):

        '''Build A matrix using Phillips algorithm  from 4.2.30-4.2.34'''
        
        #finds the length of each panel
        self.len_panel = np.zeros(n - 1)
        for i in range(0,(n-1)):
                self.len_panel[i] = np.sqrt(((x_n[i + 1] - x_n[i])**2)+((y_n[i + 1]-y_n[i])**2))  
        
        #initialize A matrix to zeros
        A = np.zeros((n,n));
        
        #nested loops build the panel coefficient matrix for each panel on the
        #airfoil. This starts with finding a xi and eta value 4.2.9, a phi 4.2.2,
        #a psi 4.2.3 and then finally the P matrix 4.2.14
        for i in range(0,(n-1)):
            for j in range(0,(n-1)):
                
                p11, p12, p21, p22 = self.generate_P(self.len_panel[j], x_n[j], x_n[j + 1], y_n[j], y_n[j + 1], x_c[i], y_c[i])

                #airfoil coefficient matrix is built using node locations and the
                #panel coefficient matrices found. 4.2.30-4.2.34
                A[i,j] = A[i,j] +(((x_n[i + 1]-x_n[i])/self.len_panel[i])*p21) -(((y_n[i + 1]-y_n[i])/self.len_panel[i])*p11);
                A[i,j + 1] = A[i,j + 1] +(((x_n[i + 1]-x_n[i])/self.len_panel[i])*p22)-(((y_n[i + 1]-y_n[i])/self.len_panel[i])*p12);      
        
        # kutta condition
        A[n - 1,0] = 1;
        A[n - 1,n - 1] = 1;
        
        return A

    def solve_gamma(self, n, alpha, A, len_p, x, y):
        
        ''' SOLVE FOR GAMMAS, VELOCITIES, CPs, CLs, CLs FROM THIN AIRFOIL THEORY '''

        RHS = np.zeros(n)

        for i in range(0,(n-1)):
            # build b matrix
            RHS[i] = self.Vinf*((y[i + 1] - y[i])*np.cos(alpha)-(x[i + 1] - x[i])*np.sin(alpha))/(len_p[i])
        # kutta condition
        RHS[n - 1] = 0
            
        # solve 4.2.35
        gamma = np.linalg.solve(A, RHS)

        return gamma
    
    def solve_coefficients(self, alpha):
        
        '''Uses equations 4.2.48 and 4.2.50 to solve for coefficient of lift
        and moment coefficient'''

        print('\nSolving coefficients from vortex panel solution...\n')
        
        CL_temp = 0.
        for i in range(0,(self.n-1)):
            # Phillips Eq. 4.2.49
            CL_temp = CL_temp + ((self.len_panel[i]/self.chord)*((self.gamma[i]+self.gamma[i + 1])/(self.Vinf)))
        CL= CL_temp
        
        Cm_temp = 0.
        for i in range(0,(self.n-1)):
            # Phillips Eq. 4.2.51
            Cm_temp = Cm_temp + (-1./3.)*(self.len_panel[i]/(self.chord*self.chord*self.Vinf))*((2*self.xn[i]*self.gamma[i] + self.xn[i]*self.gamma[i + 1] + self.xn[i + 1]*self.gamma[i] + 2*self.xn[i + 1]*self.gamma[i + 1])*np.cos(alpha) +
                                                                          (2*self.yn[i]*self.gamma[i] + self.yn[i]*self.gamma[i + 1] + self.yn[i + 1]*self.gamma[i] + 2*self.yn[i + 1]*self.gamma[i + 1])*np.sin(alpha))
        # store current moment coefficient
        Cm = Cm_temp
        # quarter chord moment coefficient
        Cm4 = Cm_temp + (1./4.)*CL_temp
        
        #coefficient of lift found from the thin airfoil theory
        CL_TAT = 2*np.pi*(alpha - self.alpha_zero)
        
        return CL, Cm, Cm4, CL_TAT
            
    def geometry_nodes(self, n):
        
        '''Generates airfoil node geometry using airfoil algorithms or from
        input file'''        
        
        if hasattr(self, 'airfoil_points') == False:

            '''Node geometry for the airfoil, using cosine clustering'''

            x_nodes = np.zeros(n)
            y_nodes = np.zeros(n)
    
            if n % 2 == 0:
                # even number of nodes
                d_theta = (np.pi)/((n/2) - 0.5) # 4.2.19
                x_cam = np.zeros(int(n/2))
                y_cam = np.zeros(int(n/2))
            else:
                # odd number of nodes
                d_theta = (np.pi)/((n/2)) # 4.2.16
                x_cam = np.zeros(int(n/2) + 1)
                y_cam = np.zeros(int(n/2) + 1)
    
            # odd number of nodes, force one at the origin
            if n % 2 != 0:
                mid_i = int((n-1)/2)
                # x_nodes[mid_i] = 0.
                x_nodes[mid_i] = 0.
                x_cam[mid_i] = 0.
    
            for i in range(1,int(n/2) + 1):
    
                # cosine clustering of x-locations
                if n % 2 == 0:
                    # even # of nodes
                    # index centered at the middle of array, starting at LE value
                    x_up = int((n/2) + i - 1)
                    x_lo = int((n/2) - i)
                    x_cos = 0.5*(1 - np.cos((i - 0.5)*d_theta))
                    x_cam[i-1] = x_cos
                    y_cam[i-1], y_nodes[x_up], y_nodes[x_lo], x_nodes[x_up], x_nodes[x_lo]  = self.naca4_geometry(x_cos)
                else:
                    # odd # of nodes
                    # index centered at the middle of array, starting at LE value
                    x_up = int(((n-1)/2) + i)
                    x_lo = int(((n-1)/2) - i)
                    x_cos = 0.5*(1 - np.cos((i)*d_theta))
                    x_cam[i] = x_cos
                    y_cam[i], y_nodes[x_up], y_nodes[x_lo], x_nodes[x_up], x_nodes[x_lo] = self.naca4_geometry(x_cos)

            #finds the center of each panel from the locations of the panels nodes
            x_c = np.zeros(n - 1)
            y_c = np.zeros(n - 1)
    
            for j in range(0,n-1):
                x_c[j] = ((x_nodes[j] + x_nodes[j + 1])/2.)
                y_c[j] = ((y_nodes[j] + y_nodes[j + 1])/2.)
            # returns camberline values
            return x_nodes, y_nodes, x_c, y_c, x_cam, y_cam

        else:

            '''Nodes from input file'''
            
            x_nodes = self.airfoil_points[:,0]
            y_nodes = self.airfoil_points[:,1]

            #finds the center of each panel from the locations of the panels nodes
            x_c = np.zeros(n - 1)
            y_c = np.zeros(n - 1)
    
            for j in range(0,n-1):
                x_c[j] = ((x_nodes[j] + x_nodes[j + 1])/2.)
                y_c[j] = ((y_nodes[j] + y_nodes[j + 1])/2.)
            # does not include camberline values
            return x_nodes, y_nodes, x_c, y_c
        
    def naca4_geometry(self, x):
        
        '''Generates upper, lower, and camber x and y points for a NACA 4 digit
        airfoil'''
        
        # camber line coordinates from either NACA1 or NACA4 series airfoils
        if self.NACA_number[:2] == 'UL':
            y_c, y_c_deriv = self.naca1_camberline(x)
        else:
            y_c, y_c_deriv = self.naca4_camberline(x)
        
        # airfoil thickness values using open or close trailing edge
        if self.naca4_TE == 'open':
            y_t = self.open_naca4_thickness(x)
        elif self.naca4_TE == 'closed':
            y_t = self.closed_naca4_thickness(x)
        
        theta = np.arctan(y_c_deriv)

        # offset x and y to adjust for thickness and camber
        x_upper = x - y_t*np.sin(theta)
        y_upper = y_c + y_t*np.cos(theta)

        x_lower = x + y_t*np.sin(theta)
        y_lower = y_c - y_t*np.cos(theta)
        
        # redimenionalize the x and y surface values
        x_lower = x_lower*self.chord
        y_lower = y_lower*self.chord
        
        return y_c, y_upper, y_lower, x_upper, x_lower

    def naca4_camberline(self, x):
        
        ''''generates camber line for a NACA4 series airfoil'''
        
        if x <= self.p:
            if self.p != 0:
                y_c = (self.m/(self.p**2))*((2*self.p*(x))-((x)**2))
                y_c_deriv = ((2*self.m)/(self.p**2))*(self.p-(x))
            else:
                y_c = 0.0
                y_c_deriv = 0.0
        elif x > self.p :
            y_c = (self.m/((1-self.p)**2))*((1-2*self.p)+(2*self.p*(x))-((x)**2))
            y_c_deriv = ((2*self.m)/((1-self.p)**2))*(self.p-(x))
            
        return y_c, y_c_deriv

    def naca1_camberline(self, x):

        ''''generates camber line for a NACA1 series uniform load airfoil'''
        
        if x <= 0.0:
            # zero thickness at the LE will take care of the surface values there..
            y_c = 0.0
            y_c_deriv = 0.0
        elif x >= 1.0:
            # estimate derivative using log of nearly zero value at TE
            y_c = 0.0
            y_c_deriv = (self.CL_d/(4*np.pi))*(np.log(1e-15))
        else:
            y_c = (self.CL_d/(4*np.pi))*((x - 1)*np.log(1 - x) - x*np.log(x))
            y_c_deriv = (self.CL_d/(4*np.pi))*(np.log((1-x)/x))
            
        return y_c, y_c_deriv

    def open_naca4_thickness(self, x):
        
        '''Generates airfoil thickness at a cordwise location using open
        trailing edge thickness equation'''

        y_t = 5*self.max_t*(0.2969*np.sqrt(x) - 0.1260*(x) - 0.3516*((x)**2) + 0.2843*((x)**3) - 0.1015*((x)**4))

        return y_t

    def closed_naca4_thickness(self, x):

        '''Generates airfoil thickness at a cordwise location using closed
        trailing edge thickness equation. From Hunsaker, Reid paper'''

        y_t = 5*self.max_t*(0.2980*np.sqrt(x) - 0.132*(x) - 0.3286*((x)**2) + 0.2441*((x)**3) - 0.0815*((x)**4))

        return y_t

    def surface_normal(self, x):
        
        '''Returns normal vectors at each point along the upper and lower 
        surface of geometry.'''
        
        # uses very small step size in x to avoid some numerical issues at the
        # LE and TE
        dx = 0.00000001
        
        x1 = x - dx
        if x1 < 0.0:
            x1 = 0.
        x2 = x + dx
        if x2 < 0.0:
            x2 = dx
        cu1, yu1, yl1, xu1, xl1 = self.naca4_geometry(x1)
        cl2, yu2, yl2, xu2, xl2 = self.naca4_geometry(x2)
        
        dxu = xu2 - xu1
        dxl = xl2 - xl1
        dyu = yu2 - yu1
        dyl = yl2 - yl1
        magu = np.sqrt(dxu*dxu + dyu*dyu)
        magl = np.sqrt(dxl*dxl + dyl*dyl)
        
        # divide by magnitudes to find unit vectors
        upper_normal = np.array([-dyu/magu, dxu/magu]).T
        lower_normal = np.array([dyl/magl, -dxl/magl]).T
        
        return upper_normal, lower_normal

    def surface_tangent(self, x):

        '''Returns tangent vectors at each point along the upper and lower 
        surface of geometry.'''

        # uses very small step size in x to avoid some numerical issues at the
        # LE and TE
        dx = 0.00000001
        x1 = x - dx
        
        if x1 < 0.0:
            x1 = 0.
            
        x2 = x + dx
        _, yu1, yl1, xu1, xl1 = self.naca4_geometry(x1)
        _, yu2, yl2, xu2, xl2 = self.naca4_geometry(x2)
        
        dxu = xu2 - xu1
        dxl = xl2 - xl1
        dyu = yu2 - yu1
        dyl = yl2 - yl1
        magu = np.sqrt(dxu*dxu + dyu*dyu)
        magl = np.sqrt(dxl*dxl + dyl*dyl)
         
        # divide by magnitudes to find unit vectors
        upper_tangent = np.array([dxu/magu, dyu/magu]).T
        lower_tangent = np.array([dxl/magl, dyl/magl]).T 
        
        return upper_tangent, lower_tangent
    
    def velocity(self, x, y):
        
        '''Velocity around an airfoil using influence matrix and vortex strengths
         (gamma). Eq. 4.2.36'''
 
        sum_x = 0.0
        sum_y = 0.0
        
        for j in range(0,(self.n-1)):
            
            p11, p12, p21, p22 = self.generate_P(self.len_panel[j], self.xn[j], self.xn[j + 1], self.yn[j], self.yn[j + 1], x, y)
            sum_x += p11*self.gamma[j] + p12*self.gamma[j + 1]
            sum_y += p21*self.gamma[j] + p22*self.gamma[j + 1]

        Vx = self.Vinf*np.cos(self.alpha_rad) + sum_x
        Vy = self.Vinf*np.sin(self.alpha_rad) + sum_y
        
        # pressure coefficient at location
        Cp = 1 - (Vx*Vx + Vy*Vy)/self.Vinf

        return Vx, Vy, Cp
    
    def surface_tangential_velocity(self, x):
        
        '''
        Finds velocity at the upper and lower surface of the geometry. 
        Maintains the sign based on the direction of the Y component.
        '''
        ds_off = 1e-5
        camber, upper_y, lower_y, upper_x, lower_x = self.naca4_geometry(x)
        norm_u = self.surface_normal(x)[0]
        norm_l = self.surface_normal(x)[1]
        VxU, VyU, CpU = self.velocity(upper_x + ds_off*norm_u[0], upper_y + ds_off*norm_u[1])
        VxL, VyL, CpL = self.velocity(lower_x + ds_off*norm_l[0], lower_y + ds_off*norm_l[1])
        
        upper_tang = self.surface_tangent(x)[0]
        lower_tang = self.surface_tangent(x)[1]
    
        upper_tangential_velocity = np.dot([VxU, VyU], upper_tang)
        lower_tangential_velocity = np.dot([VxL, VyL], lower_tang)
    
        return upper_tangential_velocity, lower_tangential_velocity
    
    def solve_surface_pressure(self):

        '''solves surface pressure coefficient around the airfoil at the 
        control points'''

        # offset distance
        ds_off = 0.001
                
        if hasattr(self, 'airfoil_points') == False:
            # generates pressure sample points based on normal offset points
            # from the airfoil control points
            
            self.Cp = np.zeros(self.n - 1)
            self.x_Cp = np.zeros(self.n - 1)
            mid_j = int((self.n - 1)/2)

            for j in range(0, self.n-1):
                # offset point from surface in normal direction
                if j < mid_j:
                    x_offset = self.xc[j] + self.surface_normal(self.xc[j])[1][0]*ds_off
                    y_offset = self.yc[j] + self.surface_normal(self.xc[j])[1][1]*ds_off
                elif j >= mid_j:
                    x_offset = self.xc[j] + self.surface_normal(self.xc[j])[0][0]*ds_off
                    y_offset = self.yc[j] + self.surface_normal(self.xc[j])[0][1]*ds_off
    
                v = self.velocity(x_offset, y_offset)
                v_mag = np.sqrt(v[0]*v[0] + v[1]*v[1])
                # store surface pressures and locations
                self.Cp[j] = 1 - ((v_mag*v_mag)/(self.Vinf*self.Vinf))
                self.x_Cp[j] = x_offset
        else:
            # here the normals are found using the deltas between nodes rather
            # than small step sizes at the control points
            dx = np.diff(self.xn)
            dy = np.diff(self.yn)
            mag = np.sqrt(dx*dx + dy*dy)
            norms = np.asarray([-dy/mag, dx/mag]).T
            xp = self.xc + norms[:,0]*ds_off
            yp = self.yc + norms[:,1]*ds_off
            v = self.velocity(xp, yp)
            v_mag = np.sqrt(v[0]*v[0] + v[1]*v[1])
            # store surface pressures and locations
            self.Cp = 1 - ((v_mag*v_mag)/(self.Vinf*self.Vinf))
            self.x_Cp = xp
            
    def derivs(self, s, x_y):

        '''
        Derivative estimate using velocities along a streamline. Used in the
        RK4 methods.
        '''

        x,y = x_y
        
        Vx, Vy, Cp = self.velocity(x, y)
        V = np.sqrt(Vx*Vx + Vy*Vy)
    
        return np.array([(Vx/V), (Vy/V)])
    
    def rnkta4(self, n, s0, p0, ds, f):
    
        '''
        Parameters
        -----------
        s0: integer or float
            arc-length initial condition
        p0: array
            current position (x and y)
        ds: integer or float
            arc-length step size
        f: function
            derivative function of values being integrated forward
    
        Returns
        -------
        y: array
            updated position values
        '''
    
        # pre-allocate arrays
        k1i = np.zeros(n)
        k2i = np.zeros(n)
        k3i = np.zeros(n)
        k4i = np.zeros(n)
    
        # Runge Kutta 4 formulation
        k1i = f(s0, p0)
        pi = p0 + 0.5*k1i*ds
    
        k2i = f(s0 + 0.5*ds, pi)
        pi = p0 + 0.5*k2i*ds
    
        k3i = f(s0 + 0.5*ds, pi)
        pi = p0 + k3i*ds
    
        k4i = f(s0 + ds, pi)
    
        #updated state values
        p = p0 + (1/6)*(k1i + 2*k2i + 2*k3i + k4i)*ds
    
        return p
    
    def streamline(self, x0, y0, delta_s):
        
        '''Uses Runge-Kutta integration to trace the path of a streamline
        until a specified x-limit is reached'''
    
        s = 0.
        x_y_array = np.array([x0,y0])
    
        # initialize lists of streamline x and y values
        x_f = [x0]
        y_f = [y0]
    
        end_flag = False
    
        while end_flag == False:
    
            x_y_array = self.rnkta4(2, s, x_y_array, delta_s, self.derivs)
    
            if x_y_array[0] < self.x_lower_limit:
                end_flag = True
                break
            if x_y_array[0] > self.x_upper_limit:
                end_flag = True
                break
    
            s += delta_s
    
            x_f.append(x_y_array[0])
            y_f.append(x_y_array[1])
    
        stream_array = np.array([x_f,y_f]).T

        return stream_array

    def secant(self, x0, f, maxI, epsilon, loc, surface):
        
        '''
        Simple newtons method but including conditions to search top and bottom
        surface as well and front and aft sections of the geometry
        '''

        dh = 0.1
        xi = x0 + dh
        xip = x0

        # determines which velocity component to use
        if surface == 'top':
            v_index = 0
        elif surface == 'bottom':
            v_index = 1
            
        currentI = 0
        error = 100.
        
        while currentI < maxI and  error > epsilon:
                    
            xin = xi - (f(xi)[v_index]*(xip - xi))/(f(xip)[v_index] - f(xi)[v_index])

            if xin < 0.:
                xin = abs(xin)*0.002

            error = abs((xin - xi)/xin)*100.0

            currentI += 1
            xip = xi
            xi = xin
            # print('x0: ', xip, ', x1: ', xi, ', x2: ', xin)
            
        xf = xin
        print('Iterations to convergence: ', currentI)

        return xf
    
    def stagnation(self):
        
        thresh = 1e-8 #threshold for determining zero velocity
        
        # find LE and TE velocities
        if self.n % 2 == 0:
            x_start = 0.0
        else:
            x_start = 1e-8
            
        LE_V = self.surface_tangential_velocity(x_start)[0]
        # print(LE_V)
        # TE_V = self.surface_tangential_velocity(self.x_trailing_edge)[0]
        # print('LE V: ', abs(LE_V))
        # print('TE V: ', abs(TE_V))
        print('\nFinding leading edge stagnation point...')
        if (abs(LE_V) < thresh) or (LE_V == 0.0):
            # condition if front stagnation point is at the LE
            x_stag_front = 0.0
            y_stag_front = 0.0
            front_stag_norm = np.array([-1,0])
            print(self.surface_tangential_velocity(x_stag_front))
        elif LE_V < 0.:
            # if the LE tangential velocity is negative, stagnation point
            # is assumed to be on the top surface of geometry
            x_stag_temp = self.secant(x_start, self.surface_tangential_velocity, 1000, 0.0001, 'front', 'top')
            y_stag_front, _, x_stag_front, _ = self.naca4_geometry(x_stag_temp)[1:]
            front_stag_norm = self.surface_normal(x_stag_temp)[0]
            print('Stagnation point (x,y):' , x_stag_front, y_stag_front)
            print('Velocity at Stagnation (Vx, Vy, Cp): ', self.velocity(x_stag_front, y_stag_front))
            print('Tangential velocity (up, low): ', self.surface_tangential_velocity(x_stag_temp))

        elif LE_V > 0.:
            # if the LE tangential velocity is positive, stagnation point
            # is assumed to be on the lower surface of geometry
            x_stag_temp = self.secant(x_start, self.surface_tangential_velocity, 1000, 0.0001, 'front', 'bottom')
            y_stag_front, _, x_stag_front = self.naca4_geometry(x_stag_temp)[2:]
            front_stag_norm = self.surface_normal(x_stag_temp)[1]
            print('Stagnation point (x,y):' , x_stag_front, y_stag_front)
            print('Velocity at Stagnation (Vx, Vy, Cp): ', self.velocity(x_stag_front, y_stag_front))
            print('Tangential velocity (up, low): ', self.surface_tangential_velocity(x_stag_temp))
            
        # offset in normal direction the beginning of the stagnation streamlines 
        ds_off = 0.001
        x_s_f_start = x_stag_front + front_stag_norm[0]*ds_off
        y_s_f_start = y_stag_front + front_stag_norm[1]*ds_off
        # force trailing edge stagnation point
        x_s_a_start = self.chord + self.chord/100.
        y_s_a_start = 0.0
    
        return x_s_f_start, y_s_f_start, x_s_a_start, y_s_a_start

    def plot_pressures(self, fig_num):
        
        self.solve_surface_pressure()
        
        plt.figure(fig_num)
        plt.title('Surface Pressure Coefficient, Airfoil: ' + self.NACA_number +', TE: ' + self.naca4_TE + ', AoA [deg]: ' + str(self.alpha_d))
        plt.plot(self.x_Cp, self.Cp, color = 'k')
        plt.xlabel('x/c')
        plt.ylabel('$C_p$')
        plt.xlim(0, self.chord)
        plt.gca().invert_yaxis()
        plt.show()
        
        CL, CM, CM4, CL_TAT = self.solve_coefficients(self.alpha_rad)

        print('\nAirfoil :', self.NACA_number)
        print('CL:     ', '{:.16f}'.format(CL))
        print('Cm:     ', '{:.16f}'.format(CM))
        print('Cm,c/4: ', '{:.16f}'.format(CM4))
        
    def plot_airfoil(self, fig_num):

        if hasattr(self, 'airfoil_points') == False:
            # Find the front and aft stagnation points
            x_s_f, y_s_f, x_s_a, y_s_a = self.stagnation()
        else:
            # Find the front and aft stagnation points
            x_s_f, y_s_f, x_s_a, y_s_a = -0.001, 0.0, 1.001, 0.0

        # generate stagnation streamlines
        x_y_s_front = self.streamline(x_s_f, y_s_f, -self.delta_s)
        x_y_s_aft = self.streamline(x_s_a, y_s_a, self.delta_s)
        
        # use the y location of the front stagnation streamline at the boundary
        # to beginning spacing of other streamlines
        y_start = x_y_s_front[-1,1]

        plt.figure(fig_num)
        plt.axis('scaled')
        
        for i in range(self.n_lines):
            # generates specifed number of streamlines at equal spacing along the
            # front boundary
            y_new_u = y_start + (i + 1)* self.delta_y
            y_new_l = y_start - (i + 1)* self.delta_y 
            
            upper_streamline = self.streamline(self.x_start, y_new_u, self.delta_s)
            lower_streamline = self.streamline(self.x_start, y_new_l, self.delta_s)
            plt.plot(upper_streamline[:,0], upper_streamline[:,1], color = 'k')
            plt.plot(lower_streamline[:,0], lower_streamline[:,1], color = 'k')

        if hasattr(self, 'x_cam') == True:
            # plots camberline if airfoil was generated within call (not read in)
            plt.plot(self.x_cam, self.y_cam, color = 'r')

        plt.plot(self.xn, self.yn, color = 'b')
        
        try:
            # plots stagnation points if they were found
            plt.plot(x_y_s_front[:,0], x_y_s_front[:,1], color = 'k')
            plt.plot(x_y_s_aft[:,0], x_y_s_aft[:,1], color = 'k')
        except:
            pass

        plt.xlim(self.x_lower_limit, self.x_upper_limit)
        plt.ylim(-0.5, 0.5)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.title('Streamlines, Airfoil: ' + self.NACA_number +', TE: ' + self.naca4_TE + ', AoA [deg]: ' + str(self.alpha_d))
        plt.show()