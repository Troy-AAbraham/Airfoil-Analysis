import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

class complex_airfoil:
    
    def __init__(self,filename):
        
        '''Reads inputs from json file specified by filename'''
        
        json_string = open(filename + ".json").read()
        json_vals = json.loads(json_string)
        
        '''GEOMETRY'''
        self.geometry_type = json_vals["geometry"]["type"]
        self.radius = json_vals["geometry"]["cylinder_radius"]
        self.epsilon = json_vals["geometry"]["epsilon"]
        self.zeta_0 = complex(json_vals["geometry"]["zeta_0"][0],json_vals["geometry"]["zeta_0"][1])
        self.x0 = np.real(self.zeta_0)
        self.y0 = np.imag(self.zeta_0)
        self.CL_design = json_vals["geometry"]["design_CL"]
        self.thick_design = json_vals["geometry"]["design_thickness"]
        self.num_points = json_vals["geometry"]["output_points"]

        '''OPERATING'''
        self.alpha_d = json_vals["operating"]["angle_of_attack[deg]"]
        self.alpha_r = self.alpha_d*(np.pi/180.)
        self.Vinf = json_vals["operating"]["freestream_velocity"]
        self.vortex_str = json_vals["operating"]["vortex_strength"]
        self.save_geometry = json_vals["operating"]["write_geometry_file"]

        # Force calculation if an airfoil geometry is desired
        if self.geometry_type == 'airfoil':
            
            self.z0ratio = -((4*self.thick_design)/(3*np.sqrt(3))) +1j*(self.CL_design/(2*np.pi*(1 + 4*self.thick_design/(3*np.sqrt(3)))))
            self.xbar0 = np.real(self.z0ratio)
            self.ybar0 = np.imag(self.z0ratio)
            self.x0 = self.xbar0*self.radius
            self.y0 = self.ybar0*self.radius

            self.zeta_0 = complex(self.x0, self.y0)
            
            self.vortex_str = 4*np.pi*self.Vinf*(np.sqrt(self.radius*self.radius - self.y0*self.y0)*np.sin(self.alpha_r) + self.y0*np.cos(self.alpha_r))
            self.epsilon = self.radius - np.sqrt(self.radius*self.radius - self.y0*self.y0) - self.x0
            
            self.solve_coefficients()
        
        '''PLOTTING'''
        self.x_start = json_vals["plot"]["x_start"]
        self.x_lower_limit = json_vals["plot"]["x_lower_limit"]
        self.x_upper_limit = json_vals["plot"]["x_upper_limit"]
        self.delta_s = json_vals["plot"]["delta_s"]
        self.n_lines = json_vals["plot"]["n_lines"]
        self.delta_y = json_vals["plot"]["delta_y"]

        self.x_leading_edge = -np.sqrt(self.radius*self.radius - self.y0*self.y0)
        self.x_trailing_edge = np.sqrt(self.radius*self.radius - self.y0*self.y0)
        
        # writes the Joukowski to a file
        if self.save_geometry:
            self.write_geometry(self.num_points)
    
    def solve_coefficients(self):
        
        '''Solves for the lift and quarter chord moment coefficient of the Joukowski airfoil'''
        
        print('\nSolving coefficients for Joukowski airfoil...\n')
        
        cl_num = (np.sin(self.alpha_r) + (self.ybar0*np.cos(self.alpha_r))/(np.sqrt(1 - self.ybar0**2)))
        cl_denom = (1 + self.xbar0/(np.sqrt(1 - self.ybar0**2) - self.xbar0))
        
        self.CL = 2*np.pi*cl_num/cl_denom
        print('CL: ', self.CL)

        xl = -2*(self.radius**2 - self.y0**2 +  self.x0**2)/(np.sqrt(self.radius**2 - self.y0**2) -  self.x0)
        chord = 4*((self.radius*self.radius - self.y0*self.y0)/(np.sqrt(self.radius*self.radius - self.y0*self.y0) - self.x0))
        x4 = xl + chord/4
        xbar = x4/self.radius
        ybar = 0.0

        cm1 = (np.pi/4)*(((1 - self.ybar0**2 - self.xbar0**2)/(1 - self.ybar0**2))**2)*np.sin(2*self.alpha_r)
        cm2num = (xbar - self.xbar0)*np.cos(self.alpha_r) + (ybar - self.ybar0)*np.sin(self.alpha_r)
        cm2denom = (np.sqrt(1 - self.ybar0**2) - self.xbar0)/(1 - self.ybar0**2)
        self.cm4 = cm1 + 0.25*self.CL*cm2num*cm2denom
        print('Cm4: ', self.cm4)

    def write_geometry(self, num_points):
        
        '''Uses cosine clustering to generate a distribution of nodes along
        the Joukowski airfoil. Writes those nodes to a file that matches
        the formatting necessary for the sister vortex panel code'''
        
        n = num_points
        chord = 4*((self.radius*self.radius - self.y0*self.y0)/(np.sqrt(self.radius*self.radius - self.y0*self.y0) - self.x0))
        xl = -2*(self.radius**2 - self.y0**2 +  self.x0**2)/(np.sqrt(self.radius**2 - self.y0**2) -  self.x0)
        
        phi_T = np.arcsin(-self.y0/self.radius) # negative
        phi_L = np.pi - phi_T # positive
        phi_up_size = phi_L - phi_T # positive
        phi_lo_size = 2*np.pi - phi_up_size # positive
        dphi_up = phi_up_size/((n/2) - 0.5)
        dphi_lo = phi_lo_size/((n/2) - 0.5)

        upper_spacing = np.zeros(int(n/2))
        lower_spacing = np.zeros(int(n/2))
        
        for i in range(int(n/2)):
            upper_spacing[i] = phi_T + i*dphi_up
            lower_spacing[i] = phi_T - i*dphi_lo        
        
        z_upper = self.zeta2zee(upper_spacing)
        z_lower = self.zeta2zee(lower_spacing)
        y_upper = np.imag(z_upper)
        y_lower = np.imag(z_lower)
        x_upper = np.real(z_upper)
        x_lower = np.real(z_lower)
        
        # non-dimensional airfoil x aand y values
        x_lower = (x_lower - xl)/chord
        x_upper = (x_upper - xl)/chord
        y_upper = y_upper/chord
        y_lower = y_lower/chord
        
        x_airfoil = np.hstack((x_lower, np.flip(x_upper)))
        y_airfoil = np.hstack((y_lower, np.flip(y_upper)))
        
        # delete the shared nose node for an odd number of nodes
        if n % 2 != 0:
            x_airfoil = np.delete(x_airfoil,int((n-1)/2))
            y_airfoil = np.delete(y_airfoil,int((n-1)/2))
        
        print('\nWriting complex geometry to file...\n')
        np.savetxt('complex_airfoil.txt', np.asarray([x_airfoil, y_airfoil]).T)

    def geometry_zeta(self, x):

        '''Defines upper and lower surface geometry for a circle at a given
        x location in the zeta plane'''

        y_upper = np.sqrt((self.radius*self.radius) - x*x)
        y_lower = -np.sqrt((self.radius*self.radius) - x*x)
        
        # condition set to handle if an array of values is desired or a single point
        if isinstance(x, float):
            y_camber = np.zeros(1)
        else:
            y_camber = np.zeros(len(x))
    
        return y_camber + self.y0, y_upper + self.y0, y_lower + self.y0, x + self.x0

    def geometry(self, x):

        '''Defines upper and lower surface geometry for a Joukawski airfoil
        in the z plane'''
        
        theta_u = np.arccos(x/self.radius) # upper theta
        theta_l = -theta_u # lower theta
        
        z_upper = self.zeta2zee(theta_u)
        z_lower = self.zeta2zee(theta_l)
        y_upper = np.imag(z_upper)
        y_lower = np.imag(z_lower)
        x_upper = np.real(z_upper)
        x_lower = np.real(z_lower)
        
        # midpoint between upper and lower surface
        y_camber = (y_upper + y_lower)/2.
    
        return y_camber, y_upper, y_lower, x_upper, x_lower
    
    def zeta2zee(self, theta):

        '''Uses a theta in the complex zeta plane to find complex location
        in the z plane'''
        
        comp = 0. + 1j*theta
        z =  self.radius*np.exp(comp) + self.zeta_0 + (((self.radius - self.epsilon)**2)/(self.radius*np.exp(comp) + self.zeta_0))

        return z
    
    def zee2zeta(self, z):

        ''' Uses the algorithm presented by Phillips to convert a complex
        z location to a complex zeta location via conformal mapping'''

        '''There has got to be a better way to handle the single vs array issue....
        ......'''

        try:
            n = len(z)
            z1 = z*z - 4*(self.radius - self.epsilon)*(self.radius - self.epsilon)
            zeta_f = np.zeros(n)
            for i in range(n):
                if np.real(z1[i]) > 0.:
                    zeta = (z[i] + np.sqrt(z1[i]))/2.
                    zeta2 = (z[i] - np.sqrt(z1[i]))/2.
                elif np.real(z1[i]) < 0.:
                    zeta = (z[i] - 1j*np.sqrt(-z1[i]))/2.
                    zeta2 = (z[i] + 1j*np.sqrt(-z1[i]))/2.
                elif np.imag(z1[i]) >= 0.:
                    zeta = (z[i] + np.sqrt(z1[i]))/2.
                    zeta2 = (z[i] - np.sqrt(z1[i]))/2.
                else:
                    zeta = (z[i] - 1j*np.sqrt(-z1[i]))/2.
                    zeta2 = (z[i] + 1j*np.sqrt(-z1[i]))/2.
                
                if abs(zeta2 - self.zeta_0) > abs(zeta - self.zeta_0):
                    zeta = zeta2
                zeta_f[i] = zeta
            return zeta_f
        except:
            n = 1
            z1 = z*z - 4*(self.radius - self.epsilon)*(self.radius - self.epsilon)

            if np.real(z1) > 0.:
                zeta = (z + np.sqrt(z1))/2.
                zeta2 = (z - np.sqrt(z1))/2.
            elif np.real(z1) < 0.:
                zeta = (z - 1j*np.sqrt(-z1))/2.
                zeta2 = (z + 1j*np.sqrt(-z1))/2.
            elif np.imag(z1) >= 0.:
                zeta = (z + np.sqrt(z1))/2.
                zeta2 = (z - np.sqrt(z1))/2.
            else:
                zeta = (z - 1j*np.sqrt(-z1))/2.
                zeta2 = (z + 1j*np.sqrt(-z1))/2.
            
            if abs(zeta2 - self.zeta_0) > abs(zeta - self.zeta_0):
                zeta = zeta2
            zeta_f = zeta
            return zeta_f
    
    def surface_normal(self, x):
        
        '''Returns normal vectors at each point along the upper and lower 
        surface of geometry.'''
        
        # uses very small step size in x to avoid some numerical issues at the
        # LE and TE
        dx = 0.0000001
        x1 = x - dx
        x2 = x + dx

        if x1 < -self.radius:
            x1 = -self.radius
        if x2 > self.radius:
            x2 = self.radius

        cu1, yu1, yl1, xu1, xl1 = self.geometry(x1)
        cl2, yu2, yl2, xu2, xl2 = self.geometry(x2)
        
        dxl = xl2 - xl1
        dxu = xu2 - xu1
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
        dx = 0.0000001
        x1 = x - dx
        x2 = x + dx
        
        if x1 < -self.radius:
            x1 = -self.radius
        if x2 > self.radius:
            x2 = self.radius

        cu1, yu1, yl1, xu1, xl1 = self.geometry(x1)
        cl2, yu2, yl2, xu2, xl2 = self.geometry(x2)
        
        dxl = xl2 - xl1
        dxu = xu2 - xu1
        dyu = yu2 - yu1
        dyl = yl2 - yl1
        magu = np.sqrt(dxu*dxu + dyu*dyu)
        magl = np.sqrt(dxl*dxl + dyl*dyl)
         
        # divide by magnitudes to find unit vectors
        upper_tangent = np.array([dxu/magu, dyu/magu]).T
        lower_tangent = np.array([dxl/magl, dyl/magl]).T 
        
        return upper_tangent, lower_tangent

    def velocity(self, x, y):

        '''Velocity around a Joukowski airfoil using conformal mapping of
        velocity around a cylinder'''

        z = complex(x,y) # position in z plane
        zeta = self.zee2zeta(z) # convert to zeta value for computation
        Rsq = self.radius*self.radius
        c1 = np.exp(complex(0,-self.alpha_r))
        c2 = np.exp(complex(0,self.alpha_r))
        d1 = 1 - (((self.radius - self.epsilon)*(self.radius - self.epsilon))/(zeta*zeta))
        
        Wz = self.Vinf*(c1 + complex(0, (self.vortex_str/(2*np.pi*self.Vinf))*(1/(zeta - self.zeta_0))) - Rsq*c2*(1/((zeta - self.zeta_0)*(zeta - self.zeta_0))))/d1
        
        Vx = np.real(Wz)
        Vy = -np.imag(Wz)
        
        Cp = 0.
    
        return Vx, Vy, Cp
    
    def surface_tangential_velocity(self, x):
        
        '''
        Finds velocity at the upper and lower surface of the geometry. 
        Maintains the sign based on the direction of the Y component.
        '''

        camber, upper_y, lower_y, x_upper, x_lower = self.geometry(x)

        ds_off = 1e-5
        norm_u = self.surface_normal(x)[0]
        norm_l = self.surface_normal(x)[1]
        VxU, VyU, CpU = self.velocity(x_upper + ds_off*norm_u[0], upper_y + ds_off*norm_u[1])
        VxL, VyL, CpL = self.velocity(x_lower + ds_off*norm_l[0], lower_y + ds_off*norm_l[1])
    
        upper_tang = self.surface_tangent(x)[0]
        lower_tang = self.surface_tangent(x)[1]
    
        upper_tangential_velocity = np.dot([VxU, VyU], upper_tang)
        lower_tangential_velocity = np.dot([VxL, VyL], lower_tang)
    
        return upper_tangential_velocity, lower_tangential_velocity
    
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
    
        while end_flag == False and abs(s) < 20.:
    
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
    
    def newton(self, x0, f, maxI, epsilon, loc, surface):
        
        '''
        Simple newtons method but including conditions to search top and bottom
        surface as well and front and aft sections of the geometry
        '''
        
        xi = x0
        
        # determines which velocity component to use
        if surface == 'top':
            v_index = 0
        elif surface == 'bottom':
            v_index = 1
        
        # determines step size to use when moving from LE or TE
        # very small step size used to avoid issues at LE/TE boundary
        if loc == 'front':
            dh = 0.000025
        elif loc == 'aft':
            dh = -0.000025
            
        currentI = 0
        error = 100.
        
        while currentI < maxI and  error > epsilon:
            
            # finite difference estimate of derivative
            ff = f(xi + dh)[v_index]
            fm = f(xi)[v_index]
            fp = (ff - fm) / dh
            
            xip = xi
        
            xi = xip - f(xip)[v_index]/fp
            
            error = abs((xi - xip)/xi)*100.0
            currentI += 1
        xf = xi
        print('\nNewtons Method Iterations: ', currentI)
        return xf

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
                xin = x0 + abs(xin)*0.002

            error = abs((xin - xi)/xin)*100.0

            currentI += 1
            xip = xi
            xi = xin
            
        xf = xin
        print('Iterations to convergence: ', currentI)

        return xf
    
    def stagnation(self, x):
        
        '''Finds the leading and trailing edge stagnation point on the potential
        flow geometry'''
        
        thresh = 10**(-8) #threshold for determining zero velocity
        
        # find LE and TE velocities
        LE_V = self.surface_tangential_velocity(self.x_leading_edge)[0]
        TE_V = self.surface_tangential_velocity(self.x_trailing_edge)[0]
        print('LE V: ', LE_V)
        print('TE V: ', TE_V)
        
        if (abs(LE_V) < thresh) or (LE_V == 0.0):
            # condition if front stagnation point is at the LE
            x_stag_front = self.geometry(self.x_leading_edge)[4]
            y_stag_front = self.geometry(self.x_leading_edge)[2]
            front_stag_norm = np.array([-1,0])
            x_stag_temp = self.x_leading_edge
        elif LE_V < 0.:
            # if the LE tangential velocity is negative, stagnation point
            # is assumed to be on the top surface of geometry
            x_stag_temp = self.newton(self.x_leading_edge, self.surface_tangential_velocity, 1000, 0.001, 'front', 'top')
            y_stag_front, _, x_stag_front, _ = self.geometry(x_stag_temp)[1:]
            front_stag_norm = self.surface_normal(x_stag_temp)[0]
        elif LE_V > 0.:
            # if the LE tangential velocity is positive, stagnation point
            # is assumed to be on the bottom surface of geometry
            x_stag_temp = self.newton(self.x_leading_edge, self.surface_tangential_velocity, 1000, 0.001, 'front', 'bottom')
            # x_stag_temp = self.secant(self.x_leading_edge, self.surface_tangential_velocity, 1000, 0.001, 'front', 'bottom')
            _, y_stag_front, _, x_stag_front = self.geometry(x_stag_temp)[1:]
            front_stag_norm = self.surface_normal(x_stag_temp)[1]
        print('\nFore Stagnation point (x,y):' , x_stag_front, y_stag_front)
        print('Velocity at Fore Stagnation Point (Vx, Vy, Cp): ', self.velocity(x_stag_front, y_stag_front))
        print('Tangential velocity at Fore Stagnation Point(up, low): ', self.surface_tangential_velocity(x_stag_temp))
        
        if self.geometry_type != 'airfoil':
            if (abs(TE_V) < thresh) or (LE_V == 0.0):
                # condition if aft stagnation point is at the TE
                x_stag_aft = self.geometry(self.x_trailing_edge)[4]
                y_stag_aft = self.geometry(self.x_trailing_edge)[2]
                aft_stag_norm = np.array([1,0])
                x_stag_temp = self.x_trailing_edge
            elif TE_V < 0.:
                # if the TE tangential velocity is positive, stagnation point
                # is assumed to be on the top surface of geometry
                x_stag_temp = self.newton(self.x_trailing_edge, self.surface_tangential_velocity, 1000, 0.001, 'aft', 'top')
                y_stag_aft, _, x_stag_aft, _ = self.geometry(x_stag_temp)[1:]
                aft_stag_norm = self.surface_normal(x_stag_temp)[0]
            elif TE_V > 0.:
                # if the TE tangential velocity is negative, stagnation point
                # is assumed to be on the bottom surface of geometry
                x_stag_temp = self.newton(self.x_trailing_edge, self.surface_tangential_velocity, 1000, 0.001, 'aft', 'bottom')
                _, y_stag_aft, _, x_stag_aft = self.geometry(x_stag_temp)[1:]
                aft_stag_norm = self.surface_normal(x_stag_temp)[1]
            print('\nAft Stagnation point (x,y):' , x_stag_aft, y_stag_aft)
            print('Velocity at Aft Stagnation Point (Vx, Vy, Cp): ', self.velocity(x_stag_aft, y_stag_aft))
            print('Tangential velocity at Aft Stagnation Point (up, low): ', self.surface_tangential_velocity(x_stag_temp))
        else:
            x_stag_temp = self.x_trailing_edge
            _, y_stag_aft, _, x_stag_aft = self.geometry(x_stag_temp)[1:]
            aft_stag_norm = self.surface_normal(x_stag_temp)[1]

        # offset in normal direction the beginning of the stagnation streamlines 
        ds_off = 0.01
        x_s_f_start = x_stag_front + front_stag_norm[0]*ds_off
        y_s_f_start = y_stag_front + front_stag_norm[1]*ds_off
        x_s_a_start = x_stag_aft + aft_stag_norm[0]*ds_off
        y_s_a_start = y_stag_aft + aft_stag_norm[1]*ds_off
    
        return x_s_f_start, y_s_f_start, x_s_a_start, y_s_a_start

    def plot(self):
        
        # generate geometry surface x and y values
        x_s = np.linspace(-self.radius, self.radius, 1000)
        # geometry in z-plane
        camber, upper_surface, lower_surface, x_upper, x_lower = self.geometry(x_s)
        # geometry in zeta-plane
        _, u_zeta, l_zeta, x_zeta_off = self.geometry_zeta(x_s)
        
        # Find the front and aft stagnation points
        x_s_f, y_s_f, x_s_a, y_s_a = self.stagnation(x_s)
        
        # generate stagnation streamlines
        x_y_s_front = self.streamline(x_s_f, y_s_f, -self.delta_s)
        x_y_s_aft = self.streamline(x_s_a, y_s_a, self.delta_s)
        
        # use the y location of the front stagnation streamline at the boundary
        # to beginning spacing of other streamlines
        y_start = x_y_s_front[-1,1]
        
        plt.figure(1, figsize = [5,5])
        ax = plt.axes()
        ax.axhline(0, color = 'gray', zorder=0)
        ax.axvline(0, color = 'gray', zorder=0)
        
        for i in range(self.n_lines):
            # generates specifed number of streamlines at equal spacing along the
            # front boundary
            y_new_u = y_start + (i + 1)* self.delta_y
            y_new_l = y_start - (i + 1)* self.delta_y 
            
            upper_streamline = self.streamline(self.x_start, y_new_u, self.delta_s)
            lower_streamline = self.streamline(self.x_start, y_new_l, self.delta_s)
            plt.plot(upper_streamline[:,0], upper_streamline[:,1], color = 'k')
            plt.plot(lower_streamline[:,0], lower_streamline[:,1], color = 'k')
        
        
        # z and zeta singularites
        x_z_sing = [-2*(self.radius - self.epsilon), 2*(self.radius - self.epsilon)]
        y_z_sing = [0.0, 0.0]
        
        x_zeta_sing = [-(self.radius - self.epsilon), (self.radius - self.epsilon)]
        y_zeta_sing = [0.0, 0.0]

        plt.scatter(x_zeta_sing, y_zeta_sing, s = 10, color = 'r')
        plt.scatter(x_z_sing, y_z_sing, s = 10, color = 'b')
        plt.scatter(self.x0, self.y0, s = 20, marker = 'o', facecolors = 'none', edgecolors = 'r')
        plt.plot(x_zeta_off, u_zeta, color ='r', linestyle = '--')
        plt.plot(x_zeta_off, l_zeta, color ='r', linestyle = '--')
        plt.plot(x_upper, upper_surface, color = 'b')
        plt.plot(x_lower, lower_surface, color = 'b')
        plt.plot(x_upper, camber, color = 'b')
        plt.plot(x_y_s_front[:,0], x_y_s_front[:,1], color = 'k')
        plt.plot(x_y_s_aft[:,0], x_y_s_aft[:,1], color = 'k')
        plt.xlim(self.x_lower_limit, self.x_upper_limit)
        plt.ylim(self.x_lower_limit, self.x_upper_limit)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Streamlines")
        # plt.tight_layout()
        plt.show()