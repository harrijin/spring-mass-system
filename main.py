from math import isinf, isnan
import numpy as np
import scipy.sparse as sp

class VerticalSpringMassSystem:

    def __init__(self, num_masses, num_fixed_ends, masses, spring_constants, gravity):
        # System parameters
        self.num_masses = num_masses
        self.num_springs = num_masses + (num_fixed_ends - 1)
        self.masses = masses
        self.spring_constants = spring_constants
        self.gravity = gravity
        # System matrices
        self.A = self.calc_A()
        self.A_t = self.A.transpose()
        self.C = self.calc_C()
        self.K = self.calc_K()
        self.f = self.calc_f()

    def calc_A(self):
        return sp.diags([-1, 1], [-1, 0], shape=(self.num_springs, self.num_masses)).toarray()

    def calc_C(self):
        return sp.diags(self.spring_constants, 0).toarray()
    
    def calc_K(self):
        return np.matmul(self.A_t, np.matmul(self.C, self.A))
    
    def calc_f(self):
        return np.multiply(self.gravity, self.masses).transpose()
    
    def solve(self):
        # SVD Decomposition
        U, s, V_t = np.linalg.svd(self.K)
        d = 1.0 / s
        D = sp.diags(d, 0, self.K.shape).toarray()
        # Find K inverse
        K_inv = np.matmul(V_t.transpose(), np.matmul(D, U.transpose()))
        # Solve system
        u = np.matmul(K_inv, self.f)
        e = np.matmul(self.A, u)
        w = np.matmul(self.C, e)
        return u, e, w
    
    def singular_values(self, matrix: str):
        matrix = matrix.upper()
        mat = None
        if matrix == 'A':
            mat = self.A
        elif matrix == 'C':
            mat = self.C
        elif matrix == 'K':
            mat = self.K
        if mat.any():
            s = np.linalg.svd(mat)[1]
            return s
        else:
            return None

    def eigenvalues(self, matrix: str):
        matrix = matrix.upper()
        mat = None
        if matrix == 'A':
            mat = self.A
        elif matrix == 'C':
            mat = self.C
        elif matrix == 'K':
            mat = self.K
        if mat.any():
            try:
                s = np.linalg.eigvals(mat)
                return s
            except:
                return None
        else:
            return None

    def condition_num(self, matrix: str):
        matrix = matrix.upper()
        mat = None
        if matrix == 'A':
            mat = self.A
        elif matrix == 'C':
            mat = self.C
        elif matrix == 'K':
            mat = self.K
        if mat.any():
            s = self.singular_values(matrix)
            return max(s)/min(s)
        else:
            return 0
    
if __name__ == '__main__':
    # Initialize problem parameters
    num_masses = -1
    num_fixed_ends = -1
    num_springs = -1
    gravity = -1
    masses = list()
    spring_constants = list()
    # Mass parameters
    while num_masses < 1:
        num_masses = int(input('Number of masses (n):'))
        if num_masses < 1:
            print("Please enter a nonzero, positive number")
    print("Input mass values from top to bottom")
    for i in range(num_masses):
        input_mass = -1
        while input_mass <= 0:
            input_mass = float(input('Mass (kg) of mass {}:'.format(i+1)))
            if input_mass <= 0:
                print("Please enter a nonzero, positive number")
        masses.append(input_mass)
    # Boundary Conditions
    while (num_fixed_ends < 0) or (num_fixed_ends > 3):
        num_fixed_ends = int(input('Number of fixed ends:\n 0: n-1 springs\n 1: n springs\n 2: n+1 spring\n'))
    # Spring parameters
    num_springs = num_masses + (num_fixed_ends - 1)
    print("Input spring constants from top to bottom")
    for i in range(num_springs):
        input_constant = -1
        while input_constant <= 0:
            input_constant = float(input('Spring constant (N/m) for spring {}:'.format(i+1)))
            if input_constant <= 0:
                print("Please enter a nonzero, positive number")
        spring_constants.append(input_constant)
    # Acceleration due to gravity
    while gravity < 0:
        gravity = input("Acceleration (m/s^2) due to gravity [9.81]:")
        # Check if input was entered
        if not gravity:
            gravity = 9.81
        elif float(gravity) < 0:
            print("Please enter a positive number")
        else:
            gravity = float(gravity)
    # Instantiate system
    sys = VerticalSpringMassSystem(num_masses, num_fixed_ends, masses, spring_constants, gravity)
    K_cond_num = sys.condition_num('K')
    if isinf(K_cond_num) or isnan(K_cond_num):
        print("WARNING: Stiffness matrix K is poorly conditioned. Results are unlikely to be useful.")
    u, e, w = sys.solve()
    # Print answers
    for i in range(len(u)):
        print("Displacement for mass {}:  {:8.4f}".format(i+1, u[i]))
    for i in range(len(e)):
        print("Elongation for spring {}:  {:8.4f}".format(i+1, e[i]))
    for i in range(len(w)):
        print("Internal force for spring {}:  {:8.4f}".format(i+1, w[i]))
    # Print condition numbers, singular values, and eigenvalues
    K_singular_vals = sys.singular_values('K')
    A_singular_vals = sys.singular_values('A')
    C_singular_vals = sys.singular_values('C')
    K_eigenvals = sys.eigenvalues('K')
    A_eigenvals = sys.eigenvalues('A')
    C_eigenvals = sys.eigenvalues('C')
    A_cond_num = sys.condition_num('A')
    C_cond_num = sys.condition_num('C')
    print("Singular values, eigenvalues, and condition numbers are equivalent for A and the transpose of A") 
    print("Condition number for K matrix: {:8.4f}".format(K_cond_num))
    print("Condition number for A matrix: {:8.4f}".format(A_cond_num))
    print("Condition number for C matrix: {:8.4f}".format(C_cond_num))
    print("Singular values for K:")
    for s in K_singular_vals:
        print('{:8.4f}'.format(s))
    print("Eigenvalues for K:")
    for s in K_eigenvals:
        print('{:8.4f}'.format(s))
    print("Singular values for A:")
    for s in A_singular_vals:
        print('{:8.4f}'.format(s))
    print("Eigenvalues for A:")
    if A_eigenvals is not None:
        for s in A_eigenvals:
            print('{:8.4f}'.format(s))
    else:
        print("No eigenvalues because the A matrix is not square")
    print("Singular values for C:")
    for s in C_singular_vals:
        print('{:8.4f}'.format(s))
    print("Eigenvalues for C:")
    for s in C_eigenvals:
        print('{:8.4f}'.format(s))