import numpy as np
from scipy.sparse import diags
from typing import Callable, Tuple

class WaveLeapFrog():
    def __init__(self, params:Tuple[float, int, float, int, float]):
        self.L, self.Nx, self.T, self.Nt, self.c = params
        self.dx, self.dt = self.L/(self.Nx-1), self.T/(self.Nt-1)
    
    def solve(self, initial_condition:Callable):
        '''
            Per ora assumo le seguenti
            - 0 al bordo
            - 0 la condizione iniziale sulla derivata
        '''
        # Prepare x space
        x = np.linspace(0, self.L, self.Nx)
        
        # Prepare solution
        solution = np.zeros((self.Nt,self.Nx))
        # Questo shift non è buono metterlo qui
        # Modifica il codice di test del modello e passa qui quello corretto
        solution[0] = np.array([initial_condition(z-self.L/2) for z in x])
        solution[1] = np.copy(solution[0])

        # Build Laplacian matrix
        alpha = (self.c * self.dt / self.dx)**2
        central_diagonal = (2 - 2 * alpha) * np.ones(self.Nx)
        side_diagonal = alpha * np.ones(self.Nx - 1)

        # Finite Central difference Matrix
        M = diags([central_diagonal, side_diagonal, side_diagonal], [0, -1, 1], shape=(self.Nx, self.Nx)).toarray()

        # Boundary conditions
        M[0, :] = M[-1, :] = 0
        M[0, 0] = M[-1, -1] = 1

        # LeapFrog numerical scheme
        for n in range(1, self.Nt-1):
            # Calcolo della prossima soluzione utilizzando il sistema lineare
            solution[n+1] = M @ solution[n] - solution[n-1]
        
        return solution