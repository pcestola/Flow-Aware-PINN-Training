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
    

class HeatLeapFrog:
    def __init__(self, params: Tuple[float, int, float, int, float]):
        """
        params = (L, Nx, T, Nt, alpha)
        L     : lunghezza del dominio spaziale
        Nx    : numero di punti spaziali
        T     : tempo totale di simulazione
        Nt    : numero di passi temporali
        alpha : coefficiente di diffusione termica
        """
        self.L, self.Nx, self.T, self.Nt, self.alpha = params
        self.dx = self.L / (self.Nx - 1)
        self.dt = self.T / (self.Nt - 1)
        
        # Coefficiente numerico
        self.lmbd = self.alpha * self.dt / self.dx**2

    def solve(self, initial_condition: Callable):
        """
        Risolve l'equazione del calore 1D con:
        - condizioni al bordo Dirichlet omogenee (u=0 ai bordi)
        - condizione iniziale specificata
        """
        x = np.linspace(0, self.L, self.Nx)
        u = np.zeros((self.Nt, self.Nx))
        u[0] = np.array([initial_condition(xi - self.L / 2) for xi in x])

        # Matrice del passo temporale (esplicita)
        main_diag = (1 - 2 * self.lmbd) * np.ones(self.Nx)
        off_diag = self.lmbd * np.ones(self.Nx - 1)
        M = diags([main_diag, off_diag, off_diag], [0, -1, 1]).toarray()

        # Condizioni al bordo: fissi i bordi a 0
        M[0, :] = M[-1, :] = 0
        M[0, 0] = M[-1, -1] = 1

        for n in range(self.Nt - 1):
            u[n+1] = M @ u[n]
            u[n+1][0] = u[n+1][-1] = 0  # Rinforza il bordo se necessario

        return u