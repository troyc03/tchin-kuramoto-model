import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class KuramotoModel:
    def __init__(self, num_oscillators, coupling_strength, natural_frequencies=None):
        self.num_oscillators = num_oscillators
        self.coupling_strength = coupling_strength
        if natural_frequencies is None:
            self.natural_frequencies = np.random.normal(0, 1, num_oscillators)
        else:
            self.natural_frequencies = natural_frequencies
        self.phases = np.random.uniform(0, 2 * np.pi, num_oscillators)
    
    def kuramoto_ode(self, phases, t):
        dphases_dt = np.zeros(self.num_oscillators)
        for i in range(self.num_oscillators):
            interaction = np.sum(np.sin(phases - phases[i]))
            dphases_dt[i] = self.natural_frequencies[i] + (self.coupling_strength / self.num_oscillators) * interaction
        return dphases_dt

    def simulate(self, t):
        self.phases = odeint(self.kuramoto_ode, self.phases, t)
        return self.phases
    
    def plot_phases(self, t):
        for i in range(self.num_oscillators):
            plt.plot(t,self.phases[:,i], label=f'Oscillator {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Phase')
        plt.title('Kuramoto Model Phases Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    num_oscillators = 5
    coupling_strength = 1.0
    t = np.linspace(0, 10, 1000)
    kuramoto = KuramotoModel(num_oscillators, coupling_strength)
    phases = kuramoto.simulate(t)
    kuramoto.plot_phases(t)

if __name__ == "__main__":
    main()
