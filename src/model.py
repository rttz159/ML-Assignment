import numpy as np

class GlucoseDynamicsSimulator:
    '''
        Meals are represented by Rameal(t), the rate of glucose appearance from the gut
        Exercise is represented by E_t, e.g. 0 to 1
    '''
    def __init__(self, weight=70, Gb=90,Ib=15):
        self.weight = weight
        self.Gb = Gb
        self.Ib = Ib
        
        # metabolic rates [mg/min]
        self.rG_brain = 71
        self.rG_heart = 3.7
        self.rG_kidney = 3.7
        self.rG_gut = 16.6
        self.rG_peri = 45.2 * (self.weight / 70)
        self.rG_liver = 14.8 * (self.weight / 70)

        # EGP basal
        self.EGPb = 155 * (self.weight / 70)  # mg/min
        
        # skeletal muscle parameters
        self.rG_SMins = 5    # insulin sensitivity [mg/min per uU/ml]
        self.rG_SMexr = 860  # exercise sensitivity [mg/min per unit E]
        
        # volumes [L]
        self.V_CS_G = 0.07 * weight
        self.V_SM_G = 0.096 * weight
        self.V_SM_tiss = 0.54 * weight
        self.V_SM_I = 0.12 * weight
        
        # insulin kinetic parameters
        self.rSM_I = 0.02    # 1/min
        
        # time constants
        self.tau_liver = 25      # min
        self.tau_EGP = 20        # min
        
        # EGP modulation
        self.eta_EGP = 4         # unitless
        
        # permeability parameters
        self.PS_dG_rest = 0.01    # ml/min/ml tissue
        self.PS_dI_rest = 0.005   # ml/min/ml tissue
        
        self.Rd = 1.46            # max cap recruitment factor
        self.gamma = 10           # cap recruitment saturating rate
        self.lambda_d = 1.1       # exercise effect on perfusion
        
        # hematocrit
        self.h = 0.4
    
    def Qd(self, E):
        return 1.0 + self.lambda_d * E
    
    def QdI(self, E):
        return (1 - self.h) * self.Qd(E)
    
    def PSdG(self, E):
        return self.PS_dG_rest * (1 + self.Rd * np.tanh(self.gamma * E))
    
    def PSdI(self, E):
        return self.PS_dI_rest * (1 + self.Rd * np.tanh(self.gamma * E))

    def kdG(self, E):
        Qd = self.Qd(E)
        PS = self.PSdG(E)
        return Qd * (1 - np.exp(-PS/Qd))
    
    def kdI(self, E):
        QdI = self.QdI(E)
        PSI = self.PSdI(E)
        return QdI * (1 - np.exp(-PSI/QdI))

    def initial_conditions(self):
        E0 = 0.0
        kdG_rest = self.kdG(E0)
        kdI_rest = self.kdI(E0)
        
        G0 = self.Gb
        
        H0 = self.Gb - self.rG_peri / (self.V_SM_tiss * kdG_rest)
        
        S0 = (self.V_SM_tiss * kdI_rest * self.Ib) / (
            self.rSM_I + self.V_SM_tiss * kdI_rest
        )
        
        M0 = 1.0
        L0 = 0.0
        
        return np.array([G0, H0, S0, M0, L0])

    def odes(self, t, y, inputs):
        G, H, S, M, L = y
        
        # get inputs
        Rameal_t, I_t, E_t = inputs(t)
        
        # dynamic coefficients
        kdG = self.kdG(E_t)
        kdI = self.kdI(E_t)
        
        # RGUs
        RGUCSfix = self.rG_brain + self.rG_heart + self.rG_kidney + self.rG_gut
        RGUCSliv = self.rG_liver * (5.66 + 5.66 * np.tanh(2.44 * (G/90) - 1.48*M))
        RGUSMins = self.rG_SMins * (H/90) * (S/15) + self.rG_peri
        RGUSMexr = self.rG_SMexr * (H/90) * E_t
        
        EGP = self.EGPb * (90/G) * (15/S) * (1 + L)
        
        dGdt = (
            -self.V_SM_tiss * kdG * (G - H) / self.V_CS_G
            - RGUCSliv / self.V_CS_G
            - RGUCSfix / self.V_CS_G
            + Rameal_t / self.V_CS_G
            + EGP / self.V_CS_G
        )
        
        dHdt = (
            self.V_SM_tiss * kdG * (G - H) / self.V_SM_G
            - RGUSMins / self.V_SM_G
            - RGUSMexr / self.V_SM_G
        )
        
        dSdt = (
            self.V_SM_tiss * kdI * (I_t - S) / self.V_SM_I
            - self.rSM_I * S
        )
        
        dMdt = 1.0 / self.tau_liver * (2.0 * np.tanh(0.55 * S/15))
        
        dLdt = 1.0 / self.tau_EGP * (-L + self.eta_EGP * E_t)
        '''
            dGdt -> how fast plasma glucose is changing
            dHdt -> how fast tissue glucose is changing
            dSdt -> how fast tissue insulin is changing
            dMdt -> how the liver glucose uptake modifier is changing
            dLdt -> how the exercise glucose production modifier is changing
        '''
        return [dGdt, dHdt, dSdt, dMdt, dLdt]
