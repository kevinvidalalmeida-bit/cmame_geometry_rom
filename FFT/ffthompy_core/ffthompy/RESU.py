import numpy as np

def mandel_to_voigt_eng(Cm):
    """
    Mandel (6x6) -> Voigt ingenieril (6x6).
    Voigt ingenieril usa gamma_ij = 2*epsilon_ij, y C44=G.
    """
    T_inv = np.diag([1, 1, 1, 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)])
    return T_inv @ Cm @ T_inv

def engineering_constants_from_Cmandel(Cm):
    Cm = 0.5*(Cm + Cm.T)

    Cv = mandel_to_voigt_eng(Cm)
    S  = np.linalg.inv(Cv)

    E1 = 1.0 / S[0,0]
    E2 = 1.0 / S[1,1]
    E3 = 1.0 / S[2,2]

    # Voigt ingenieril: 4->23, 5->13, 6->12
    G23 = 1.0 / S[3,3]
    G13 = 1.0 / S[4,4]
    G12 = 1.0 / S[5,5]

    nu12 = -S[0,1] / S[0,0]
    nu13 = -S[0,2] / S[0,0]
    nu23 = -S[1,2] / S[1,1]

    return dict(E1=E1,E2=E2,E3=E3, G12=G12,G13=G13,G23=G23, nu12=nu12,nu13=nu13,nu23=nu23)
def extract_properties(Ceff_path):
    """
    Loads Ceff from Ceff_path (.npy) and returns engineering constants dict.
    """
    import numpy as np
    import os
    
    if not os.path.exists(Ceff_path):
        print(f"  [RESU] ERROR: No se encontró {Ceff_path}")
        return None
        
    Ceff = np.load(Ceff_path)
    props = engineering_constants_from_Cmandel(Ceff)
    return props

if __name__ == "__main__":
    # Test con matriz hardcoded si se corre directo (mantener compatibilidad)
    Ceff_example = np.array([
        [ 6.00808661e+00,  2.93692094e+00,  2.93748210e+00,  5.73412981e-04,  5.03638681e-03, -2.84305523e-02],
        [ 2.93692094e+00,  5.97282211e+00,  2.93170646e+00, -4.08168810e-03,  1.16911482e-02, -5.45862412e-02],
        [ 2.93748210e+00,  2.93170646e+00,  5.95424009e+00,  7.30107456e-03,  1.08936121e-02, -6.43114167e-03],
        [ 5.73412981e-04, -4.08168810e-03,  7.30107456e-03,  3.06419603e+00, -1.11178883e-02,  1.67095732e-02],
        [ 5.03638681e-03,  1.16911482e-02,  1.08936121e-02, -1.11178883e-02,  3.08097122e+00,  3.68122534e-03],
        [-2.84305523e-02, -5.45862412e-02, -6.43114167e-03,  1.67095732e-02,  3.68122534e-03,  3.08125030e+00]
    ], dtype=float)
    
    props = engineering_constants_from_Cmandel(Ceff_example)
    for k,v in props.items():
        print(f"{k:5s} = {v:.6f}")