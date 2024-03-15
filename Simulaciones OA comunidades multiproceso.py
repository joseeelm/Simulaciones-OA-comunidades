# Ejecutar codigo de OA comunidades v3 (alfas segun yo)

import multiprocessing
from multiprocessing import Pool
import numpy as np

import Ott_Antonsen_comunidades_v3_GITHUB as OAC

# Valores de x0 para la interaccion ENTRE comunidades:
vec_x0_3 = np.array([-0.04, 0, 0.04, 0.08])

# Ejecucion de las simulaciones para varios x0:
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    pool.map(OAC.ejecutar, vec_x0_3)



