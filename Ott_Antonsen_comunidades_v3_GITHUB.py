# Resolución numérica de Ott-Antonsen para G(x) espaciales: COMUNIDADES
# Los alfas estan segun YO.


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integ
from scipy import stats as stats
import os


# Convertir ángulo a modulo(2*pi)
def angulo_modulo_2pi(list_theta):
    '''
    Convierte cualquier ángulo theta en su equivalente modulo 2pi, es decir,
    entre el intervalo [-pi, pi]
    '''
    thetas_mod2pi = []
    for theta in list_theta:
        nuevo_theta = np.arctan2(np.sin(theta), np.cos(theta))
        thetas_mod2pi.append(nuevo_theta)
    return thetas_mod2pi




# Función de acoplamiento G(x) de omelchenko:
def G(x, label=False):
    '''
    Función de acoplamiento entre osciladores (según paper Omelchenko).
    
    Proviene de la expansión en serie de Fourier de una función de acopla-
    miento genérica, 2*pi periódica.

    Significado: los osciladores que están más cerca entre sí se acoplan
    más, pero los que están muy lejos igual se acoplan, solo que poquito.

    G(x) = (1/2*pi) * [1 + A*cos(x - x0) ]

    Parametros
    ----------
    x     : [float]
            Posición x para calcular su acoplamiento.
    label : [bool]
            True para retornar el label con la forma de G(x)
    '''
    # Ajustar x para que G sea 2pi-periódica:
    x_menos_x0_mod2pi = angulo_modulo_2pi([x-x0_1])[0]
    
    label_G = r'$(1/2\pi)[1 + {}*\cos(x-{})]$'.format(A1, x0_1)
    title = 'G(x) Comunidad 1'
    G = ( 1 / (2*np.pi) ) * (1 + A1*np.cos(x_menos_x0_mod2pi) )

    if label == True:
            return G, label_G, title
    else:
        return G
    
    
def K(x, label=False):
    '''
    Función de acoplamiento entre osciladores (según paper Omelchenko).
    
    Proviene de la expansión en serie de Fourier de una función de acopla-
    miento genérica, 2*pi periódica.

    Significado: los osciladores que están más cerca entre sí se acoplan
    más, pero los que están muy lejos igual se acoplan, solo que poquito.

    K(x) = (1/2*pi) * [1 + A*cos(x - x0)]

    Parametros
    ----------
    x     : [float]
            Posición x para calcular su acoplamiento.
    label : [bool]
            True para retornar el label con la forma de G(x)
    '''
    # Ajustar x para que G sea 2pi-periódica:
    x_menos_x0_mod2pi = angulo_modulo_2pi([x-x0_2])[0]
    
    label_K = r'$(1/2\pi)[1 + {}*\cos(x-{})]$'.format(A2, x0_2)
    title = 'K(x) Comunidad 2'
    K = ( 1 / (2*np.pi) ) * (1 + A2*np.cos(x_menos_x0_mod2pi) )

    if label == True:
            return K, label_K, title
    else:
        return K
    

# Funcion de acoplamiento P(x) entre comunidades:
def P(x, label=False):
    '''
    Función de acoplamiento entre comunidades.
    
    P(x) = (epsilon/2*pi) * [1 + A*cos(x - x0)]

    Parametros
    ----------
    x     : [float]
            Posición x para calcular su acoplamiento.
    label : [bool]
            True para retornar el label con la forma de G(x)
    '''
    # Ajustar x para que G sea 2pi-periódica:
    x_menos_x0_mod2pi = angulo_modulo_2pi([x-x0_3])[0]
    
    label_P = r'$({} / 2\pi)[1 + {}*\cos(x-{})]$'.format(epsilon, A3, x0_3)
    title = 'P(x) entre Comunidades'
    P = ( epsilon / (2*np.pi) ) * (1 + A3*np.cos(x_menos_x0_mod2pi) )

    if label == True:
            return P, label_P, title
    else:
        return P




# Grafico G(x):
def graficar_G(func, A, x0, title):
    '''
    Grafica la función de acoplamiento entre osciladores, G(x).

    Parametros
    ----------
    func  : [function]
            Función de acoplamiento G(x)
    A, x0 : [float]
            Parametros de la funcion de acoplamiento func.
    title : [str]
            Titulo del acoplamiento.
    '''
    vec_x = np.linspace(-np.pi, np.pi, 1000)
    G_x = []
    for x in vec_x:
        G_x.append( func(x) )
    label_plot = func(0, label=True)[1]
    
    fig, ax = plt.subplots(1,1)
    ax.plot(vec_x, G_x, label=label_plot)
    ax.set_xlabel(r'Posición $x$')
    x_ticks_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi$/2', r'$\pi$']
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(x_ticks_labels)
    ax.set_ylabel(r'$G(x)$')
    ax.legend(loc='upper left')
    if TIPO_G == 'omelchenko':
        fig.savefig(r'{}/{}/Funcion de acoplamiento G ({}) - A={}, x0={}.pdf'
                    .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_pdf, title, A, x0))
        fig.savefig(r'{}/{}/Funcion de acoplamiento G ({}) - A={}, x0={}.svg'
                    .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_svg, title, A, x0))
        ax.set_title('Función de acoplamiento no-local G(x) - {}'.format(title))
        fig.savefig(r'{}/{}/Funcion de acoplamiento G ({}) - A={}, x0={}.png'
                    .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_png, title, A, x0))
    # Plotear sin parar el código:
    #fig.canvas.draw()
    #renderer = fig.canvas.renderer
    #fig.draw(renderer)
    #plt.pause(0.001)
    





''' ################   INTERGACION OTT-ANTONSEN   ################### '''

# Condición inicial para los osciladores:
def condicion_inicial_thetas(N, label=False, tipo='gausiana-ordenada'):
    '''
    Crea una condición inicial para las fases de los osciladores, obtenidas
    de una distribución uniforme entre -pi y pi.

    Parametros
    ----------
    label : [bool]
            True si se quiere recibir el label de la distrib. de fases.
    '''
    
    if tipo == 'all-random':
        # TODAS LAS CONDICIONES INICIALES RANDOM (-PI, PI)
        theta_0 = stats.uniform.rvs(loc=-np.pi, scale=2*np.pi, size=N)
        label_c = 'uniform'

    if tipo == 'mitad-0':
        # Alternativa para definir theta_0 exactamente igual a Omelchenko.
        # MITAD DE OSCILADORES CON FASE CERO, MITAD CON FASE RANDOM (-PI, PI)
        theta_0_dist_0 = stats.uniform.rvs(loc=-np.pi, scale=2*np.pi, size=int(N/2))
        theta_0_cero = np.zeros(N - int(N/2))
        theta_0 = np.array(list(theta_0_dist_0) + list(theta_0_cero))
        np.random.shuffle(theta_0)
        label_c = 'uniform-zeros'

    if tipo == 'mitad-random-boosted-in-0':
        # Alternativa para obtener la mayoria de las fases iniciales en cero,
        # pero no de los últimos osciladores sino que 'centrados'.
        theta_0_p1 = stats.uniform.rvs(loc=-np.pi, scale=2*np.pi, size=int(N/2))
        theta_0_p2 = stats.uniform.rvs(loc=-np.pi/6, scale=np.pi/3, size=int(N/2))
        theta_0 = np.array(list(theta_0_p1) + list(theta_0_p2))
        np.random.shuffle(theta_0)
        label_c = 'mitad-random-boosted-in-0'

    if tipo == 'gausiana':
        # Alternativa para que las fases iniciales sean de una distribucion
        # gaussiana entre [-pi, pi]. Se corrige que si se salen de ese rango,
        # se les aplica modulo 2*pi.
        theta_0_raw = stats.norm.rvs(loc=0, scale=np.pi/4, size=N)
        theta_0 = np.array(angulo_modulo_2pi(theta_0_raw))
        label_c = 'gausiana'

    if tipo == 'gausiana-ordenada':
        # Alternativa para que las fases iniciales sean de una distribucion
        # gaussiana entre [-pi, pi]. Se corrige que si se salen de ese rango,
        # se les aplica modulo 2*pi, y que los osciladores cercanos a x=0
        # tengan fase inicial cercana a cero
        n = int(N/3)
        m1 = int(np.round((N - n)/2))
        m2 = N - n - m1
        theta_0_1 = stats.norm.rvs(loc=0, scale=np.pi/2, size=m1)
        theta_0_1mod = angulo_modulo_2pi( theta_0_1 )
        theta_0_2 = stats.norm.rvs(loc=0, scale=np.pi/6, size=n)
        theta_0_2mod = angulo_modulo_2pi( theta_0_2 )
        theta_0_3 = stats.norm.rvs(loc=0, scale=np.pi/2, size=m2)
        theta_0_3mod = angulo_modulo_2pi( theta_0_3 )
        theta_0 = np.array(list(theta_0_1mod) + list(theta_0_2mod) + list(theta_0_3mod))
        label_c = 'gausiana-ordenada'

    if label == True:
        return theta_0, label_c
    else:
        return theta_0




# Función gausiana para la condición inicial de z(x, t=0)
def gausiana(x, mu, sigma):
    '''
    Entrega el valor de una gaussiana de promedio mu y desv. estandar sigma
    '''
    #output = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 /(2*sigma**2) )
    # Sin normalizar para que |z| esté entre 0 y 1.
    output_nonorm = np.exp(-(x - mu)**2 /(2*sigma**2) )
    
    return output_nonorm



# Condición inicial para z(x,t=0) directamente
def condicion_inicial_z(CI='omelchenko'):
    '''
    Crea una condición inicial para el parámetro de orden z(x,t=0).

    Obs: z(x,t=0) es complejo (suma de exponenciales de las fases)
    '''
    vec_x = np.linspace(-np.pi, np.pi, num_x)
    
    # Parámetros gaussiana
    mu = 0; sigma = np.pi/4

    ''' Parámetro condición inicial |z| '''
    c = 0.7     # parámetro para condición inicial de |z|
    # (1 - c) es el orden de los osciladores desincronizados
    
    mod_z_0 = []    # arreglo de |z(x, t=0)|

    if CI == 'gausiana':
        for x in vec_x:
            mod_z_x = c*gausiana(x, mu=mu, sigma=sigma)+(1-c)   # |z|
            mod_z_0.append(mod_z_x)
    elif CI == 'omelchenko':
        for x in vec_x:
            mod_z_x = (np.cos(x/2))**2
            mod_z_0.append(mod_z_x)

    mod_z_0 = np.array(mod_z_0)
    label_ci = 'cos2(x medios)'
    
    #arg_z_0 = np.array(arg_z_0)
    #fases_0  = condicion_inicial_thetas(num_x, label=True)
    #arg_z_0  = fases_0[0]   # arg(z) en t=0
    
    z_0 = []
    for i in range(num_x):
        mod_z_0_i = mod_z_0[i]
        #arg_z_0_i = arg_z_0[i]
        #z_0.append( complex(mod_z_0_i, arg_z_0_i) )
        ''' Estoy considerando todas las fases iniciales = 0 '''
        z_0.append( complex(mod_z_0_i, 0) )
        
    return np.array(z_0), label_ci




# Graficar |z(x, t=0)| condición inicial para el parámetro de orden:
def graficar_z_inicial(z_0, A, x0, alpha, gamma, label=''):
    '''
    Grafica el parámetro de orden inicial |z(x, t=0)| en función de x.
    '''
    mod_z0 = np.abs(z_0)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(vec_x, mod_z0)
    ax.set_xlabel(r'Posición $x$', size=12)
    x_ticks_labels = [r'$-\pi$', '0', r'$\pi$']
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(x_ticks_labels)
    y_ticks_labels = ['0', '1']
    ax.set_yticks([0, 1])
    ax.set_yticklabels(y_ticks_labels)
    ax.set_ylabel(r'$|z(t={})|$'.format(min_t), size=12)

    if TIPO_G == 'omelchenko':
        fig.savefig(r'{}/{}/Cond. inicial mod(z) ({}) - y={}, A={}, x0={}, alpha={}, min_t={}, max_t={}, num_x={}.pdf'
                    .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_pdf, label, gamma, A, x0, np.round(alpha,3),
                            min_t, max_t, num_x))
        fig.savefig(r'{}/{}/Cond. inicial mod(z) ({}) - y={}, A={}, x0={}, alpha={}, min_t={}, max_t={}, num_x={}.svg'
                    .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_svg, label, gamma, A, x0, np.round(alpha,3),
                            min_t, max_t, num_x))
        ax.set_title('Cond. inicial mod(z) - {}'.format(label))
        fig.savefig(r'{}/{}/Cond. inicial mod(z) ({}) - y={}, A={}, x0={}, alpha={}, min_t={}, max_t={}, num_x={}.png'
                    .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_png, label, gamma, A, x0, np.round(alpha,3),
                            min_t, max_t, num_x))
    # Plotear sin parar el código:
    #fig.canvas.draw()
    #renderer = fig.canvas.renderer
    #fig.draw(renderer)
    #plt.pause(0.001)




# Operador de convolución:
def operador_convolucion(F, z, x):
    '''
    Operador de convolución de la función F con la variable z(x, t=t*).

    Parametros
    ----------
    F : [function]
        Función de acoplamiento entre osciladores.
    z : [np.array]
        Arreglo de valores de z(x=x*, t=t*).
    x : [float]
        Posición x* asociada a z(x=x*, t=t*).
    '''
    vec_y = np.linspace(-np.pi, np.pi, N1)    # Arreglo de y's

    arg_integral = []   # Lista [G(x-y1)*z(y1), G(x-y2)*z(y2), ...]
    for j in range(N1):
        y = vec_y[j]    # y
        z_y = z[j]      # z(x=y, t)
        
        F_x_menos_y = F(x - y)  # G(x - y)

        arg_integral_j = F_x_menos_y * z_y  # G(x - y)*z(y)
        arg_integral.append(arg_integral_j) # Agrego eso a la lista

    arg_integral = np.array(arg_integral)
    integral = integ.trapz(arg_integral, vec_y)     # Integral en dy

    return integral





# Lado derecho de la ec. de Ott-Antonsen Espacial
def lado_derecho_OA_coms(t, y):
    '''
    Lado derecho de la ec. de Ott-Antonsen para acoplamientos espaciales.

    Parametros
    ----------
    y : [np.array]
        Arreglo con todos los valores de z(x,t*) en el tiempo t*.
    
    OBS: y = [z1, z1](x, t*)
    '''
    '''
    # Elección de función de acoplamiento:
    if TIPO_G == 'omelchenko':
        func = G
    elif TIPO_G == 'omel_local':
        func = G_local
    elif TIPO_G == 'asim_vecinos':
        func = G_asimetrica_vecinos
    '''

    # Parámetro de orden z1(x, t=t*) = [z1, z2]
    z1_t = y[0:N1]              # z1(x, t=t*)
    z1_t_conj = np.conj(z1_t)   # z1(x, t=t*)
    
    z2_t = y[N1:]               # z2(x, t=t*)
    z2_t_conj = np.conj(z2_t)   # z2(x, t=t*)
    
    # Multiplicaciones por gamma_1 y gamma_2:
    gamma_1_por_z1 = gamma_1 * z1_t     # Término gamma_1*z_1(x, t=t*)
    gamma_2_por_z2 = gamma_2 * z2_t     # Término gamma_1*z_2(x, t=t*)
    
    # Cálculo del lado derecho de la ecuación:
    lado_derecho_x = []
    
    # === Evolucion z1 COMUNIDAD 1:    
    for i in range(N1):     # para cada x = vec_x[i] de z(x,t)
        x_i = vec_x[i]    # x* asociado a ese z1(x,t=t*)
        
        # --- INTERACCION COMUNIDAD 1 CONSIGO MISMA:
        # Termino 1 = (1/2) * exp(+i*alpha_1) * Z_1
        Z_1 = operador_convolucion(G, z=z1_t, x=x_i)           # Z_1(x)
        termino_1_c1 = (1/2) * exp_pos_1 * Z_1
        
        # Termino 2 = (1/2) * exp(-i*alpha_1) * z_1^2 * Z_1*
        Z_1_conj = operador_convolucion(G, z=z1_t_conj, x=x_i) # Z_1*(x)
        z1_x_t = z1_t[i]                                     # z1(x=x*, t=t*)
        termino_2_c1 = (1/2) * exp_neg_1 * (z1_x_t**2) * Z_1_conj
        
        # --- INTERACCION ENTRE COMUNIDADES:
        # Termino 3: (1/2) * exp(+i*alpha_3) * P_2
        P_2 = operador_convolucion(P, z=z2_t, x=x_i)          # P_2(x)
        termino_3_c1 = (1/2) * exp_pos_3 * P_2
        
        # Termino 4: (1/2) * exp(-i*alpha_3) * z_1^2 * P_2*
        P_2_conj = operador_convolucion(P, z=z2_t_conj, x=x_i)     # P_2*(x)
        termino_4_c1 = (1/2) * exp_neg_3 * (z1_x_t**2) * P_2_conj
            
        # >>> Lado derecho de la ecuacion COMUNIDAD 1:
        lado_der_i_com1 = ( -gamma_1_por_z1[i] + termino_1_c1 - termino_2_c1
                           + termino_3_c1 - termino_4_c1 )

        # Lo agrego al arreglo de z(x=x*, t=t*+dt)
        lado_derecho_x.append(lado_der_i_com1)
    
    # === Evolucion z2 COMUNIDAD 2:        
    for j in range(N2):
        x_j = vec_x[j]    # x* asociado a ese z2(x,t=t*)
        
        # --- INTERACCION COMUNIDAD 2 CONSIGO MISMA:
        # Termino 1 = (1/2) * exp(+i*alpha_2) * Z_2
        Z_2 = operador_convolucion(K, z=z2_t, x=x_j)           # Z_1(x)
        termino_1_c2 = (1/2) * exp_pos_2 * Z_2
        
        # Termino 2 = (1/2) * exp(-i*alpha_2) * z_2^2 * Z_2*
        Z_2_conj = operador_convolucion(K, z=z2_t_conj, x=x_j) # Z_2*(x)
        z2_x_t = z2_t[j]                                     # z2(x=x*, t=t*)
        termino_2_c2 = (1/2) * exp_neg_2 * (z2_x_t**2) * Z_2_conj
        
        # --- INTERACCION ENTRE COMUNIDADES:
        # Termino 3: (1/2) * exp(+i*alpha_3) * P_1
        P_1 = operador_convolucion(P, z=z1_t, x=x_j)          # P_1(x)
        termino_3_c2 = (1/2) * exp_pos_3 * P_1
        
        # Termino 4: (1/2) * exp(-i*alpha_3) * z_2^2 * P_1*
        P_1_conj = operador_convolucion(P, z=z1_t_conj, x=x_j)     # P_1*(x)
        termino_4_c2 = (1/2) * exp_neg_3 * (z2_x_t**2) * P_1_conj
            
        # >>> Lado derecho de la ecuacion COMUNIDAD 2:
        lado_der_j_com2 = ( -gamma_2_por_z2[j] + termino_1_c2 - termino_2_c2
                           + termino_3_c2 - termino_4_c2 )

        # Lo agrego al arreglo de z(x=x*, t=t*+dt)
        lado_derecho_x.append(lado_der_j_com2)

    lado_derecho_x = np.array(lado_derecho_x)
    
    return lado_derecho_x





""" PARÁMETROS INTEGRACIÓN TEMPORAL z(x,t) """

# Evolucion temporal de z(x,t) con RK
def evolucion_temporal_z(A1, A2, A3, x0_1, x0_2, x0_3,
                         alpha_1, alpha_2, alpha_3, epsilon,
                         max_t1, max_t2,
                         cond_inicial=True, label_ci=''):
    '''
    Evolucion temporal de z(x,t).
    '''    
    # Graficar G(x)
    graficar_G(G, A1, x0_1, title='Com 1')
    graficar_G(K, A2, x0_2, title='Com 2')
    graficar_G(P, A3, x0_3, title='Interaccion entre comunidades')
    
    # Condicion inicial:
    if cond_inicial == True:    # tomar como CI el z de cada comunidad antes
                                # de empezar a interactuar
        # Comunidad 1:
        '''
        # Esto es cuando se leen los datos completos de z(x,t):
        matriz_z1, t_values_1, n_tiempos_1, label_ci1 = leer_datos_z_una_com(gamma_1,
                                                                             A1, x0_1,
                                                                             alpha_1,
                                                                             num_x=N1)
        z1_0 = np.array(matriz_z1[-1, :])[0]    # z1[t*=tf, x] (ultimo tiempo)
        '''
        # Esto es para leer los nuevos datos que solo contienen z(x, t*=tf)
        z1_0, label_ci_1 = leer_datos_z_una_com(gamma_1, A1, x0_1, alpha_1,
                                                num_x=N1, max_t=max_t1)
        
        # Comunidad 2:
        '''
        # Esto es cuando se leen los datos completos de z(x,t):
        matriz_z2, t_values_2, n_tiempos_2, label_ci2 = leer_datos_z_una_com(gamma_2,
                                                                             A2, x0_2,
                                                                             alpha_2,
                                                                             num_x=N2)
        z2_0 = np.array(matriz_z2[-1, :])[0]    # z2[t*=tf, x] (ultimo tiempo)
        '''
        # Esto es para leer los nuevos datos que solo contienen z(x, t*=tf)
        z2_0, label_ci_2 = leer_datos_z_una_com(gamma_2, A2, x0_2, alpha_2,
                                                num_x=N2, max_t=max_t2)
        
    else:   # si no doy la condición inicial, la creo:
        z1_0, label_ci1 = condicion_inicial_z()
        z2_0, label_ci2 = condicion_inicial_z()
        print('* Condición inicial z_0 calculada.\n')
        
    # Grafico la condicion inicial:
    graficar_z_inicial(z1_0, A1, x0_1, alpha_1, gamma_1, label='Comunidad 1')
    graficar_z_inicial(z2_0, A2, x0_2, alpha_2, gamma_2, label='Comunidad 2')
    print('* Condiciones iniciales graficadas.\n')
    
    # Concatenar listas de z para ambas comunidades (condicion inicial):
    z_0_global = np.array( list(z1_0) + list(z1_0) ) 
    
    # Evolución temporal:
    solution = integ.RK45(lado_derecho_OA_coms, t0=min_t, y0=z_0_global,
                          t_bound=max_t_coms, max_step=dt)

    # Ejecutar la integración de la EDO y guardar valores
    t_values = [min_t]
    z_values = []
    
    while True: # Para cada paso de tiempo
        solution.step()
        print('Paso t=({}/{})s de la integración realizado.'
              .format(np.round(solution.t, 1), max_t_coms))
        
        t_values.append(solution.t)     # Guardo ese tiempo t*
        z_values.append(solution.y)     # Guardo los valores de z(x, t=t*)

        if solution.status == 'failed':
            print('*** Falló la intergación :(')
            break
                
        if solution.status == 'finished':
            print('!!! Resultó la integración :D')
            break

    n_tiempos = len(t_values)   # Numero de dt's de tiempo de la integración

    ''' Paso todos los valores de z(x, t) a una matriz '''
    # Matriz de z(x,t): sus columnas (h) son z(x, t=t*)
    matriz_z = np.zeros([n_tiempos, num_x], dtype=complex)
    matriz_z[0, :] = z_0_global           # Guardo condición inicial
    
    for i in range(n_tiempos-1):     # Para c/ tiempo t*
        matriz_z[i+1, :] = z_values[i]  # guardo z(x, t=t*)
   
    # ======== Guardar datos en un archivo de texto:
    # >>> COMUNIDAD 1:
    if TIPO_G == 'omelchenko':
        f1 = open(r'{}/{}/Ott Antonsen z(x,t) {} - CI={}, dt={}, A3={}, x0_3={}, alpha_3={}, num_x={}, min_t={}, max_t={}.txt'
                 .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta, 'Com1', label_ci, dt, A3, x0_3, np.round(alpha_3,3),
                         num_x, min_t, max_t_coms), 'w')
    
    f1.write('*** Valores de z1(x,t) de la Ec. de Ott-Antonsen Espacial 1D/n')
    f1.write('~ Tipo de interaccion: COMUNIDADES [COMUNIDAD 1]/n')
    f1.write('~ Tipo de acoplamiento: P(x) = (epsilon/2pi)*(1 + A3 * cos(x - x0_3) )/n')
    f1.write('Parametros:/n')
    if TIPO_G == 'omelchenko':
        f1.write('dt={}, gamma_1={}, gamma_2={}\n'.format(dt, gamma_1, gamma_2))
        f1.write('A1={}, A2={}, A3={}, x0_1={}, x0_2={}, x0_3={}\n'.format(A1, A2,
                                                                          A3, x0_1,
                                                                          x0_2, x0_3))
        f1.write('beta_1={}, beta_2={}, beta_3={}\n'.format(beta_1, beta_2,
                                                           beta_3))
        f1.write('epsilon={}, num_x={}, min_t={}, max_t={} s/n'
                .format(epsilon, num_x, min_t, max_t_coms))

    f1.write('Valores de tiempo t:/n')
    f1.write('{}/n'.format(t_values))
    f1.write('/n')
    f1.write('*** Matriz de z(x,t)/n')
    f1.write('- Filas (horizontales) : z(x)/n')
    f1.write('- Columnas (verticales): Tiempo t/n/n')

    f1.write('[*] Parámetro de orden z(x,t): [copiar todo P y pegar como np.matrix(P)]/n')
    z1_to_write = matriz_z[:, 0:N1]

    list_z1s = []
    for i in range(n_tiempos):      # Para cada tiempo t
        z1_t = []                        # z(x, t=t*)
        for k in range(N1):          # Para cada posición x
            z1_x_t = z1_to_write[i, k]        # z(x, t) (en vdd z(t,x) )
            z1_t.append(z1_x_t)
        list_z1s.append(z1_t)

    f1.write('{}'.format(list_z1s))            
    f1.close()
    
    # >>> COMUNIDAD 1:
    if TIPO_G == 'omelchenko':
        f2 = open(r'{}/{}/Ott Antonsen z(x,t) {} - CI={}, dt={}, A3={}, x0_3={}, alpha_3={}, num_x={}, min_t={}, max_t={}.txt'
                 .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta, 'Com2', label_ci, dt, A3, x0_3, np.round(alpha_3,3),
                         num_x, min_t, max_t_coms), 'w')
    
    f2.write('*** Valores de z1(x,t) de la Ec. de Ott-Antonsen Espacial 1D/n')
    f2.write('~ Tipo de interaccion: COMUNIDADES [COMUNIDAD 2] /n')
    f2.write('~ Tipo de acoplamiento: P(x) = (epsilon/2pi)*(1 + A3 * cos(x - x0_3) )/n')
    f2.write('Parametros:/n')
    if TIPO_G == 'omelchenko':
        f2.write('dt={}, gamma_1={}, gamma_2={}\n'.format(dt, gamma_1, gamma_2))
        f2.write('A1={}, A2={}, A3={}, x0_1={}, x0_2={}, x0_3={}\n'.format(A1, A2,
                                                                          A3, x0_1,
                                                                          x0_2, x0_3))
        f2.write('beta_1={}, beta_2={}, beta_3={}\n'.format(beta_1, beta_2,
                                                           beta_3))
        f2.write('epsilon={}, num_x={}, min_t={}, max_t={} s/n'
                .format(epsilon, num_x, min_t, max_t_coms))

    f2.write('Valores de tiempo t:/n')
    f2.write('{}/n'.format(t_values))
    f2.write('/n')
    f2.write('*** Matriz de z(x,t)/n')
    f2.write('- Filas (horizontales) : z(x)/n')
    f2.write('- Columnas (verticales): Tiempo t/n/n')

    f2.write('[*] Parámetro de orden z(x,t): [copiar todo P y pegar como np.matrix(P)]/n')
    z2_to_write = matriz_z[:, N1:]

    list_z2s = []
    for i in range(n_tiempos):      # Para cada tiempo t
        z2_t = []                        # z(x, t=t*)
        for k in range(N2):          # Para cada posición x
            z2_x_t = z2_to_write[i, k]        # z(x, t) (en vdd z(t,x) )
            z2_t.append(z2_x_t)
        list_z2s.append(z2_t)

    f2.write('{}'.format(list_z2s))            
    f2.close()
    
    return matriz_z, t_values, n_tiempos, label_ci

#matriz_z, t_values, n_tiempos, label_ci = evolucion_temporal_z()



""" GRÁFICOS DEL MÓDULO DEL PARÁMETRO DE ORDEN LOCAL z(x,t) """
# Grafico mapa de color z(t,x):
def grafico_z(z, gamma, alpha, A, x0, title):
    '''
    Muestra gráficos del módulo de z(t,x) en función del tiempo en un mapa
    de color. En el eje horizontal aparece la posición x, en el eje vertical
    sale el tiempo t, y en color aparece |z| desde 0 a 1.

    Parametros
    ----------
    z : [np.matrix]
        Matriz con los valores del parámetro de orden local (complejos).
    '''    
    flip_z = np.flip(z, axis=0) # Giro para que quede bien el eje t
    modulo_z = np.abs(flip_z)   # Matriz del módulo de z = |z(t,x)|

    # Plot:
    fig, ax = plt.subplots(1,1)
    im = ax.imshow(modulo_z, aspect='auto', cmap='cividis', vmin=0, vmax=1)
    # Labels de los ejes:
    ax.set_xlabel(r'Posición $x$', size=12)
    ax.set_ylabel(r'Tiempo $t$', size=12)
    # Ticks posición:
    x_ticks_labels = [r'$-\pi$', '0', r'$\pi$']
    ax.set_xticks([0, int(N1/2), N1-1])
    ax.set_xticklabels(x_ticks_labels)
    # Ticks tiempo (están al revés pq el tiempo va de abajo hacia arriba,
    #              (y los índices de la matriz van de arriba hacia abajo).
    ax.set_yticks([0, n_tiempos-1])
    ax.set_yticklabels(['{}'.format(max_t_coms), '{}'.format(min_t)])
    # Colorbar:
    cbar = fig.colorbar(im, ticks=[0, 1], orientation='vertical')
    cbar.set_label(r'$|z|$', size=13)
    # Savefig y titulo:
    if TIPO_G == 'omelchenko':
        fig.savefig(r'{}/{}/Modulo z ({}) - dt={}, y={}, A={}, x0={}, alpha={}, num_x={}, min_t={}, max_t={}.pdf'
                    .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_pdf, title, dt, gamma, A, x0,
                            np.round(alpha,3), num_x, min_t, max_t_coms))
        fig.savefig(r'{}/{}/Modulo z ({}) - dt={}, y={}, A={}, x0={}, alpha={}, num_x={}, min_t={}, max_t={}.svg'
                    .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_svg, title, dt, gamma, A, x0,
                            np.round(alpha,3), num_x, min_t, max_t_coms))
        ax.set_title(r'{}, $\gamma$={}, $\alpha$={}, $A$={}, $x_0$={}'
                     .format(title, gamma, np.round(alpha,3), A, x0))
        fig.savefig(r'{}/{}/Modulo z ({}) - dt={}, y={}, A={}, x0={}, alpha={}, num_x={}, min_t={}, max_t={}.png'
                    .format(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_png, title, dt, gamma, A, x0,
                            np.round(alpha,3), num_x, min_t, max_t_coms))
    # Plotear sin parar el código:
    #fig.canvas.draw()
    #renderer1 = fig.canvas.renderer
    #fig.draw(renderer1)
    #plt.pause(0.001)

#grafico_z(z)





# Función para leer datos ya obtenidos en simulaciones anteriores:
def leer_datos_z_una_com_enteros(gamma, A, x0, alpha, num_x,
                         label_ci='cos2(x medios)', mod_z=False):
    '''
    Lee los datos de la simulación numérica y entrega:

    Parametros
    ----------
    label_ci : [str]
               Describe la condición inicial usada.
    mod_z    : [bool]
               True si se quiere recibir el |z(x,t)| en vez de z(x,t)
    
    Output:
    -------
    matriz_z, t_values, n_tiempos, label_ci
    '''
    # La siguiente variable se debe actualizar con el directorio donde estan los datos:
    # OBS: Se debe usar el caracter / para separar carpetas en el nombre del directorio:
    #NOMBRE_DIRECTORIO_DATOS = 'D:/Proyectos/Jose Luis Lopez/Simulaciones-Kuramoto-Comunidades/DATOS simulaciones OA'
    # ejemplo:
    NOMBRE_DIRECTORIO_DATOS = 'E:/NETWORKS and DYNAMICS/Ott-Antonsen/Comunidades/DATOS simulaciones OA/Solo z0 t_f (alfas segun yo)'
    
    
    if TIPO_G == 'omelchenko':
        f = open(r'{}/Ott Antonsen z(x,t) - CI={}, dt={}, y={}, A={}, x0={}, alpha={}, num_x={}, min_t={}, max_t={}.txt'
                 .format(NOMBRE_DIRECTORIO_DATOS, label_ci, dt, gamma,
                         A, x0, np.round(alpha,3), num_x, min_t, max_t), 'r')
    
    # Extraer t_values:
    for i in range(6):
        f.readline()
    t_values = eval(f.readline())                   # t_values

    # Extraer matriz_thetas:
    for i in range(6):
        f.readline()
    matriz_z = np.matrix(eval(f.readline()))        # matriz_z

    f.close()
    
    n_tiempos = len(t_values)                       # n_tiempos

    if mod_z == True:
        #flip_z = np.flip(matriz_z, axis=0)
        modulo_z = np.abs(matriz_z)
        return modulo_z, t_values, n_tiempos, label_ci

    else:
        return matriz_z, t_values, n_tiempos, label_ci

#matriz_z, t_values, n_tiempos, label_ci = leer_datos(label_ci)







# Función para leer datos ya obtenidos en simulaciones anteriores:
def leer_datos_z_una_com(gamma, A, x0, alpha, num_x, max_t,
                         label_ci='cos2(x medios)', mod_z=False):
    '''
    Lee los datos de la simulación numérica y entrega:

    Parametros
    ----------
    label_ci : [str]
               Describe la condición inicial usada.
    mod_z    : [bool]
               True si se quiere recibir el |z(x,t)| en vez de z(x,t)
    
    Output:
    -------
    matriz_z, t_values, n_tiempos, label_ci
    '''
    global NOMBRE_DIRECTORIO_DATOS
    
    if TIPO_G == 'omelchenko':
        f = open(r'{}/Ott Antonsen z(x,t) F - CI={}, dt={}, y={}, A={}, x0={}, alpha={}, num_x={}, min_t={}, max_t={}.txt'
                 .format(NOMBRE_DIRECTORIO_DATOS, label_ci, dt, gamma,
                         A, x0, np.round(alpha,3), num_x, min_t, max_t), 'r')
    
    # Extraer t_values:
    for i in range(8):
        f.readline()
    
    ultimos_z = np.array(eval(f.readline()))        # matriz_z
    f.close()
    
    if mod_z == True:
        #flip_z = np.flip(matriz_z, axis=0)
        modulo_ultimos_z = np.abs(ultimos_z)
        return modulo_ultimos_z, label_ci

    else:
        return ultimos_z, label_ci

#ultimos_z, label_ci = leer_datos(label_ci)






""" ================= EJECUCIÓN SIMULACIÓN NUMÉRICA ================== """

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~ PARÁMETROS ~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
TIPO_G = 'omelchenko'     # = {'omelchenko', 'omel_local', 'asim_vecinos'}

# Parámetros G(x) omelchenko
#A = 1                       # De la distribución G(x)
#B = 0                       # De la distribución G(x) y G_local(x)

# Parámetros G_local(x) omelchenko ('omel_local')
#A = 1
#x0 = 0.01                      # Centro del coseno

# Parámetros G_asim_vecinos(x) tipo escalón
#a = 3*np.pi/5               # Ancho del escalón de G_asimetrica
#x0 = x0                     # Centro del escalón de G_asimétrica

# Parámetros G_gausiana(x) gausiana
#sigma = 0                   # Desviación estándar (ancho gausiana)
#x0 = x0                     # Centro de la gausiana



# Parametros generales para LEER DATOS de las comunidades por separado:
min_t = 0
max_t = 1500                 # t_max para la integración
dt = 0.1




# CREAR SIMULACIONES DE LAS QUIMERAS:
def ejecutar(x03, cond_inicial=True):
    '''
    Ejecuta simulaciones de las quimeras.
    
    Parametros
    ----------
    dt : [float]
         Paso de integración.
    '''
    global matriz_z, t_values, n_tiempos, label_ci, min_t, dt
    global N1, N2, max_t_coms, x0_3
    global A1, A2, A3, x0_1, x0_2, beta_1, beta_2, alpha_1, alpha_2
    global beta_3, alpha_3, epsilon
    global exp_neg_1, exp_pos_1, exp_neg_2, exp_pos_2, exp_neg_3, exp_pos_3
    global gamma_1, gamma_2, num_x, vec_x, dx_vec_x
    global nombre_carpeta
    global nombre_carpeta_png, nombre_carpeta_pdf, nombre_carpeta_svg
    
    vec_beta_1 = np.array([0.08, 0.11])    # Primera comunidad: alpha = {1.491, 1.461}
    vec_beta_2 = np.array([0.08, 0.11])    # Segunda comunidad: alpha = {1.491, 1.461}
    vec_beta_3 = np.array([0.08, 0.12, 0.15])    # Interaccion entre comunidades: alpha = {1.491, 1.451, 1.421}
    
    # gamma_1 (comunidad 1):
    gamma_1 = 0.01
    
    # gamma_2 (comunidad 2):
    vec_gamma_2 = [0.01]

    vec_x0_1 = np.array([0, 0.003, 0.005, 0.008])    # Primera comunidad
    vec_x0_2 = np.array([0, 0.003])    # Segunda comunidad

    # Definir los valores de las constantes de las comunidades y ejecutar:
    # Primera comunidad:
    N1 = 300
    A1 = 1

    # Segunda comunidad:
    N2 = 300
    A2 = 1

    # Parámetros G(x) omelchenko - INTERACCION ENTRE COMUNIDADES
    A3 = 1                       # De la distribución G(x)
    vec_epsilon = [0.01, 0.1, 1]
    x0_3 = x03
    
    # Autocalculables:
    num_x = N1 + N2
    vec_x = np.linspace(-np.pi, np.pi, N1)   # Arreglo de posiciones espaciales
    dx_vec_x = abs(vec_x[1]-vec_x[0])           # distancia entre el eje x
    
    # Tiempo maximo de la simulacion ENTRE comunidades:
    max_t_coms = 3000
    
    num_carpeta = -6

    # Ejecutar simulacion:
    for epsilon in vec_epsilon:
        num_carpeta += 1
        for gamma_2 in vec_gamma_2:
            num_carpeta += 1
            for beta_1 in vec_beta_1:
                alpha_1 = np.pi/2 - beta_1
                num_carpeta += 1
                
                # Exponenciales de alpha (phase lag parameter)
                exp_neg_1 = complex(np.cos(alpha_1), -np.sin(alpha_1))    # e^{-i*alpha}
                exp_pos_1 = complex(np.cos(alpha_1),  np.sin(alpha_1))    # e^{i*alpha}
                
                for beta_2 in vec_beta_2:
                    alpha_2 = np.pi/2 - beta_2
                    num_carpeta += 1
                    
                    exp_neg_2 = complex(np.cos(alpha_2), -np.sin(alpha_2))    # e^{-i*alpha}
                    exp_pos_2 = complex(np.cos(alpha_2),  np.sin(alpha_2))    # e^{i*alpha}
                    
                    for x0_1 in vec_x0_1:
                        num_carpeta += 1
                        
                        for x0_2 in vec_x0_2:
                            num_carpeta += 1
                            for beta_3 in vec_beta_3:
                                alpha_3 = np.pi/2 - beta_3
                                num_carpeta += 1
                                
                                # Crear carpetas para guardar datos de simulaciones:
                                nombre_carpeta = '{}/x03={}'.format(num_carpeta, x0_3)
                                nombre_carpeta_png = '{}/x03={}/png'.format(num_carpeta, x0_3)
                                nombre_carpeta_pdf = '{}/x03={}/pdf'.format(num_carpeta, x0_3)
                                nombre_carpeta_svg = '{}/x03={}/svg'.format(num_carpeta, x0_3)
                                
                                path = os.path.join(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta)
                                path_png = os.path.join(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_png)
                                path_pdf = os.path.join(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_pdf)
                                path_svg = os.path.join(NOMBRE_DIRECTORIO_TO_SAVE, nombre_carpeta_svg)
                                
                                os.makedirs(path, exist_ok=True)
                                os.makedirs(path_png, exist_ok=True)
                                os.makedirs(path_pdf, exist_ok=True)
                                os.makedirs(path_svg, exist_ok=True)
                                # --------------------------------------------------
                                
                                exp_neg_3 = complex(np.cos(alpha_3), -np.sin(alpha_3))    # e^{-i*alpha}
                                exp_pos_3 = complex(np.cos(alpha_3),  np.sin(alpha_3))    # e^{i*alpha}
                                
                                if x0_1 == 0.0:
                                    max_t1 = 1500
                                else:
                                    max_t1 = 3000
                                    
                                if x0_2 == 0.0:
                                    max_t2 = 1500
                                else:
                                    max_t2 = 3000
                                
                                matriz_z, t_values, n_tiempos, label_ci = evolucion_temporal_z(A1, A2, A3,
                                                                                               x0_1, x0_2, x0_3,
                                                                                               alpha_1, alpha_2,
                                                                                               alpha_3, epsilon,
                                                                                               max_t1, max_t2)
                                # Separo matriz_z por comunidades:
                                matriz_z1 = matriz_z[:, 0:N1]       # z1(x,t)
                                matriz_z2 = matriz_z[:, N1:]        # z2(x,t)
                                    
                                # Graficar |z| para ambas comunidades:
                                grafico_z(matriz_z1, gamma_1, alpha_1, A1, x0_1, 'Com 1')
                                grafico_z(matriz_z2, gamma_2, alpha_2, A2, x0_2, 'Com 2')

#ejecutar(x03=0)



""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~ PARÁMETROS ~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
# OBS: por como esta el codigo, N1 debe ser igual a N2 (numero de
# discretizaciones espaciales para cada comunidad de osciladores).


# La siguiente variable se debe actualizar con el directorio donde estan los datos:
# OBS: Se debe usar el caracter / para separar carpetas en el nombre del directorio:
#NOMBRE_DIRECTORIO_DATOS = 'D:/Proyectos/Jose Luis Lopez/Simulaciones-Kuramoto-Comunidades/DATOS simulaciones OA'
# ejemplo:
NOMBRE_DIRECTORIO_DATOS = 'E:/NETWORKS and DYNAMICS/Ott-Antonsen/Comunidades/DATOS simulaciones OA/Solo z0 t_f (alfas segun yo)'

# Este es el directorio donde se guardar los datos de las simulaciones:
NOMBRE_DIRECTORIO_TO_SAVE = 'E:/NETWORKS and DYNAMICS/Ott-Antonsen/Comunidades/OA/alfas segun yo'
#NOMBRE_DIRECTORIO_TO_SAVE = 'D:/Proyectos/Jose Luis Lopez/Simulaciones-OA-Comunidades'









