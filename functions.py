import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

columns = ['coordinate_x', 'coordinate_y', 'displacement_x', 'displacement_y', 'load_x', 'load_y']
data = [[0,0,0,0,np.nan,np.nan],
        [10,0,np.nan,np.nan,0,-100],
        [20,0,0,0,np.nan,np.nan],
        [5,10,np.nan,np.nan,0,0],
        [15,10,np.nan,np.nan,0,0]]
nodes = pd.DataFrame(data, columns = columns)

columns = ['start_node', 'end_node', 'area', 'Youngs_modulus', 'density']
data = [[0,1,2e-6,10e9,2700],
        [1,2,2e-6,10e9,2700],
        [0,3,2e-6,10e9,2700],
        [1,3,2e-6,10e9,2700],
        [1,4,2e-6,10e9,2700],
        [4,2,2e-6,10e9,2700],
        [3,4,2e-6,10e9,2700]]
elements = pd.DataFrame(data, columns = columns)

columns = ['start_node', 'end_node', 'area', 'Youngs_modulus', 'density']
data = [[0,1,2e-6,10e9,3500],
        [1,2,2e-6,10e9,3500],
        [0,3,2e-6,10e9,3500],
        [1,3,2e-6,10e9,3500],
        [1,4,2e-6,10e9,3500],
        [4,2,2e-6,10e9,3500],
        [3,4,2e-6,10e9,3500]]
elements2 = pd.DataFrame(data, columns = columns)


def element_stiffness(element):
  
  start = element['start_node']
  end = element['end_node']

  x_coordinate_start = nodes.loc[start, 'coordinate_x']
  y_coordinate_start = nodes.loc[start, 'coordinate_y']

  x_coordinate_end = nodes.loc[end, 'coordinate_x']
  y_coordinate_end = nodes.loc[end, 'coordinate_y']

  deltaX = x_coordinate_end - x_coordinate_start
  deltaY = y_coordinate_end - y_coordinate_start
  length = np.sqrt(deltaX**2+deltaY**2)

  c = deltaX/length
  s = deltaY/length

  rotation_matrix = np.array([[c*c, c*s, -c*c, -c*s],
                              [c*s, s*s, -c*s, -s*s],
                              [-c*c, -c*s, c*c, c*s],
                              [-c*s, -s*s, c*s, s*s]])
  stiffness_matrix = element['area']*element['Youngs_modulus']/length*rotation_matrix
  return length, stiffness_matrix

def element_mass(element):
  element_mass = element['density']*element['area']*element['length']
  values = np.empty(4)
  values.fill(element_mass/2)
  mass_matrix = np.diag(values)
  return mass_matrix

def global_stiffness(element):
  N = len(nodes)
  indices = np.arange(2*N)
  indices = indices.reshape(-1,2)
  K = np.zeros((2*N, 2*N))
  start = element['start_node']
  end = element['end_node']
  indices = np.hstack([indices[start], indices[end]])
  K[np.ix_(indices, indices)] = element['stiffness_matrix']
  return K

def global_mass(element):
  N = len(nodes)
  indices = np.arange(2*N)
  indices = indices.reshape(-1,2)
  M = np.zeros((2*N, 2*N))
  start = element['start_node']
  end = element['end_node']
  indices = np.hstack([indices[start], indices[end]])
  M[np.ix_(indices, indices)] = element['mass_matrix']
  return M

def partition(K, A, B):
  KAA = K[np.ix_(A, A)]
  KAB = K[np.ix_(A, B)]
  KBA = K[np.ix_(B, A)]
  KBB = K[np.ix_(B, B)]
  return KAA, KAB, KBA, KBB

def connectpoints(x,y,p1,p2,color=True,label=False):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    if color == True:
      if label == True:
        plt.plot([x1,x2],[y1,y2],'--or', label='deformed')
      else:
        plt.plot([x1,x2],[y1,y2],'--or')
    else:
      if label == True:
        plt.plot([x1,x2],[y1,y2],'--ob', label='original')
      else:
        plt.plot([x1,x2],[y1,y2],'--ob')

def modal_analysis(M, K):
  A = np.dot(np.linalg.inv(M),K)
  eigvals, eigvecs = np.linalg.eig(A)
  return np.sqrt(eigvals), eigvecs

def Newmark_Beta(M, K, f, a, a_dot, a_ddot, deltaT, T, beta=1/4, gamma=1/2):
  M_new = 1/(beta*deltaT*deltaT)*M+K
  for i in range(T-1):
    expression = 1/(beta*deltaT*deltaT)*a[:,i]+1/(beta*deltaT)*a_dot[:,i]+1/(2*beta)*a_ddot[:,i]-a_ddot[:,i]
    F_new = np.dot(M, expression) + f[:,i+1]
    a_new = np.dot(np.linalg.inv(M_new), F_new)
    a = np.append(a, a_new.reshape(np.shape(M)[0],1), axis=1)
    a_ddot_new = 1/(beta*deltaT*deltaT)*(a[:,i+1]-a[:,i])-1/(beta*deltaT)*a_dot[:,i]-1/(2*beta)*a_ddot[:,i]+a_ddot[:,i]
    a_ddot = np.append(a_ddot, a_ddot_new.reshape(np.shape(M)[0],1), axis=1)
    a_dot_new = a_dot[:,i]+deltaT*(1-gamma)*a_ddot[:,i]+deltaT*gamma*a_ddot_new
    a_dot = np.append(a_dot, a_dot_new.reshape(np.shape(M)[0],1), axis=1)
  return a, a_dot, a_ddot

def Alpha(M, K, d, d_dot, v, v_dot, dt, t, rho_inf = 0.5):
  alpha_f = 1/(1+rho_inf)
  alpha_m = (3-rho_inf)/(2*(1+rho_inf))
  gamma = 1/2+alpha_m-alpha_f

  K_bar = alpha_m**2/(alpha_f*gamma**2*dt**2)*M+alpha_f*K
  for i in range(t-1):
    F_bar = -(1-alpha_m)*np.dot(M,v_dot[:,i])-(1-alpha_f)*np.dot(K,d[:,i])+alpha_m*np.dot(M, (alpha_m/(alpha_f*gamma**2*dt**2)*d[:,i]+1/(alpha_f*gamma*dt)*v[:,i]-(gamma-1)/gamma*v_dot[:,i]-(gamma-alpha_m)/(alpha_f*gamma**2*dt)*d_dot[:,i]))
    d_new = np.dot(np.linalg.inv(K_bar), F_bar)
    d = np.append(d, d_new.reshape(np.shape(M)[0],1), axis=1)
    d_dot_new = 1/(gamma*dt)*(d_new-d[:,i])+(gamma-1)/gamma*d_dot[:,i]
    d_dot = np.append(d_dot, d_dot_new.reshape(np.shape(M)[0],1), axis=1)
    v_new = alpha_m/(alpha_f*gamma*dt)*(d_new-d[:,i])+(gamma-alpha_m)/(gamma*alpha_f)*d_dot[:,i]+(alpha_f-1)/alpha_f*v[:,i]
    v = np.append(v, v_new.reshape(np.shape(M)[0],1), axis=1)
    v_dot_new = alpha_m/(alpha_f*gamma**2*dt**2)*(d_new-d[:,i])-1/(alpha_f*gamma*dt)*v[:,i]+(gamma-1)/gamma*v_dot[:,i]+(gamma-alpha_m)/(alpha_f*gamma**2*dt)*d_dot[:,i]
    v_dot = np.append(v_dot, v_dot_new.reshape(np.shape(M)[0],1), axis=1)
  return d, d_dot, v, v_dot

def Runge_Kutta(M, K, T_range, x_0):
  def func(t, x):
    I = np.identity(M.shape[0])
    zeros = np.zeros((M.shape[0], M.shape[0]))
    A_1 = np.vstack((I, zeros))
    A_2 = np.vstack((zeros, M))
    A = np.hstack((A_1, A_2))
    B_1 = np.vstack((zeros, -K))
    B_2 = np.vstack((I, zeros))
    B = np.hstack((B_1, B_2))
    return np.dot(np.dot(np.linalg.inv(A), B), x)
  sol = scipy.integrate.solve_ivp(func, T_range, x_0)
  return sol.t, sol.y

def damping_factors(frequencies, freq_1, freq_2, damp_1, damp_2, M, K):
  alpha = 2*freq_1*freq_2*(damp_1*freq_2-damp_2*freq_1)/(freq_2**2-freq_1**2)
  beta = 2*(damp_2*freq_2-damp_1*freq_1)/(freq_2**2-freq_1**2)
  C = alpha*M + beta*K
  
  zetas = []
  for i in range(len(frequencies)):
    zetas.append(alpha/(2*frequencies[i])+beta/2*frequencies[i])
  return C, zetas

def Newmark_Beta_damping(M, C, K, f, a, a_dot, a_ddot, deltaT, T, beta=1/4, gamma=1/2):
  M_new = 1/(beta*deltaT*deltaT)*(M+gamma*deltaT*C)+K
  for i in range(T-1):
    expression = 1/(beta*deltaT*deltaT)*a[:,i]+1/(beta*deltaT)*a_dot[:,i]+1/(2*beta)*a_ddot[:,i]-a_ddot[:,i]
    F_new = np.dot((M+gamma*deltaT*C), expression)-np.dot(C, (a_dot[:,i]+(1-gamma)*deltaT*a_ddot[:,i]))+f[:,i+1]
    a_new = np.dot(np.linalg.inv(M_new), F_new)
    a = np.append(a, a_new.reshape(np.shape(M)[0],1), axis=1)
    a_ddot_new = 1/(beta*deltaT*deltaT)*(a[:,i+1]-a[:,i])-1/(beta*deltaT)*a_dot[:,i]-1/(2*beta)*a_ddot[:,i]+a_ddot[:,i]
    a_ddot = np.append(a_ddot, a_ddot_new.reshape(np.shape(M)[0],1), axis=1)
    a_dot_new = a_dot[:,i]+deltaT*(1-gamma)*a_ddot[:,i]+deltaT*gamma*a_ddot_new
    a_dot = np.append(a_dot, a_dot_new.reshape(np.shape(M)[0],1), axis=1)
  return a, a_dot, a_ddot

def Alpha_damping(M, C, K, d, d_dot, v, v_dot, dt, t, rho_inf = 0.5):
  alpha_f = 1/(1+rho_inf)
  alpha_m = (3-rho_inf)/(2*(1+rho_inf))
  gamma = 1/2+alpha_m-alpha_f

  K_bar = alpha_m**2/(alpha_f*gamma**2*dt**2)*M+alpha_m/(gamma*dt)*C+alpha_f*K
  for i in range(t-1):
    F_bar = -(1-alpha_m)*np.dot(M,v_dot[:,i])-(1-alpha_f)*np.dot(C,v[:,i])-(1-alpha_f)*np.dot(K,d[:,i])+alpha_f*np.dot(C, (alpha_m/(alpha_f*gamma*dt)*d[:,i]-(gamma-alpha_m)/(gamma*alpha_f)*d_dot[:,i]-(alpha_f-1)/alpha_f*v[:,i]))+alpha_m*np.dot(M, (alpha_m/(alpha_f*gamma**2*dt**2)*d[:,i]+1/(alpha_f*gamma*dt)*v[:,i]-(gamma-1)/gamma*v_dot[:,i]-(gamma-alpha_m)/(alpha_f*gamma**2*dt)*d_dot[:,i]))
    d_new = np.dot(np.linalg.inv(K_bar), F_bar)
    d = np.append(d, d_new.reshape(np.shape(M)[0],1), axis=1)
    d_dot_new = 1/(gamma*dt)*(d_new-d[:,i])+(gamma-1)/gamma*d_dot[:,i]
    d_dot = np.append(d_dot, d_dot_new.reshape(np.shape(M)[0],1), axis=1)
    v_new = alpha_m/(alpha_f*gamma*dt)*(d_new-d[:,i])+(gamma-alpha_m)/(gamma*alpha_f)*d_dot[:,i]+(alpha_f-1)/alpha_f*v[:,i]
    v = np.append(v, v_new.reshape(np.shape(M)[0],1), axis=1)
    v_dot_new = alpha_m/(alpha_f*gamma**2*dt**2)*(d_new-d[:,i])-1/(alpha_f*gamma*dt)*v[:,i]+(gamma-1)/gamma*v_dot[:,i]+(gamma-alpha_m)/(alpha_f*gamma**2*dt)*d_dot[:,i]
    v_dot = np.append(v_dot, v_dot_new.reshape(np.shape(M)[0],1), axis=1)
  return d, d_dot, v, v_dot

def Runge_Kutta_damping(M, C, K, T_range, x_0):
  def func(t, x):
    I = np.identity(M.shape[0])
    zeros = np.zeros((M.shape[0], M.shape[0]))
    A_1 = np.vstack((I, zeros))
    A_2 = np.vstack((zeros, M))
    A = np.hstack((A_1, A_2))
    B_1 = np.vstack((zeros, -K))
    B_2 = np.vstack((I, -C))
    B = np.hstack((B_1, B_2))
    return np.dot(np.dot(np.linalg.inv(A), B), x)
  sol = scipy.integrate.solve_ivp(func, T_range, x_0)
  return sol.t, sol.y

def transition_matrix_Newmark_Beta(M, K, deltaT, beta=1/4, gamma=1/2):
  M_new = 1/(beta*deltaT*deltaT)*M+K
  M_hat = np.dot(np.linalg.inv(M_new), M)

  A11 = 1/(beta*deltaT**2)*M_hat
  A12 = 1/(beta*deltaT)*M_hat
  A13 = (1/(2*beta)-1)*M_hat
  A21 = gamma/(beta*deltaT)*(1/(beta*deltaT**2)*M_hat-np.identity(np.shape(M_hat)[0]))
  A22 = gamma/beta*(1/(beta*deltaT**2)*M_hat-(1-beta/gamma)*np.identity(np.shape(M_hat)[0]))
  A23 = deltaT*gamma*(1/(2*beta)-1)*(1/(beta*deltaT**2)*M_hat-(1-(1-gamma)/(gamma*(1/(2*beta)-1)))*np.identity(np.shape(M_hat)[0]))
  A31 = 1/(beta*deltaT**2)*(1/(beta*deltaT**2)*M_hat-np.identity(np.shape(M_hat)[0]))
  A32 = 1/(beta*deltaT)*(1/(beta*deltaT**2)*M_hat-np.identity(np.shape(M_hat)[0]))
  A33 = (1/(2*beta)-1)*(1/(beta*deltaT**2)*M_hat-np.identity(np.shape(M_hat)[0]))

  A1 = np.hstack((A11, A12, A13))
  A2 = np.hstack((A21, A22, A23))
  A3 = np.hstack((A31, A32, A33))

  A = np.vstack((A1, A2, A3))
  return A

def force_matrix_Newmark_Beta(M, K, deltaT, beta=1/4, gamma=1/2):
  M_new = 1/(beta*deltaT*deltaT)*M+K
  M_hat = np.dot(np.linalg.inv(M_new), M)

  B1 = np.linalg.inv(M_hat)
  B2 = gamma/(beta*deltaT)*np.linalg.inv(M_hat)
  B3 = gamma/(beta*deltaT**2)*np.linalg.inv(M_hat)

  B = np.vstack((B1, B2, B3))
  return B

def generate_pseudo_data(x, T, factor):
  y = np.zeros(T+1)
  for i in range(int(np.shape(x)[0]/3)):
    y_i = x[i,:] + np.random.normal(0, factor*np.std(x[i,:]), T+1)
    y = np.vstack((y, y_i))
  return np.delete(y, (0), axis=0)

def Kalman_filter(A, B, x, f, y, sigma_0, sigma_r, sigma_q, T):
  Q = sigma_q*np.identity(np.shape(A)[0])
  R = sigma_r*np.identity(np.shape(y)[0])
  H = np.hstack((np.identity(int(np.shape(A)[0]/3)), np.zeros((int(np.shape(A)[0]/3),int(np.shape(A)[0]/3))), np.zeros((int(np.shape(A)[0]/3),int(np.shape(A)[0]/3)))))
  
  P = [sigma_0*np.identity(np.shape(A)[0])]

  for i in range(T-1):
    x_predict = np.dot(A, x[:,i]) + np.dot(B, f[:,i+1])
    P_predict = np.dot(A, np.dot(P[i], np.transpose(A))) + Q
    V = y[:,i] - np.dot(H, x_predict)
    S = np.dot(H, np.dot(P_predict, np.transpose(H))) + sigma_r
    K = np.dot(P_predict, np.dot(np.transpose(H), np.linalg.inv(S)))
    x_estimate = x_predict + np.dot(K, V)
    P_estimate = P_predict - np.dot(K, np.dot(S, np.transpose(K)))

    x = np.append(x, x_estimate.reshape(np.shape(A)[0],1), axis=1)
    P.append(P_estimate)

  return x

