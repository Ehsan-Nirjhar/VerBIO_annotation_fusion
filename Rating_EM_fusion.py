import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter

def make_A(vec, n):
    m = len(vec)
    X = np.zeros((m,n))
    for i in range(m):
        temp_vec = np.zeros(n)
        num_elements = min(i+1,n)
        start_idx = max(i-num_elements+1, 0)
        end_idx = i+1
        element_vec = vec[start_idx:end_idx]
        start_idx = n-num_elements
        end_idx = n+2
        temp_vec[start_idx:end_idx] = element_vec
        X[i,:] = temp_vec
        
    X = np.fliplr(X)     #### My addition
    return X

def make_Fn(d_n, T):
    
    d_n = np.flip(d_n)   #### My addition
    
    w = len(d_n)
    F_n = np.zeros((T,T))
    for i in range(T):
        
        temp_vec = np.zeros(T);
        num_elements = min(i+1,w)
        start_idx = w-num_elements
        end_idx = w
        element_vec = d_n[start_idx:end_idx]
        start_idx = max(i-w+1,0)
        end_idx = i+1
        temp_vec[start_idx:end_idx] = element_vec
        F_n[i,:] = temp_vec
        
    return F_n

def calc_log_likelihood(dataset_dict, a_star, F_weight, F_bias, l2_reg):
    

    loss_val = 0
    for pid in data_dict.keys():
        a_star_arr = a_star[pid]
        annotator_id_list = list(data_dict[pid].keys())
        
        sum_indiv = 0
        for n in range(len(annotator_id_list)):
            a_n = data_dict[pid][annotator_id_list[n]]
            T = len(a_n)
            d_n = F_weight[:,n]
            d_b = F_bias[n]
            F_n = make_Fn(d_n , T)
            bias_vec = d_b * np.ones(T)
            err_vec = a_n - np.matmul(F_n, a_star_arr) - bias_vec
            if l2_reg:
                l2_vec = np.matmul(np.transpose(d_n), d_n)  
                sum_indiv = sum_indiv + np.matmul(np.transpose(err_vec), err_vec) + l2_vec
            else:
                sum_indiv = sum_indiv + np.matmul(np.transpose(err_vec), err_vec)
            
            
        loss_val = loss_val + sum_indiv
#    print(loss_val)   
    return loss_val

ic=0
l2_reg = True
annotator_id_list = ['R1', 'R2', 'R4', 'R5']
session = 'PRE'
data_dir = f'../Annotation files/processed_short'
window_size = 4
num_annotator = 4
max_iter = 100
tol_val = 0.0005
eps = 1e-15
#### Creates dataset dictionary
data_dict = {}
a_star = {}
for pid in range(1,74):

    
    r1_file = f'{data_dir}/R1/{session}/{session}_P'+ str(pid).zfill(3) +'_annotation_R1.xlsx'
    r2_file = f'{data_dir}/R2/{session}/{session}_P'+ str(pid).zfill(3) +'_annotation_R2.xlsx'
    r4_file = f'{data_dir}/R4/{session}/{session}_P'+ str(pid).zfill(3) +'_annotation_R4.xlsx'
    r5_file = f'{data_dir}/R5/{session}/{session}_P'+ str(pid).zfill(3) +'_annotation_R5.xlsx'
    
    subject_id = f'P{str(pid).zfill(3)}'
    data_dict[subject_id] = {}
    a_star[subject_id] = []
    try:
        t = pd.read_excel(r1_file)['Time (seconds)'].to_numpy()
        data_dict[subject_id]['R1'] = pd.read_excel(r1_file)['Rating'].to_numpy()
        data_dict[subject_id]['R2'] = pd.read_excel(r2_file)['Rating'].to_numpy()
        data_dict[subject_id]['R4'] = pd.read_excel(r4_file)['Rating'].to_numpy()
        data_dict[subject_id]['R5'] = pd.read_excel(r5_file)['Rating'].to_numpy()
        
    except:
        del data_dict[subject_id]
        del a_star[subject_id]
        continue

for pid in data_dict.keys():
    r1 = data_dict[pid]['R1']
    r2 = data_dict[pid]['R2']
    r4 = data_dict[pid]['R4']
    r5 = data_dict[pid]['R5']
    t = len(r1)
    a_star[pid] = (r1+r2+r4+r5)/4
    # a_star[pid] = np.random.choice([1,2,3,4,5], t)
    # a_star[pid] = np.random.rand(t)
    

### Filter weight bias initialization
# F_weight = np.random.rand(window_size, num_annotator)
F_weight = np.zeros((window_size, num_annotator))
F_weight[0, :]=1
# F_bias = np.random.rand(num_annotator)
F_bias = np.array([0.01, 0.01, 0.01, 0.01])

#### Maximization Step

# for n in range(len(annotator_id_list)):
#     old_d_n = F_weight[:,n]
#     old_d_b = F_bias[n]
    
#     new_d_b = 0
#     new_d_n = np.zeros((window_size,))
    
#     sum_left = np.zeros((window_size,window_size))
#     sum_right = np.zeros((window_size,))
#     T_sum = 0
    
#     for pid in data_dict.keys():
#         a_star_arr = a_star[pid]
#         a_n = data_dict[pid][annotator_id_list[n]]
#         T = len(a_n)
#         bias_vec = old_d_b * np.ones(T)
#         F_n = make_Fn(old_d_n , T)
#         A = make_A(a_star_arr, window_size)
        
#         ###For bias
#         new_d_b = new_d_b + np.matmul(np.transpose(np.ones(T)), a_n) - np.matmul(np.transpose(np.ones(T)), np.matmul(F_n, a_star_arr))
#         T_sum = T_sum + T
        
#         ### For weights
#         sum_left = sum_left + np.matmul(np.transpose(A), A)
#         sum_right = sum_right + np.matmul(np.transpose(A), a_n) - np.matmul(np.transpose(A), bias_vec)
        
#         if l2_reg:
#             sum_left = sum_left + np.matmul(np.transpose(A), A) + np.identity(window_size)
#         else:
#             sum_left = sum_left + np.matmul(np.transpose(A), A)
#         sum_right = sum_right + np.matmul(np.transpose(A), a_n) - np.matmul(np.transpose(A), bias_vec)
        
#     if math.fabs(1.0/np.linalg.cond(sum_left)) < eps:
#         ic+=1
#         sum_left += np.identity(window_size)
        

        
#     F_bias[n] = new_d_b/T_sum
    
#     F_weight[:,n] = np.matmul(np.linalg.inv(sum_left), sum_right)


loss_arr = []
end_loop = False

for iterval in range(max_iter):
##### Expectation Step
    for pid in data_dict.keys():
        annotator_id_list = list(data_dict[pid].keys())
        T = len(data_dict[pid][annotator_id_list[0]])
        sum_left = np.zeros((T,T))
        sum_right = np.zeros((T,))
        
        for n in range(len(annotator_id_list)):
            a_n = data_dict[pid][annotator_id_list[n]]
            d_n = F_weight[:,n]
            d_b = F_bias[n]
            F_n = make_Fn(d_n , T)
            
            bias_vec = d_b * np.ones(T)
            sum_left = sum_left + np.matmul(np.transpose(F_n), F_n)
            sum_right = sum_right + np.matmul(np.transpose(F_n), a_n) - np.matmul(np.transpose(F_n), bias_vec)
        
        if math.fabs(1.0/np.linalg.cond(sum_left)) < eps:
            ic+=1
            sum_left += np.identity(T)
        a_star[pid] = savgol_filter(np.matmul(np.linalg.inv(sum_left), sum_right), 5, 3)
    #    break
    
    
    ###### Calculate Log-likelihood
    loss_val = calc_log_likelihood(data_dict, a_star, F_weight, F_bias, l2_reg)
    # loss_arr.append(loss_val)
    
    #### Maximization Step
    
    for n in range(len(annotator_id_list)):
        old_d_n = F_weight[:,n]
        old_d_b = F_bias[n]
        
        new_d_b = 0
        new_d_n = np.zeros((window_size,))
        
        sum_left = np.zeros((window_size,window_size))
        sum_right = np.zeros((window_size,))
        T_sum = 0
        for pid in data_dict.keys():
            a_star_arr = a_star[pid]
            a_n = data_dict[pid][annotator_id_list[n]]
            T = len(a_n)
            bias_vec = old_d_b * np.ones(T)
            F_n = make_Fn(old_d_n , T)
            A = make_A(a_star_arr, window_size)
            
            ###For bias
            new_d_b = new_d_b + np.matmul(np.transpose(np.ones(T)), a_n) - np.matmul(np.transpose(np.ones(T)), np.matmul(F_n, a_star_arr))
            T_sum = T_sum + T
            
            ### For weights
            if l2_reg:
                sum_left = sum_left + np.matmul(np.transpose(A), A) + np.identity(window_size)
            else:
                sum_left = sum_left + np.matmul(np.transpose(A), A)
            sum_right = sum_right + np.matmul(np.transpose(A), a_n) - np.matmul(np.transpose(A), bias_vec)
            
        if math.fabs(1.0/np.linalg.cond(sum_left)) < eps:
            ic+=1
            sum_left += np.identity(window_size)
            
        F_bias[n] = new_d_b/T_sum

        F_weight[:,n] = np.matmul(np.linalg.inv(sum_left), sum_right)
        
    
    ##### Calculate Log-likelihood
    loss_val = calc_log_likelihood(data_dict, a_star, F_weight, F_bias, l2_reg)
    loss_arr.append(loss_val)
    
    if iterval == 0:
        old_loss = loss_val
    else:
        new_loss = loss_val
        
        if abs(new_loss - old_loss)/old_loss < tol_val:
            end_loop = True
        old_loss = new_loss
    
    
    if end_loop:
        break
        
            

plt.plot(loss_arr)
plt.ylabel('Loss')
plt.xlabel('Iteration')
# plt.savefig('loss.png', bbox_inches = 'tight')

for pid in data_dict.keys():
    a_star_arr = a_star[pid]
    r1 = data_dict[pid]['R1']
    r2 = data_dict[pid]['R2']
    r4 = data_dict[pid]['R4']
    r5 = data_dict[pid]['R5']
    
    r_mean = (r1+r2+r4+r5)/4
    t = np.arange(len(r_mean))
#    plt.plot(t,a_star_arr)
#    plt.show()
#    plt.plot(t, r_mean)
#    plt.show()
#    break

    fig = plt.figure()
    ax0 = fig.add_subplot(2, 1,1)
        
        
#    plt.xlabel('Time (seconds)')
#    plt.ylabel('Rating')
    
    ax0.plot(t, r_mean,linewidth=4)
    ax0.legend(['Average Rating'])   
    ax1 = fig.add_subplot(2, 1,2)
    fig.suptitle(pid, fontsize=16)

    ax1.plot(t, a_star_arr,linewidth=4)

    
    
    ax0.set_xlabel('Time (seconds)')
    ax0.set_ylabel('Rating')
####       plt.plot(t,r1,t,r2, t, r_dtw, t,r_sdtw, linewidth=4)
    ax1.legend(['Calculated Rating'])
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Rating')
    
    plt.show()
#    if pid== 'P003':
#        break
        
        
        
        
        
#    nm = f'./EM_output/{pid}.png'
#    fig.savefig(nm,bbox_inches = 'tight')
#    fig.clf()

for pid in a_star.keys():
    em_res = a_star[pid]
    op_df = pd.DataFrame(em_res, columns = ['EM'])
    op_df.to_excel(f'{pid}_EM.xlsx', index=False)