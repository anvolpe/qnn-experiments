import csv
import gc
import json
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane

from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from victor_thesis_experiments_main import *
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from sgd_for_scipy import *
from jax import jacrev
import os
import pandas as pd
from scipy.optimize import minimize
import re

num_layers = 1
num_qubits = 2
dimensions = 6
max_iters = [100,500,1000]
tols = [1e-5, 1e-10]
bounds = []
learning_rates = [0.01, 0.001, 0.0001]

def nelder_mead_experiment(objective,initial_param_values):
    results = {}
    run_n = 0
    for max_iter in max_iters:
        for fatol in tols:
            for xatol in tols:
                start = time.time()
                res = minimize(objective, initial_param_values, method="Nelder-Mead", 
                        options={"maxiter": max_iter, "fatol":fatol, "xatol":xatol})
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "fatol":fatol, "xatol":xatol, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                run_n += 1
    return results

def cobyla_experiment(objective,initial_param_values):
    results = {}
    run_n = 0
    for max_iter in max_iters:
        for tol in tols:
            for catol in tols:
                start = time.time()
                res = minimize(objective, initial_param_values, method="COBYLA", 
                        options={"maxiter": max_iter, "tol":tol, "catol":catol})
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "tol":tol, "catol":catol, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                run_n += 1
    return results

def bfgs_experiment(objective,initial_param_values):
    results = {}
    run_n = 0
    for max_iter in max_iters:
        for gtol in tols:
            for xrtol in tols:
                for eps in tols:
                    start = time.time()
                    res = minimize(objective, initial_param_values, method="BFGS", 
                            options={"maxiter": max_iter, "gtol":gtol, "xrtol":xrtol, "eps":eps})
                    duration = time.time() - start
                    # fill results dict
                    # specifications of this optimizer run
                    results[run_n] = {"maxiter": max_iter, "gtol":gtol, "xrtol":xrtol, "eps":eps, "duration":duration}
                    # result info
                    for attribute in res.keys():
                        results[run_n][attribute] = str(res[attribute])
                    run_n += 1
    return results

def powell_experiment(objective,initial_param_values):
    results = {}
    run_n = 0
    for max_iter in max_iters:
        for ftol in tols:
            for xtol in tols:
                start = time.time()
                res = minimize(objective, initial_param_values, method="Powell", 
                        options={"maxiter": max_iter, "ftol":ftol, "xtol":xtol})
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "ftol":ftol, "xtol":xtol, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                run_n += 1
    return results

def slsqp_experiment(objective,initial_param_values):
    results = {}
    run_n = 0
    for max_iter in max_iters:
        for ftol in tols:
            for eps in tols:
                start = time.time()
                res = minimize(objective, initial_param_values, method="SLSQP", 
                        options={"maxiter": max_iter, "ftol":ftol, "eps":eps})
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "ftol":ftol, "eps":eps, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                run_n += 1
    return results

def sgd_experiment(objective,initial_param_values,opt):
    results = {}
    run_n = 0
    for max_iter in max_iters:
        for learning_rate in learning_rates:
            for eps in tols:
                start = time.time()
                res = minimize(objective, initial_param_values, method=opt, 
                        options={"maxiter": max_iter, "learning_rate":learning_rate, "eps":eps})
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "learning_rate":learning_rate, "eps":eps, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                run_n += 1
    return results

# gibt nicht direkt sowas wie ftol und eps
def dual_annealing_experiment(objective,bounds):
    results = {}
    run_n = 0
    for max_iter in max_iters:
        #for tol in tols:
        #for catol in tols:
                start = time.time()
                res = dual_annealing(objective, bounds, maxiter=max_iter)
                duration = time.time() - start
                # fill results dict
                # specifications of this optimizer run
                results[run_n] = {"maxiter": max_iter, "duration":duration}
                # result info
                for attribute in res.keys():
                    results[run_n][attribute] = str(res[attribute])
                run_n += 1
    return results

def single_config_experiments(conf_id, data_type, num_data_points, s_rank, unitary, data_points):
    # prepare csv file for experiment results
    os.makedirs("experimental_results/results/optimizer_results", exist_ok=True)
    file = open(f"experimental_results/results/optimizer_results/conf_{conf_id}_opt.csv", mode="w")
    writer = csv.writer(file)
    writer.writerow(["Optimizer", "Result", "Duration"])
    # TODO: Infos zu Config speichern (alles was oben als Argument übergeben wird)
    # evtl json statt csv? oder txt?
    
    # specifications of qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
        
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()

    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return torch.tensor(cost.item())
    # TODO: verschiedene inital_param_values ausprobieren und avg bilden? (zuerst schauen wie lang es dauert)
    # same initial_param_values for all optimizers
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    initial_param_values_tensor = torch.tensor(initial_param_values) 

    # alle optimierer experiments laufen lassen
    # return ist jeweils result-dict (das auch alle Spezifikationen enthält?)
    # das in der Datei conf_[conf_id]_opt.csv speichern

        
def single_optimizer_experiment(conf_id, databatch_id, data_type, num_data_points, s_rank, unitary, data_points):
    result_dict = {}
    data_points_string = (
        np.array2string(data_points.numpy(), separator=",")
        .replace("\n", "")
        .replace(" ", "")
    )
    result_dict["databatch"] = data_points_string
    
    # specifications of qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return torch.tensor(cost.item())

    # verschiedene inital_param_values ausprobieren und avg bilden? 
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    initial_param_values_tensor = torch.tensor(initial_param_values)

    # run optimizer experiments
    sgd_optimizers = [sgd, rmsprop, adam]
    optimizers = [nelder_mead_experiment, bfgs_experiment, cobyla_experiment, powell_experiment, slsqp_experiment, sgd_experiment]

    # TODO: ProcessPoolExecutor: funktioniert nicht, weil pickle verwendet wird und objective eine lokal definierte Funktion ist 
    # (AttributeError: Can't pickle local object 'test_experiment.<locals>.objective')
    # Multiprocessing (??) könnte funktionieren. Oder eigene Klasse??
    # with ProcessPoolExecutor(cpu_count()) as exe:
    for opt in optimizers:
        if opt == sgd_experiment:
            for variant in sgd_optimizers:
                #future = exe.submit(sgd_experiment, objective, initial_param_values_tensor, variant)
                result = sgd_experiment(objective,initial_param_values,variant)
                opt_name = variant.__name__
                #result_dict[opt_name] = future.result()
                result_dict[opt_name] = result
        else:
            #future = exe.submit(opt, objective, initial_param_values)
            result = opt(objective,initial_param_values)
            opt_name = opt.__name__.removesuffix('_experiment')
            #result_dict[opt_name] = future.result()
            result_dict[opt_name] = result

    return result_dict

    

def run_all_optimizer_experiments():
    filename = "Code/entangled_qnn_training-main/experimental_results/configs/configurations_16_6_4_10_13_3_14.txt"
    file = open(filename, 'r')
    Lines = file.readlines()
    n = 0
    conf_id = 0
    databatch_id = 0
    data_type = ""
    num_data_points = 0
    s_rank = 0
    unitary = []
    databatches = []
    result_dict = {}

    for line in Lines:
        if(line.strip() == "---"): # config has been fully read, run optimizer experiments for each data_point-tensor (5)
            # setup dictionary for dumping info into json file later
            result_dict = {"conf_id":conf_id, "data_type":data_type, "num_data_points":num_data_points, "s_rank":s_rank}
            unitary_string = (
                np.array2string(unitary.numpy(), separator=",")
                .replace("\n", "")
                .replace(" ", "")
            )
            result_dict["unitary"] = unitary_string
            n = 0
            #databatch = databatches[n]
            #print(len(databatches))
            for i in range(len(databatches)): 
                data_points = databatches[i]  
                dict = single_optimizer_experiment(conf_id, i, data_type, num_data_points, s_rank, unitary, data_points)
                databatch_key = f"databatch_{databatch_id}"
                result_dict[databatch_key] = dict
                #print(conf_id, data_type, num_data_points, s_rank)
            #write results to json file
            os.makedirs("experimental_results/results/optimizer_results", exist_ok=True)
            file = open(f"experimental_results/results/optimizer_results/conf_{conf_id}_opt.json", mode="w")
            json.dump(result_dict, file)
            databatches = []
            unitary = []
            n += 1

        else:
            var, val = line.split("=")
            if(var == "conf_id"): conf_id = int(val) 
            elif(var == "data_type"): data_type = val # random, orthogonal, non_lin_ind, var_s_rank
            elif(var == "num_data_points"): num_data_points = int(val) 
            elif(var == "s_rank"): s_rank = int(s_rank) # Schmidt-Rank
            elif(var == "unitary"): 
                val,_ = re.subn('\[|\]|\\n', '', val) 
                unitary = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4))#unitary: 4x4 tensor
            elif(var.startswith("data_batch_")): 
                val,_ = re.subn('\[|\]|\\n', '', val)
                #print(torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4)))
                databatches.append(torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4))) #data_points: 1x4x4 tensor

def test_experiment():
    # setup qnn configuration
    conf_id = 0
    data_type = "random" # random, orthogonal, non_lin_ind, var_s_rank
    num_data_points = 1
    s_rank = 1 # Schmidt-Rank
    var = "data_batch_0"
    databatch_id = int(var.strip("datbch_"))

    # setup dictionary for dumping info into json file later
    result_dict = {"conf_id":conf_id, "data_type":data_type, "num_data_points":num_data_points, "s_rank":s_rank}
    val = "[[-0.45070455-0.02711853j,0.78437395-0.06613086j,0.06358678+0.19963393j,-0.07343613+0.35668523j],[-0.01890143-0.03813363j,0.32408202+0.25557629j,0.05872864-0.68979805j,0.55466693-0.20227297j],[-0.11215405+0.64023111j,-0.13344055+0.29565494j,-0.49012687-0.19046288j,-0.04241254+0.44046348j],[0.55771659+0.24656916j,0.31851997-0.05798805j,0.28761525-0.34294258j,-0.56718418+0.03616933j]]"
    result_dict["unitary"] = val#.strip("\"")
    val,_ = re.subn('\[|\]|\\n', '', val)
    unitary = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4))#unitary: 4x4 tensor
    val = '[[[0.09314128-0.12946863j,0.39382838-0.19799267j,0.05133879+0.12112185j,0.08106995-0.04021906j],[0.07622026-0.09754417j,0.31152873-0.14143589j,0.03608905+0.09551662j,0.06411194-0.02869752j],[0.11804856-0.19626647j,0.54031288-0.32976236j,0.08774511+0.16729872j,0.1112873-0.06711204j],[-0.01827577+0.10086995j,-0.17383409+0.2237231j,-0.06326177-0.05610261j,-0.03593256+0.04574145j]]]'
    result_dict[var] = {}
    result_dict[var]["databatch"] = val#.strip("\"")
    val,_ = re.subn('\[|\]|\\n', '', val) 
    data_points = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4)) #data_points: 1x4x4 tensor


    # setup qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return cost.item()
    results = {}
    # verschiedene inital_param_values ausprobieren und avg bilden? 
    # TODO: gleich für alle Konfigs oder nur gleich für eine Konfig aber da für alle Optmimierer?
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    initial_param_values_tensor = torch.tensor(initial_param_values)


    #writer = csv.writer(file)
    #writer.writerow(["Optimizer", "Result", "Duration"])

    # all results of all optimizer experiments for this config
    results = {}

    # run optimizer experiments
    sgd_optimizers = [sgd, rmsprop, adam]
    #optimizers = [nelder_mead_experiment, bfgs_experiment, cobyla_experiment, powell_experiment, slsqp_experiment, sgd_experiment]
    optimizers = [cobyla_experiment, sgd_experiment]

    #with ProcessPoolExecutor(cpu_count()) as exe:
    for opt in optimizers:
        if opt == sgd_experiment:
            for variant in sgd_optimizers:
                #future = exe.submit(sgd_experiment, objective, initial_param_values_tensor, variant)
                result = sgd_experiment(objective,initial_param_values,variant)
                opt_name = variant.__name__
                #result_dict[var][opt_name] = future.result()
                result_dict[var][opt_name] = result
        else:
            #future = exe.submit(opt, objective, initial_param_values)
            result = opt(objective,initial_param_values)
            opt_name = opt.__name__.removesuffix('_experiment')
            #result_dict[var][opt_name] = future.result()
            result_dict[var][opt_name] = result

    '''
    result = nelder_mead_experiment(objective, initial_param_values)
    result_dict[var]['Nelder-Mead'] = result
    result = bfgs_experiment(objective, initial_param_values)
    result_dict[var]["BFGS"] = result
    result = cobyla_experiment(objective, initial_param_values)
    result_dict[var]["COBYLA"] = result
    result = powell_experiment(objective, initial_param_values)
    result_dict[var]["Powell"] = result
    result = slsqp_experiment(objective, initial_param_values)
    result_dict[var]["SLSQP"] = result
    result = sgd_experiment(objective,initial_param_values_tensor,sgd)
    result_dict[var]["SGD"] = result
    result = sgd_experiment(objective,initial_param_values_tensor,rmsprop)
    result_dict[var]["RMSPROP"] = result
    result = sgd_experiment(objective,initial_param_values_tensor,adam)
    result_dict[var]["ADAM"] = result
    '''

    

    # save results in json file "conf_[conf_id]_opt.json"
    os.makedirs("experimental_results/results/optimizer_results", exist_ok=True)
    file = open(f"experimental_results/results/optimizer_results/conf_{conf_id}_test.json", mode="w")
    json.dump(result_dict, file)

def test():
    # setup qnn configuration
    conf_id = 0
    data_type = "random" # random, orthogonal, non_lin_ind, var_s_rank
    num_data_points = 1
    s_rank = 1 # Schmidt-Rank
    var = "data_batch_0"
    databatch_id = int(var.strip("datbch_"))

    # setup dictionary for dumping info into json file later
    result_dict = {"conf_id":conf_id, "data_type":data_type, "num_data_points":num_data_points, "s_rank":s_rank}
    val = "[[-0.45070455-0.02711853j,0.78437395-0.06613086j,0.06358678+0.19963393j,-0.07343613+0.35668523j],[-0.01890143-0.03813363j,0.32408202+0.25557629j,0.05872864-0.68979805j,0.55466693-0.20227297j],[-0.11215405+0.64023111j,-0.13344055+0.29565494j,-0.49012687-0.19046288j,-0.04241254+0.44046348j],[0.55771659+0.24656916j,0.31851997-0.05798805j,0.28761525-0.34294258j,-0.56718418+0.03616933j]]"
    result_dict["unitary"] = val#.strip("\"")
    val,_ = re.subn('\[|\]|\\n', '', val)
    unitary = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4))#unitary: 4x4 tensor
    val = '[[[0.09314128-0.12946863j,0.39382838-0.19799267j,0.05133879+0.12112185j,0.08106995-0.04021906j],[0.07622026-0.09754417j,0.31152873-0.14143589j,0.03608905+0.09551662j,0.06411194-0.02869752j],[0.11804856-0.19626647j,0.54031288-0.32976236j,0.08774511+0.16729872j,0.1112873-0.06711204j],[-0.01827577+0.10086995j,-0.17383409+0.2237231j,-0.06326177-0.05610261j,-0.03593256+0.04574145j]]]'
    result_dict[var] = {}
    result_dict[var]["databatch"] = val#.strip("\"")
    val,_ = re.subn('\[|\]|\\n', '', val) 
    data_points = torch.from_numpy(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4)) #data_points: 1x4x4 tensor


    # setup qnn
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    expected_output = torch.matmul(unitary, data_points)
    y_true = expected_output.conj()
    
    # objective function based on cost function of qnn 
    def objective(x):
        qnn.params = torch.tensor(x, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape) # stimmt das???????
        cost = cost_func(data_points, y_true, qnn, device="cpu") 
        return cost.item()
    results = {}
    # verschiedene inital_param_values ausprobieren und avg bilden? 
    # TODO: gleich für alle Konfigs oder nur gleich für eine Konfig aber da für alle Optmimierer?
    initial_param_values = np.random.uniform(0, 2*np.pi, size=dimensions) # [0,2pi] siehe victor_thesis_landscapes.py, bei allen optimierern gleich
    initial_param_values_tensor = torch.tensor(initial_param_values)


    # try adam for 50 iterations, looking at jacobian & x & fun(x)
    res = minimize(objective, initial_param_values, method=adam, 
                        options={"maxiter": 5000, "learning_rate":0.001, "eps":1e-5})
    

if __name__ == "__main__":
    #single_optimizer_experiment(1, "random",1,1,[],[])
    #start = time.time()
    #run_all_optimizer_experiments()
    #print(f"total runtime: {np.round((time.time()-start)/60,2)}min") 
    # total runtime: 17.59min, max_iter: 10000, optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP']
    # total runtime: ca 40 min, max_iter = 1000, optimizers = ['COBYLA', 'BFGS', 'Nelder-Mead', 'Powell', 'SLSQP', sgd, adam, rmsprop]
    
    start = time.time()
    #run_all_optimizer_experiments()
    test_experiment()
    #test()
    print(f"total runtime: {np.round((time.time()-start)/60,2)}min") 
    # die ersten 92 configs: 2h runtime



    #opt = torch.optim.Adam()