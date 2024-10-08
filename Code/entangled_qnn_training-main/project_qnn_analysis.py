import csv
import gc
import json
import time
from datetime import datetime
from qnns.cuda_qnn import CudaPennylane
import re

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
from scipy.optimize import minimize, dual_annealing
import re

def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            print(f"Lade Datei: {file_path}")
            with open(file_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    data.append(json_data)
                except json.JSONDecodeError:
                    print(f"Fehler beim Laden der Datei: {file_path}")
    if not data:
        print("Keine JSON-Dateien gefunden oder alle Dateien sind fehlerhaft.")
    return data

def extract_solution_fun_data(json_data):
    '''
        Extract mean x_min and x_max value for every optimizer for every bound for one config.
    '''
    gradient_free = ["nelder_mead", "powell", "cobyla"]
    gradient_based = ["sgd", "adam", "rmsprop", "bfgs", "dual_annealing","slsqp"]
    optimizers = gradient_based + gradient_free
    bounds_batches = ["bounds_0", "bounds_1", "bounds_2", "bounds_3", "bounds_4"]
    databatches = ["databatch_0", "databatch_1", "databatch_2", "databatch_3", "databatch_4"]
    optimizer_data = {}
    #gradient_based_data = []
    #gradient_free_data = []

    # prepare results dict
    # bounds_i : opt_1 : [(x_min1, x_max1), (x_min2, x_max2),...], opt_2 : ...
    res_min = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}
    res_max = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}

    for i in range(len(json_data)):
        print(f"Verarbeite config_{i}")
        for databatch_id in databatches:
            print(f"Verarbeite {databatch_id}")
            for bounds_id in bounds_batches:
                print(f"Verarbeite {bounds_id}")
                for opt in optimizers:
                    print(f"Verarbeite {opt}")
                    try:
                        dict = json_data[i][databatch_id][bounds_id][opt]["0"]
                        if opt not in res_min[bounds_id]:
                            res_min[bounds_id][opt] = []
                        if opt not in res_max[bounds_id]:
                            res_max[bounds_id][opt] = []
                        x = dict["x"]
                        x = [float(idx) for idx in x.strip('[ ]').split()]
                        res_min[bounds_id][opt].append(np.min(x))
                        res_max[bounds_id][opt].append(np.max(x))
                    except KeyError as e:
                        print(f"Fehler beim Lesen der Daten: {e}")

    # Berechne Pro Optimierer (pro Bounds) untere x-Grenze und obere x-Grenze
    res_min_max = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}
    for bounds_id in bounds_batches:
        print(f"Verarbeite {bounds_id}")
        for opt in optimizers:
            print(f"Verarbeite {opt}")
            try:
                res_min_max[bounds_id][opt] = (np.min(res_min[bounds_id][opt]),np.max(res_max[bounds_id][opt]))
            except KeyError:
                print(f'Optimierer existiert für diese bounds nicht.')
    
    return res_min, res_max, res_min_max   

def extract_solution_x_data(json_data):
    '''
        Extract mean x_min and x_max value for every optimizer for every bound for one config.
    '''
    gradient_free = ["nelder_mead", "powell", "cobyla"]
    gradient_based = ["sgd", "adam", "rmsprop", "bfgs", "dual_annealing","slsqp"]
    optimizers = gradient_based + gradient_free
    bounds_batches = ["bounds_0", "bounds_1", "bounds_2", "bounds_3", "bounds_4"]
    databatches = ["databatch_0", "databatch_1", "databatch_2", "databatch_3", "databatch_4"]
    optimizer_data = {}
    #gradient_based_data = []
    #gradient_free_data = []

    # prepare results dict
    # bounds_i : opt_1 : [(x_min1, x_max1), (x_min2, x_max2),...], opt_2 : ...
    res_min = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}
    res_max = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}

    for i in range(len(json_data)):
        print(f"Verarbeite config_{i}")
        for databatch_id in databatches:
            print(f"Verarbeite {databatch_id}")
            for bounds_id in bounds_batches:
                print(f"Verarbeite {bounds_id}")
                for opt in optimizers:
                    print(f"Verarbeite {opt}")
                    try:
                        dict = json_data[i][databatch_id][bounds_id][opt]["0"]
                        if opt not in res_min[bounds_id]:
                            res_min[bounds_id][opt] = []
                        if opt not in res_max[bounds_id]:
                            res_max[bounds_id][opt] = []
                        x = dict["x"]
                        x = [float(idx) for idx in x.strip('[ ]').split()]
                        res_min[bounds_id][opt].append(np.min(x))
                        res_max[bounds_id][opt].append(np.max(x))
                    except KeyError as e:
                        print(f"Fehler beim Lesen der Daten: {e}")

    # Berechne Pro Optimierer (pro Bounds) untere x-Grenze und obere x-Grenze
    res_min_max = {"bounds_0": {}, "bounds_1": {}, "bounds_2": {}, "bounds_3": {}, "bounds_4": {}}
    for bounds_id in bounds_batches:
        print(f"Verarbeite {bounds_id}")
        for opt in optimizers:
            print(f"Verarbeite {opt}")
            try:
                res_min_max[bounds_id][opt] = (np.min(res_min[bounds_id][opt]),np.max(res_max[bounds_id][opt]))
            except KeyError:
                print(f'Optimierer existiert für diese bounds nicht.')
    
    return res_min, res_max, res_min_max

def create_min_max_boxplots(res_min, res_max, save_path):
    bounds = {"bounds_0": "No Bounds", "bounds_1": r"$[0, 2\pi]$", "bounds_2": r"$[0, 4\pi]$", "bounds_3": r"$[-2\pi, 2\pi]$", "bounds_4": r"$[-4\pi, 4\pi]$"}
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for bounds_id in bounds.keys():
        plt.figure(figsize=(10,10))
        data_min = res_min[bounds_id]
        data_max = res_max[bounds_id]
        x = np.array([(i+1)*1000 for i in range(len(data_min.keys()))])
        plt.boxplot(data_min.values(), sym="", vert=False,positions=x-200,widths=200)
        plt.boxplot(data_max.values(), sym="", vert=False,positions=x+200,widths=200)
        plt.yticks(ticks=x,labels=data_min.keys())
        plt.ylabel('Optimizer')
        plt.xlabel('Minimal (lower) and maximal (upper) x-values')
        plt.title(f"Minimal and Maximal x-Values for bounds: {bounds[bounds_id]}",fontsize='xx-large')
        #plt.legend()
        #plt.ylim(bottom=-0.015,  top=max(fun) + 0.05)
        plt.grid(True)
        file_path = os.path.join(save_path, f'{bounds_id}_boxplot_no_outliers.png')
        plt.savefig(file_path)
        plt.close()

def extract_optimizer_data(json_data,use_nits=True):
    gradient_based = ["nelder_mead", "powell", "cobyla"]
    gradient_free = ["sgd", "adam", "rmsprop", "bfgs", "dual_annealing", "slsqp"]
    optimizers = gradient_based + gradient_free

    optimizer_data = {}
    gradient_based_data = []
    gradient_free_data = []
    it_key = "nits"
    if(use_nits==False):
        it_key = "maxiter"

    for entry in json_data:
        if isinstance(entry, dict):
            # alle databatches durchgehen
            for batch_key in entry:
                if batch_key.startswith("databatch_"):
                    print(f"Verarbeite Datenbatch: {batch_key}")
                    for optimizer in optimizers:
                        if optimizer in entry[batch_key]:
                            print(f"Verarbeite Optimierer: {optimizer}")

                            # liste für optimierer
                            if optimizer not in optimizer_data:
                                optimizer_data[optimizer] = []

                            # daten für jeden durchlauf entnehmen
                            batch_data = entry[batch_key][optimizer]
                            for key in batch_data:
                                data = batch_data[key]
                                
                                # data muss dictionary sein und schlüssel enthalten
                                if isinstance(data, dict):
                                    nit = data.get(it_key, None)
                                    fun = data.get("fun", None)
                                    
                                    if nit is not None and fun is not None:
                                        try:
                                            nit = int(nit)
                                            fun = float(fun)
                                            optimizer_data[optimizer].append((nit, fun))
                                            if optimizer in gradient_based:
                                                gradient_based_data.append((nit, fun))
                                            if optimizer in gradient_free:
                                                gradient_free_data.append((nit, fun))
                                        except ValueError as e:
                                            print(f"Fehler beim Konvertieren der Daten: {e}")
                                    else:
                                        print(f"Fehlende Schlüssel in den Daten: {data}")
                                else:
                                    print(f"Unerwartete Datenstruktur: {data}")
                        else:
                            print(f"Optimierer {optimizer} nicht in den Datenbatch {batch_key} gefunden")
        else:
            print("Eintrag ist kein Dictionary")

    # Berechne mean fun values
    def calculate_mean_data(data):
        if not data:
            return [], []
        data.sort()
        nits, funs = zip(*data)
        unique_nits = sorted(set(nits))
        mean_funs = [np.mean([fun for nit, fun in data if nit == unique_nit]) for unique_nit in unique_nits]
        return unique_nits, mean_funs

    mean_optimizer_data = {optimizer: calculate_mean_data(results) for optimizer, results in optimizer_data.items()}
    mean_gradient_based_data = calculate_mean_data(gradient_based_data)
    mean_gradient_free_data = calculate_mean_data(gradient_free_data)
    
    return mean_optimizer_data, mean_gradient_based_data, mean_gradient_free_data, optimizer_data

def boxplot_fun_values_per_optimizers(data, opt):
    '''
        Only works if iterations are 100, 500 and 1000
    '''
    save_path = 'qnn-experiments/experimental_results/results/box_plots/'
    data_opt = data[opt]
    #nits, funs = zip(*data_opt)
    #hist, bin_edges = np.histogram(nits)
    data_dict = {100: [], 500: [], 1000: []}
    for n, f in data_opt:
        if(n==100):
            data_dict[100].append(f)
        elif(n==500):
            data_dict[500].append(f)
        elif(n==1000):
            data_dict[1000].append(f)
        else:
            print(f"Iterationszahl {n} ist nicht 100, 500, oder 1000")
    plt.figure()
    x = [100,500,1000]
    plt.boxplot(data_dict.values(), labels=data_dict.keys())
    #plt.xticks(ticks=x,labels=x)
    plt.xlabel('Iterations')
    plt.ylabel('Function value')
    #plt.ylim((0,1))
    plt.title(f"Achieved loss function values per number of (maximum) iterations:\n{opt}",fontsize='large')
    plt.grid(True)
    file_path = os.path.join(save_path, f'{opt}_boxplot_fun.png')
    plt.savefig(file_path)
    plt.close()
    

def get_conf_ids(data_type, num_data_points, s_rank):
    '''
        Returns a list of config ids that correspond to data_type, num_data_points and s_rank.
        data_type (String): random, orthogonal, non_lin_ind, var_s_rank
        num_data_points (String): 1,2,3,4
        s_rank (String): 1,2,3,4
    '''
    data = []
    conf_id_list = []
    file_path = "Code/entangled_qnn_training-main/data/configDict.json"
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            for i in range(len(data)):
                if(data[str(i)]["data_type"]==data_type and data[str(i)]["num_data_points"]==num_data_points and data[str(i)]["s_rank"]==s_rank):
                    conf_id_list.append(i)
        except json.JSONDecodeError:
            print(f"Fehler beim Laden der Datei: {file_path}")
    return(conf_id_list)

def load_json_data(directory, conf_id_list):
    '''
        Load Json-data for each config_id in conf_id_list from files saved in directory.
        File names start with "conf_{config_id}_" and end with ".json".
    '''
    all_data = {}
    for id in conf_id_list:
        all_data[id] = []
        for filename in os.listdir(directory):
            if filename.endswith('.json') and filename.startswith(f'conf_{id}_'):
                file_path = os.path.join(directory, filename)
                print(f"Lade Datei: {file_path}")
                with open(file_path, 'r') as file:
                    try:
                        all_data[id].append(json.load(file))
                    except json.JSONDecodeError:
                        print(f"Fehler beim Laden der Datei: {file_path}")
        if not all_data:
            print("Keine JSON-Dateien gefunden oder alle Dateien sind fehlerhaft.")
    return all_data

def extract_mean_callback_data(directory, max_iter, opt, data_type, num_data_points, s_rank):
    '''
        For each entry in json_data (each configuration) extract list of every tenth fun-value (callback), no of maximum iterations,
        config id if the config_id fullfills data_type, num_data_points, s_rank. 
        Exactly one of data_type, num_data_points or s_rank is None, hence callback data for all possible values of this parameter will be extracted.
    
        data_type (String): random, orthogonal, non_lin_ind, var_s_rank
        num_data_points (String): 1,2,3,4
        s_rank (String): 1,2,3,4
    '''
    # Stepsize: Stepsize between Iterations whose fun value is saved in callback
    # for Powell, BFGS and Dual Annealing: stepsize = 1 (every iteration)
    # for all other optimizers: stepsize = 10 (every 10th iteration)
    stepsize = 10
    if opt in ['powell', 'bfgs', 'dual_annealing']:
        stepsize = 1

    # check only one parameter (data_type, num_data_points, s_rank) is None:
    param_values = {'data_type': ["random", "orthogonal", "non_lin_ind", "var_s_rank"], 'num_data_points': ["1","2","3","4"], 's_rank': ["1","2","3","4"]}
    param_names = ['data_type', 'num_data_points', 's_rank']
    params = [data_type, num_data_points, s_rank]
    none_indices = [i for i in range(len(params)) if params[i] == None]
    if(len(none_indices)>1):
        raise Exception('Only one parameter of data_type, num_data_points and s_rank is allowed to be None')
    
    # determine all config_ids that fulfill (data_type, num_data_points, s_rank)
    none_param = param_names[none_indices[0]]
    values = param_values[none_param]
    conf_id_list = {} # dictionary: for each possible value of the non-specified parameter a list of corresponding config_ids is saved
    for value in values:
        params_current = [value if v is None else v for v in params] #determine correct parameter values (substitute value for None)
        conf_id_list[value] = get_conf_ids(data_type=params_current[0], num_data_points=params_current[1], s_rank=params_current[2])

    # for each list in conf_id_list (i.e. each possible value of non-specified parameter) (and each databatch): determine a list of mean callback values
    mean_fun_values = {}
    max_nit_values = {}
    for value in conf_id_list.keys():
        print(value,conf_id_list[value])
        all_data = load_json_data(directory, conf_id_list[value])
        fun_values = []
        nit_values = []
        for id in conf_id_list[value]:
            print(id)
            for entry in all_data[id]:
                if isinstance(entry, dict):
                    # go through each databatch
                    for batch_key in entry:
                        if batch_key.startswith("databatch_"):
                            print(f"Verarbeite Datenbatch: {batch_key}")
                            if opt in entry[batch_key]:
                                print(f"Verarbeite Optimierer: {opt}")
                                # get data for optimizer opt
                                batch_data = entry[batch_key][opt]
                                for key in batch_data:
                                    data = batch_data[key]
                                    
                                    # data must be dictionary and contain keys
                                    if isinstance(data, dict):
                                        nit = data.get("nit", None) # nit: number of total iterations needed to reach optimal fun-value
                                        fun = data.get("fun", None) # fun: optimal fun-value reached during optimization
                                        iter = data.get("maxiter", None) # maxiter: number of maximum iterations optimizer was given (100, 500, or 1000)
                                        callback = data.get("callback", None) # callback: list of fun_values for every tenth iteration
                                        if(iter == max_iter):
                                            if nit is None or opt == 'dual_annealing': #cobyla doesn't save nit and dual_annealing saves the wrong value (max_iter) for nit
                                                nit = (len(callback)-1)*stepsize
                                            if nit is not None and fun is not None:
                                                try:
                                                    nit = int(nit)
                                                    fun = float(fun)
                                                    if(callback[-1] != fun): # append optimal fun value, if it isn't already the last value in callback-list
                                                        callback.append(fun)
                                                    fun_values.append(callback) 
                                                    nit_values.append(nit)
                                                except ValueError as e:
                                                    print(f"Fehler beim Konvertieren der Daten: {e}")
                                            else:
                                                print(f"Fehlende Schlüssel in den Daten: {data}")
                                    else:
                                        print(f"Unerwartete Datenstruktur: {data}")
                            else:
                                print(f"Optimierer {opt} nicht in den Datenbatch {batch_key} gefunden")
                else:
                    print("Eintrag ist kein Dictionary")
        # compute mean of callback fun values over each config_id and each databatch per config_id
        # for runs that used less iterations than others: fill those lists with the optimal fun value achieved
        max_len = len(max(fun_values, key=len))
        # pad right of each sublist of fun_values with optimal fun value to make it as long as the longest sublist
        for sublist in fun_values:
            opt_fun_val = sublist[-1]
            sublist[:] = sublist + [opt_fun_val] * (max_len - len(sublist))
        fun_arrays = [np.array(x) for x in fun_values]
        mean_fun_values[value] = [np.mean(k) for k in zip(*fun_arrays)]
        # compute maximum of total number of iterations (nit) for each config_id and each databatch per config_id
        if nit_values.count(nit_values[0]) != len(nit_values):
            print("INFO: Multiple number of total iterations found. Maximum of those values will be chosen.")
        max_nit_values[value] = np.max(nit_values)
    return mean_fun_values,max_nit_values

def convergence_plot_per_optimizer(save_path, mean_fun_data, mean_nit_data, opt, maxiter, data_type, num_data_points, s_rank):
    '''
        Convergence plot for mean callback values where exactly one parameter of data_type, num_data_points or s_rank is None and thus variable.
        mean_fun_data is a dictionary where the possible values for the variable parameter are the key and each value saved for a key is a list of fun_values
        mean_nit_data is a list of the corresponding number of iterations for the found optimal fun value (last value in each list in mean_fun_data)
    '''
    # create correct directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Stepsize: Stepsize between Iterations whose fun value is saved in callback (influences x-axis of plot)
    # for Powell, BFGS and Dual Annealing: stepsize = 1 (every iteration)
    # for all other optimizers: stepsize = 10 (every 10th iteration)
    stepsize = 10
    if opt in ['powell', 'bfgs', 'dual_annealing']:
        stepsize = 1

    title = f'Convergence plot for {opt}, maxiter = {maxiter}, \n Datatype: {data_type}, Number of Data Points: {num_data_points}, Schmidt rank: {s_rank}'
    #determine what parameter is variable (i.e. None in argument list) and check that only one parameter is None
    param_names = ['data_type', 'num_data_points', 's_rank']
    params = [data_type, num_data_points, s_rank]
    none_indices = [i for i in range(len(params)) if params[i] == None]
    if(len(none_indices)>1):
        raise Exception('Only one parameter of data_type, num_data_points and s_rank is allowed to be None')
    none_param = param_names[none_indices[0]]

    #colors for each config id
    cmap = matplotlib.cm.get_cmap('Spectral')
    plt.figure()
    c = 0 # needed to determine correct color
    for param_value in mean_fun_data.keys():
        color = cmap(c/len(mean_fun_data))
        label = f"{none_param} = {param_value}"
        y = mean_fun_data[param_value]
        x = np.append(np.arange(0,(len(y)-1)*stepsize, stepsize), mean_nit_data[param_value])
        plt.plot(x,y, color=color, label=label)
        print(param_value, "ok")
        c += 1
    plt.ylim(0,1)
    plt.xlabel('Iterations')
    plt.ylabel('Function value')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    file_path = os.path.join(save_path, f'{opt}_convergence_fun_{data_type}{num_data_points}{s_rank}.png') # TODO: better naming system
    plt.savefig(file_path)
    plt.close()

def load_and_extract_callback_data(directory, data_type, num_data_points, s_rank, max_iter, opt, databatch):
    '''
        DEPRECATED
        For each entry in json_data (each configuration) extract list of every tenth fun-value (callback), no of maximum iterations,
        config id if the config_id fullfills data_type, num_data_points, s_rank. 
        If one of data_type, num_data_points or s_rank is None callback data for all possible values of this parameter will be extracted.
        Only one of those parameters can be None.
        data_type (String): random, orthogonal, non_lin_ind, var_s_rank
        num_data_points (String): 1,2,3,4
        s_rank (String): 1,2,3,4
    '''
    
    # determine all config_ids that fulfill (data_type, num_data_points, s_rank)
    conf_id_list = get_conf_ids(data_type, num_data_points, s_rank)

    # load json data for each config_id
    all_data = load_json_data(directory, conf_id_list)

    fun_values = {}
    for id in conf_id_list:
        for entry in all_data[id]:
            fun_values[id] = []
            if isinstance(entry, dict):
                # go through each databatch
                for batch_key in entry:
                    if batch_key == f'databatch_{databatch}':
                        print(f"Verarbeite Datenbatch: {batch_key}")
                        if opt in entry[batch_key]:
                            print(f"Verarbeite Optimierer: {opt}")
                            # get data for optimizer opt
                            batch_data = entry[batch_key][opt]
                            for key in batch_data:
                                data = batch_data[key]
                                
                                # data must be dictionary and contain keys
                                if isinstance(data, dict):
                                    nit = data.get("nit", None) # nit: number of total iterations needed to reach optimal fun-value
                                    fun = data.get("fun", None) # fun: optimal fun-value reached during optimization
                                    iter = data.get("maxiter", None) # maxiter: number of maximum iterations optimizer was given (100, 500, or 1000)
                                    callback = data.get("callback", None) # callback: list of fun_values for every tenth iteration
                                    if(iter == max_iter):
                                        if nit is not None and fun is not None:
                                            try:
                                                nit = int(nit)
                                                fun = float(fun)
                                                if(len(callback)*10 != nit): # append optimal fun value, if it isn't already the last value in callback-list
                                                    callback.append(fun)
                                                fun_values[id].append((nit, callback)) 
                                            except ValueError as e:
                                                print(f"Fehler beim Konvertieren der Daten: {e}")
                                        else:
                                            print(f"Fehlende Schlüssel in den Daten: {data}")
                                else:
                                    print(f"Unerwartete Datenstruktur: {data}")
                        else:
                            print(f"Optimierer {opt} nicht in den Datenbatch {batch_key} gefunden")
            else:
                print("Eintrag ist kein Dictionary")
    return fun_values

def convergence_plot_per_optimizerOLD(data, opt, data_type, num_data_points, s_rank, maxiter, databatch):
    '''
        DEPRECATED
    '''
    save_path = 'qnn-experiments/experimental_results/results/convergence_plots/'
    title = f'Convergence plot for {opt}, maxiter = {maxiter}, databatch = {databatch}\n Datatype: {data_type}, Number of Data Points: {num_data_points}, Schmidt rank: {s_rank}'
    #colors for each config id
    cmap = matplotlib.cm.get_cmap('Spectral')
    plt.figure()
    c = 0
    for id in data.keys():
        color = cmap(c/len(data))
        c += 1
        label = f"Config {id}"
        for i in range(len(data[id])):
            values = data[id][i]
            y = values[1]
            x = np.append(np.arange(10,values[0], 10), values[0])
            if(i==0):
                plt.plot(x,y, color=color, label=label)
            else:
                plt.plot(x,y, color=color)
    plt.xlabel('Iterations')
    plt.ylabel('Function value')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    file_path = os.path.join(save_path, f'{opt}_convergence_fun_{data_type}{num_data_points}{s_rank}.png') # TODO: better naming system
    plt.savefig(file_path)
    plt.close()




if __name__ == "__main__":
    
    optimizers = ['nelder_mead', 'powell', 'sgd', 'adam', 'rmsprop', 'bfgs','slsqp','dual_annealing','cobyla']
    datatype_list = ['random', 'orthogonal', 'non_lin_ind']#, 'var_s_rank']
    num_data_points_list = ['1', '2', '3', '4']
    s_rank_list = ['1', '2', '3', '4']
    origin_path = 'experimental_results/results/optimizer_results/'
    

    # convergence plots for variable s_rank, but fixed datatype and num_data_points
    for datatype in datatype_list:
        for num_data_points in num_data_points_list:
            save_path = f'qnn-experiments/experimental_results/results/convergence_plots/datatype/{datatype}/num_data_points/{num_data_points}'
            for opt in optimizers:
                fun_values, nit_values = extract_mean_callback_data(origin_path,1000,opt,datatype, num_data_points,None) 
                convergence_plot_per_optimizer(save_path, fun_values,nit_values, opt, 1000, datatype, num_data_points, None)
                print(opt, "ok")

    # convergence plots for variable num_data_points, but fixed datatype and s_rank
    for datatype in datatype_list:
        for s_rank in s_rank_list:
            save_path = f'qnn-experiments/experimental_results/results/convergence_plots/datatype/{datatype}/s_rank/{s_rank}'
            for opt in optimizers:
                fun_values, nit_values = extract_mean_callback_data(origin_path,1000,opt,datatype, None, s_rank) 
                convergence_plot_per_optimizer(save_path, fun_values,nit_values, opt, 1000, datatype, None, s_rank)
                print(opt, "ok")   

    # convergence plots for variabel datatype, but fixed num_data_points and s_rank
    for s_rank in s_rank_list:
        for num_data_points in num_data_points_list:
            save_path = f'qnn-experiments/experimental_results/results/convergence_plots/s_rank/{s_rank}/num_data_points/{num_data_points}'
            for opt in optimizers:
                fun_values, nit_values = extract_mean_callback_data(origin_path,1000,opt,None, num_data_points,s_rank) 
                convergence_plot_per_optimizer(save_path, fun_values,nit_values, opt, 1000, None, num_data_points, s_rank)
                print(opt, "ok")

    

    #path = "experimental_results/results/optimizer_results/bounds/"
    #data = load_json_files(path)
    #print(data[0]["conf_id"])
    #res_min,res_max,res = extract_solution_x_data(data)
    #create_min_max_boxplots(res_min, res_max, 'qnn-experiments/experimental_results/results/box_plots/bounds/no_outliers')
    '''path = 'experimental_results/results/2024-07-19_allConfigs_allOpt'
    json_data = load_json_files(path)
    _,_,_,data = extract_optimizer_data(json_data,use_nits=False)

    optimizers = ["nelder_mead", "powell", "cobyla", "bfgs", "slsqp","sgd", "adam", "rmsprop", "dual_annealing"]
    for opt in optimizers:
        boxplot_fun_values_per_optimizers(data, opt)'''
    