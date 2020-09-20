
# Tests MRF scalability
from relpomdp.object_search.world_specs.build_world import init_map, walls_to_states
from relpomdp.object_search.grid_map import GridMap
from relpomdp.pgm.mrf import SemanticMRF, relations_to_mrf
from relpomdp.object_search.relation import *
import time
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import sys
import random

def inference_time_vs_num_relations_no_loop(world_size=(10,10)):
    walls = init_map(*world_size)
    grid_map = GridMap(*world_size, walls_to_states(walls), [])

    # No loop
    target_item = "ItemType0"
    relations = []
    times_construct = []
    times_inference = []
    for i in range(100):
        object_class = "ItemType%d" % i
        if target_item == object_class:
            continue
        near = Near(target_item, object_class, grid_map)
        relations.append(near)

        # Construct mrf
        start_time = time.time()
        mrf = relations_to_mrf(relations)
        time_construct = time.time() - start_time

        # Do inference
        target_variable = target_item + "_pose"
        start_time = time.time()
        target_phi = mrf.query(variables=[target_variable])
        time_inference = time.time() - start_time
        times_construct.append(time_construct)
        times_inference.append(time_inference)
        print(" %d   | %.4fs   | %.4f"  % (i, time_construct, time_inference))

    print("Times used to construct MRF")
    print(times_construct)
    os.makedirs(os.path.join("results", "mrf_test"), exist_ok=True)
    with open(os.path.join("results", "mrf_test", "times_construct_no_loop.yaml"), "w") as f:
        yaml.dump(times_construct, f)
    print("Times used to infer")
    print(times_inference)
    with open(os.path.join("results", "mrf_test", "times_inference_no_loop.yaml"), "w") as f:
        yaml.dump(times_inference, f)


def inference_time_vs_num_relations_loop(world_size=(10,10), loop_chance=0.5):
    """
    When adding a relation, `loop_chance` percentage of times the relation
    will be between two existing variables. The complement, a new relation.
    """
    walls = init_map(*world_size)
    grid_map = GridMap(*world_size, walls_to_states(walls), [])

    # No loop
    variables = set()
    variable_pairs = set()
    relations = []
    times_construct = []
    times_inference = []
    while len(relations) < 1:
        if len(variables) >= 2 and random.uniform(0,1) <= loop_chance:
            # Add a relation between two existing variables
            variables = random.sample(variables, 2)
            if tuple(variables) not in variable_pairs\
               and tuple(reversed(varaibles)) not in variable_pairs:
                near = Near(variables[0], variables[1], grid_map)
                print("doing loop...")
            else:
                continue
        else:
            new_variable = "ItemType%d" % len(variables)
            variables.add(new_variable)
            if len(variables) < 2:
                continue
            variable = random.sample(variables - set({new_variable}), 1)[0]
            near = Near(new_variable, variable, grid_map)
        variable_pairs.add((near.class1.name, near.class2.name))
        print(near)
        relations.append(near)

        # Construct mrf
        start_time = time.time()
        mrf = relations_to_mrf(relations)
        time_construct = time.time() - start_time

        # Do inference
        variable = random.sample(variables, 1)[0]        
        target_variable = variable + "_pose"
        start_time = time.time()
        target_phi = mrf.query(variables=[target_variable])
        time_inference = time.time() - start_time
        times_construct.append(time_construct)
        times_inference.append(time_inference)
        print(" %d   | %.4fs   | %.4f"  % (len(relations), time_construct, time_inference))

    print("Times used to construct MRF")
    print(times_construct)
    os.makedirs(os.path.join("results", "mrf_test"), exist_ok=True)
    with open(os.path.join("results", "mrf_test", "times_construct_loop.yaml"), "w") as f:
        yaml.dump(times_construct, f)
    print("Times used to infer")
    print(times_inference)
    with open(os.path.join("results", "mrf_test", "times_inference_loop.yaml"), "w") as f:
        yaml.dump(times_inference, f)        

def plot_results():
    # MRF no loop
    with open(os.path.join("results", "mrf_test", "times_construct_no_loop.yaml")) as f:
        times_construct = yaml.load(f, Loader=yaml.Loader)
    with open(os.path.join("results", "mrf_test", "times_inference_no_loop.yaml")) as f:
        times_inference = yaml.load(f, Loader=yaml.Loader)
        
    xvals = np.arange(len(times_construct))
    plt.xlabel("Number of relations")
    plt.ylabel("Time (seconds)")
    plt.plot(xvals, times_construct, label="Construct MRF")
    plt.plot(xvals, times_inference, label="Inference")
    plt.legend(loc="upper left")
    os.makedirs(os.path.join("results", "mrf_test", "plots"), exist_ok=True)
    plt.savefig(os.path.join("results", "mrf_test", "plots", "times_no_loop.png"))
    

if __name__ == "__main__":
    inference_time_vs_num_relations_loop(world_size=(10,10), loop_chance=0.5)
    # inference_time_vs_num_relations()
    # plot_results()
