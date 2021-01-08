# Relations-POMDP


## Overview

**Problem** A robot must search for a target object in an environment that contains a number of other objects and the locations of these objects follow a (known or learnable) joint probability distribution.

**This package** contains an attempt at building a grid-world environment and developing a POMDP-based method with heuristic-based planning to solve this problem. Each grid world instance consists of a number of objects. Some objects are of type containers (right now the only container is Room), and others are items that exist in containers.

We simulated a 2D mobile robot with a pose specified by (x,y,θ) and that moves with a discrete action space. The robot projects a fan-shaped sensor capable of detecting objects. The geometry and noise level of the sensors vary dependent on the type of object class they are for. For example, room detectors are more robust to false negatives and have larger range compared to a detector for e.g. salt. We do not consider modeling false positives for now, because additional assumptions have to be made about what physical or non-physical entity causes the false positive detection and the right way to do this may involve tracking an unknown number of such entities and this becomes hard to manage in a POMDP framework.

We assume in our problem setting that the robot has access to a number of training environments with the same distribution of object locations as the test environment. We learn a correlation scoring function `Corr(A,B)` that gives a score for how spatially correlated the instances of classes `A` and `B`, and a difficulty function `Dffc(A|B)` that gives a score for how difficult it is to find `A` after `B` has been found. Based on the correlation scoring function, we designed an observation model for belief update, and then based on both of these functions, we designed a heuristic for setting "subgoals", classes that should be found prior to finding the target class. The criticism of this method is it is all very ad-hoc and not principled (the scores are just heuristics without units), and we have not considered realistic assumptions of the object physical properties (e.g. what if you detected a part of the object). Despite these concerns, we have some preliminary results in the grid world domain:

![results_6by6](https://i.imgur.com/5Oxe4c8.png)

![results_10by10](https://i.imgur.com/pBHTXoa.png)


### Example 1
**Subgoal agent with correlation belief update**

Subgoal: Kitchen.
Task goal: Salt.
![subgoal-nk_3](https://i.imgur.com/9y5c0Kk.gif)

### Example 2
**Subgoal agent with correlation belief update**

Subgoal: Kitchen.
Task goal: Salt.
![subgoal-nk_1](https://i.imgur.com/IEuhJQj.gif)


**Subgoal agent WITHOUT correlation belief update**
![subgoal-nk-nocorr_2](https://i.imgur.com/A1JUTva.gif)

## Usage

1. Before running anything, switch to the [for_relpomdp](https://github.com/zkytony/pomdp-py/tree/for_relpomdp) branch in pomdp_py.

2. Do `make build` to build pomdp_py again.

3. Then, go to `relpomdp/home2d/tests`. You should be able to run the following. The agent will search for salt.

MDP agent
```
python test_mdp.py
```
POMDP agent
```
python test_pomdp.py
```
POMDP agent with no knowledge of the map in the beginning
```
python test_pomdp_nk.py
```
Heuristic agent with no knowledge of the map in the beginning
```
python test_heuristic_nk.py
```

Random agent with no knowledge of the map in the beginning
```
python test_heuristic_nk.py
```

4. To run a subgoal agent with correlation belief update, go to `relpomdp/home2d/planning`. Then,
```
python test_subgoals_nk.py ../experiments/configs/10x10_11-17-2020.yaml ../experiments/data/difficulty-train-envs-6x6-pomdp-20201113134230180.csv ../experiments/data/correlation-try1-10-20-2020.csv ../experiments/data/scores_C#6x6_10-31-2020_D#train-envs-6x6-pomdp_20201113140449444.csv -T Salt --seed 100
```
Another command to try
```
python test_subgoals_nk.py ../experiments/configs/10x10_10-20-2020.yaml ../experiments/data/difficulty-try1-10-20-2020-20201026162744897.csv ../experiments/data/correlation-try1-10-20-2020.csv ../experiments/data/subgoal-scores\=try1.csv -T Salt
```


## Progress Log

[11-26-2020]

We are now moving past 2D gridworld. We have done what we can given the assumptions in this domain and presented the preliminary results. The focus of the next stage is:

* More realism as an object search problem. On the AI2Thor domains
* Principled math to formulate problem and (derive) approach
* Writing - How does this work situate in the literature comapred to other:
   * Object search work (including robotics ones, and the semantic visual navigation in vision community)
   * POMDP learning/planning work

[11-07-2020]

Currently, we are staying on the route towards object search with relations.

The current stage of code has all tests running:

- The tests under `home2d/tasks/{task_name}/test.py`
- The tests under `home2d/agent/tests/test_*.py`
- The script under `home2d/learning/testing/test_subgoals_nk.py`. For example, by
  ```
  python test_subgoals_nk.py ../configs/6x6_10-31-2020.yaml ../data/difficulty-try1-10-20-2020-20201026162744897.csv ../data/correlation-try1-10-20-2020.csv scores.csv -T Salt
  ```
  These configuration and data files are not tracked by the repo, but can be found in [this Google Drive directory](https://drive.google.com/drive/u/1/folders/1-xTE3QVCt8MVjARoYobW-cI9HbsXfdtU).


[10-04-2020]

By this stage of the project, several topics are explored,
in the order of time:

- Habitat: The code in `/habitat` are examples of how to process scene data (get
  the furniture locations, shape, etc.), and a simple `interaction.py` example that
  runs a mobile robot in a scene for a few manually specified waypoints. The in
  this directory works, assuming that you have installed `habitat_sim` as well
  as `habitat-api`, and downloaded the habitat scenes dataset (which should
  come with the `habitat_sim` repository.

  - `data_viz.py`: Visualize the objects (e.g. couch, wall, etc.) from a scene
        by plotting their bounding boxes

  - `stereo.py`: example of running a stereo camera in an example scene

  - `interaction.py`: example of running a mobile robot and recording the frames


  **Conclusion** Habitat is nice, but (1) it is difficult to construct a map out of a scene,
      (2) spending more time on this deviates from the research question.

- Object-Oriented POMDP reimplementation (`oopomdp`): When I first shifted away from
  habitat, one of the things I was thinking was that the relations in our
  project can have something to do with the relations defined in OO-MDP. So I
  understood that paper better, and implemented OO-POMDP using the kind of
  condition/effect transition mechanism in OO-MDP, and also used the same
  mechanism to describe observations. A relation is mostly used to describe
  whether a statement about the state is true and if so what effect will happen
  (in the transition dynamics).

  I think this implementation of OO-POMDP is better than in `pomdp_py` because
  the transition/observation models do not assume object independence. It is also
  "closer" to the formulation in OO-MDP.

  **Conclusion** This OO-POMDP implementation supported the later toy experiments
  in the salt and pepper domain, as well as a trial implementation of subgoal-based
  planning.


- Probablistic Graphical Models (`pgm`): Exploring the use of `pgmpy` in Python,
  specifically its implementation of MRF and exact belief propagation.

  - `pgm/mrf.py` contains a `SemanticMRF` class that is an interface to use the
        pgmpy library's MRF and BP functionalities. The most tricky bit is that
        pgmpy keeps track of discrete values only as integers, so user needs to
        maintain a mapping from the actual semantic value to the integer.

  **Conclusion** I was able to use this to describe simple relations and do inference
  with it to help belief update in the salt and pepper domain

- `home2d`: The rest of the effort is in developing the salt and pepper domain and
  an algorithm to plan in it. There has been several iterations, but in the most
  recent one, I organized the code into `home2d` with the intention that this framework
  can generalize to other home-related domains. The idea is that there are something
  different tasks share in tasks at home (like the action of moving), the map, and
  basic relation such as touching a wall, etc. But there are more specific tasks, such
  as searching for an object, that can have its own unique action/observations.

  - The state and observation are expressed by `ObjectState` and `ObjectObservation`
        in the `oopomdp` framework.
  - Only N/W/S/E move actions are defined in the overall `home2d/action.py`.
  - The robot is assumed to be given the map room layout
  - A relational graph is given
  - Three tasks are written under `home2d/tasks`: Search item, search room, and leave room.
    Each individual task directory has a `test.py` script that you can run to test just one task.
  - Then, `home2d/planning` implements the approach of using the relational
    graph passively for belief update (`home2d/planning/belief_update.py`) and
    using it actively for subgoal generation (`home2d/planning/grounding.py`).
    The `home2d/planning/test.py` script is an example of how this algorithm
    works in the salt and pepper domain.

  **Conclusion** The iterations over the salt and pepper domain yields (1) The
  relational graph, if used directly and actively for planning, should be just
  for tasks where the goal is to ground some attribute (goal), and that grounding something
  else that's related is helpful to ground the goal attribute more efficiently.
  (2) Only doing navigation-based object search has been a very well studied area.
  It is hard to argue that the system we'll have is better than the previous ones
  because the resulting planning behavior could be similar (co-occurence based).
  (3) Planning with subgoals is indeed promising. But the assumption that a
  room layout is given, is not very satisfying.
