# Relations-POMDP

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
  
### Pivot after a lot of discussions

Consider a robot placed in a kitchen, tasked to find a fork.  The robot is
manufactured by a remote company, such that it is equipped with perception and
manipulation skills to detect objects, open drawers, cupboards, fridges, pick
and place objects, etc. However, the robot is not initially familiar with the
particular kitchen: It does not know the locations or number of instances of any
object class beforehand.

A naive approach to solve this task is for the robot to scan the entire kitchen
and build a map, and then exhaustively apply its skills to check all containers,
and look everywhere. This is obviously very inefficient.

What does a human do in this case? Suppose you are asked the same task when
visiting a friend's home. Before you enter the kitchen, you do not know its
configuration. Once you enter, you might look for a few things, such as spoons,
knives, or drawers, if you do not see a fork immediately. But you would most
likely not look for or check the fridge. If you accidentally see a fork holder,
this immediately makes you want to look at it more closely.  This kind of policy
is strongly influenced by a prior knowledge about object co-occurrence. More
generally, the kind of co-occurrence may be not only about object location, but
also other attributes, such as color and shape.

We consider the problem where the robot is tasked to determine an attribute of
an object in an open world. We use POMDP to model the task since it naturally
captures partial observability of the attribute and stochasticity of the
robot. This POMDP has a large, possibly continuous action space since the robot
can interact with many objects in the world or change its own pose.  To produce
an efficient policy similar to human's, we aim to enable robots to form subgoals
actively from a given attribute co-occurrence graph, which is not grounded and
not necessarily accurate. The main research question is, how do we solve the
POMDP more efficiently using the information in the co-occurrence graph?
