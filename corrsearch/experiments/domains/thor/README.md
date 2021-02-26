The code here uses Ai2Thor's RoboTHOR.

The `ai2thor` package version used is [2.7.2](https://github.com/allenai/ai2thor/releases/tag/2.7.2).


### Plan

1. Determine if there is a reasonable joint distribution between objects

2. Which 10 objects should be used as targets?

3. How to define and implement detectors?

4. How to define transition?

5. How to perform object search with groundtruth object detections? (with artificial noise?)

6. If still have time, try using a real vision detector?


### Implementation Plan

1. Two parts: a POMDP and a THOR manager

2. The POMDP part is essentially a 2D gridworld, in the same way as field2d,
   but with a gridmap that has obstacles.

3. The POMDP agent outputs actions that get converted into Teleop commands for THOR

4. 2D poses are mapped 1-to-1 between POMDP and THOR environment.


### Issues with RoboThor

1. Cannot retrieve information about walls

2. Occlusion cannot be estimated correctly; There is no way to
   distinguish between a wall and an unreachable location

3. Spatial correlation based on proximity won't be implementable
   because of reason 1; Two locations may be close in euclidean
   distance but actually be separated by a wall.

4. Scalability: 0.25 is not manageable by the joint distribution.
   0.5 is ok but likely does not scale to more objects.

5. Overall the scene lacks information necessary to build a POMDP agent. The map
   is larger than desired for creating the distribution.

### iTHOR

iTHOR is a lot more manageable!
