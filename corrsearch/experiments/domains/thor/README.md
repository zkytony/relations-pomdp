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