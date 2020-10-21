# Learning


The algorithm is very simple:

1. Randomly generate environments

2. For **correlation**, compute a count based on room + distance between every pair of classes

3. For **difficulty**,

   - Run the robot randomly in the training environments by picking a random unvisited location
   - Activate all detectors
   - Record observations at each step
   - Compute a difficulty score of finding an object of a class to be proportional to the step that the robot saw that class

4. For **information gain**,

    - Essentially in the 2D grid world, it is the same thing as the correlation; If two objects are highly correlated,
      that means they are spatially close to each other, and finding one helps finding the other.

5. Write a function that reads these files generated from training and then take as input (class1, class2), and output
   a score.
