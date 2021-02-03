# relpomdp2

This is an attempt to investigate the optimal planning behavior
for the problem of correlated object search. The question was
whether the detector of an intermediate object would be fired
during the search for the target object, and how much impact
does using such intermediate object detector has.

The main content of this package is the **hallway search** domain,
a 1D instance of the problem that we care about.
The `hallway_xy` folder contains implementation of the version
with only two objects. `hallway` allows for any number of object.
I implemented a joint distribution in `probability.py`.


## Results

Return vs. Increasing hallway length (perfect sensors)

![Return vs. Increasing hallway length](https://i.imgur.com/3HKZpXS.png)

Return vs. Increasing noise for target detector

![Return vs. Increasing noise](https://i.imgur.com/Hkk8FiV.png)

Example

![Example](https://i.imgur.com/78P3x7y.gif)
