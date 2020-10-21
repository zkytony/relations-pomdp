import argparse
import pickle
from relpomdp.home2d.utils import euclidean_dist
import pandas as pd
import os

def get_classes(envs):
    """Returns a set of object classes, including room
    categories from the given environments"""
    classes = set()
    room_types = set()
    for envid in envs:
        grid_map = envs[envid].grid_map
        for room_name in grid_map.rooms:
            room_type = grid_map.rooms[room_name].room_type
            classes.add(room_type)
            room_types.add(room_type)

        for objid in envs[envid].state.object_states:
            objclass = envs[envid].state.object_states[objid].objclass
            classes.add(objclass)
    return classes, room_types


def compute_obj_obj_correlation(class1, class2, envs, dist_thresh=1):
    """
    Computes a number to represent the "spatial" correlation between
    class1 and class2, over the given environments.

    This is currently very simple; In each environment, obtain all
    occurrences of objects of these two classes. If there are two instances
    of each class appearing in the same room, then +1 per pair. If this is true,
    and that the two instances appear within a close distance (`dist_thresh`),
    then +1 again.
    """
    correlation_score = 0
    if class1 == "Kitchen" and class2 == "Salt":
        import pdb; pdb.set_trace()
    for envid in envs:
        instances = {class1:{}, class2:{}}  # maps from class->{room_name->[]}}
        grid_map = envs[envid].grid_map
        for objid in envs[envid].state.object_states:
            s = envs[envid].state.object_states[objid]
            if s.objclass == class1 or s.objclass == class2:
                room = grid_map.room_of(s["pose"][:2])
                if room.name not in instances[s.objclass]:
                    instances[s.objclass][room.name] = []
                instances[s.objclass][room.name].append(objid)
        for room_name in grid_map.rooms:
            if room_name in instances[class1]\
               and room_name in instances[class2]:
                for objid_c1 in instances[class1][room_name]:
                    for objid_c2 in instances[class2][room_name]:
                        if objid_c1 == objid_c2:
                            continue
                        correlation_score += 1

                        pose_c1 = envs[envid].state.object_states[objid_c1]["pose"]
                        pose_c2 = envs[envid].state.object_states[objid_c2]["pose"]
                        if euclidean_dist(pose_c1, pose_c2) <= dist_thresh:
                            correlation_score += 1
    return correlation_score

def compute_obj_room_correlation(objclass, room_type, envs):
    correlation_score = 0
    for envid in envs:
        # Check if the room of object class instance lies in a room of given room type
        grid_map = envs[envid].grid_map
        for objid in envs[envid].state.object_states:
            s = envs[envid].state.object_states[objid]
            room = grid_map.room_of(s["pose"][:2])
            if room.room_type == room_type:
                correlation_score += 1
    return correlation_score


def main():
    parser = argparse.ArgumentParser(description="Compute correlations for a given set of environments")
    parser.add_argument("path_to_envs",
                        type=str, help="Path to a .pickle file that contains a collection of environments")
    parser.add_argument("output_dir",
                        type=str, help="Directory to output computed correlations saved in a file")
    args = parser.parse_args()

    with open(args.path_to_envs, "rb") as f:
        envs = pickle.load(f)

    filename = os.path.splitext(os.path.basename(args.path_to_envs))[0]

    # Obtain a list of classes
    classes, room_types = get_classes(envs)
    rows = []
    for class1 in classes:
        for class2 in classes:
            if class1 in room_types and class2 in room_types:
                continue # skip room-room
            elif class1 in room_types:
                score = compute_obj_room_correlation(class2, class1, envs)
            elif class2 in room_types:
                score = compute_obj_room_correlation(class1, class2, envs)
            else:
                score = compute_obj_obj_correlation(class1, class2, envs)
            print("Correlation score between %s and %s is: %d" % (class1, class2, score))
            rows.append((class1, class2, score))
    df = pd.DataFrame(rows, columns=["class1", "class2", "corr_score"])
    df.to_csv(os.path.join(args.output_dir, "correlation-%s.csv" % filename))


if __name__ == "__main__":
    main()
