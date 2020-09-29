from relpomdp.home2d.domain.maps import all_maps
from relpomdp.home2d.planning.relations import *
from relpomdp.home2d.planning.belief_update import belief_update
from relpomdp.home2d.planning.grounding import next_grounding_task
from relpomdp.oopomdp.infograph import *
from relpomdp.pgm.mrf import SemanticMRF, factors_to_mrf
from relpomdp.utils import perplexity
from relpomdp.home2d.tasks.search_item.search_item_task import SearchItemTask
from relpomdp.home2d.tasks.search_room.search_room_task import SearchRoomTask
from relpomdp.home2d.utils import objstate, objobs, ooobs, euclidean_dist
from relpomdp.home2d.tasks.common.sensor import *
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.tasks.search_item.search_item_task import *
from relpomdp.home2d.tasks.search_item.visual import SearchItemViz
import pomdp_py


def office_floor1(init_robot_pose=(9,0,0)):
    """
    Office floor with salt, pepper, and computers
    This describes the full environment configuration (i.e. all
    object locations and the robot's location).
    """
    grid_map = all_maps["map_small_1"]()    
    init_salt_pose = (0,6)
    init_pepper_pose = (1,5)
    init_robot_pose = init_robot_pose

    salt_id = 10
    pepper_id = 15
    robot_id = 1
    robot_state = objstate("Robot",
                           pose=init_robot_pose,
                           camera_direction="-x")
    salt_state = objstate("Salt",
                          pose=init_salt_pose)
    pepper_state = objstate("Pepper",
                            pose=init_pepper_pose)
    
    computer_poses = [(5,9), (6,2)]
    computer_states = []
    for pose in computer_poses:
        computer_states.append(objstate("Computer",
                                        pose=pose))
    init_state = {robot_id: robot_state,
                  pepper_id: pepper_state,
                  salt_id: salt_state}
    for i, s in enumerate(computer_states):
        init_state[3000+i] = s

    # maps from object class to id        
    ids = {}  
    for objid in init_state:
        c = init_state[objid].objclass
        if c not in ids:
            ids[c] = []
        ids[c].append(objid)
    ids["Robot"] = ids["Robot"][0]
    colors = {"Salt": (128, 128, 128),
              "Pepper": (200, 10, 10)}

    pepper = PoseAttr("Pepper")
    salt = PoseAttr("Salt")
    computer = PoseAttr("Computer")
    kitchen = RoomAttr("Kitchen")
    office = RoomAttr("Office")
    
    near_ps = Near(pepper, salt)
    near_sp = Near(salt, pepper)
    not_near_sc = Near(salt, computer, negate=True)
    not_near_cs = Near(computer, salt, negate=True)    
    in_sk = In(salt, kitchen)
    in_pk = In(pepper, kitchen)
    not_in_so = In(salt, office, negate=True)
    graph = RelationGraph({near_ps, near_sp, not_near_sc, not_near_cs,
                           in_sk, in_pk, not_in_so})    

    return ids, grid_map, init_state, colors, graph


def setup():
    ids, grid_map, init_state, colors, graph = office_floor1()
    robot_id = ids["Robot"]
    salt_id = ids["Salt"][0]

    sensor = Laser2DSensor(robot_id, grid_map,  # the sensor uses the grid map for wall blocking
                           fov=90, min_range=1, max_range=2,
                           angle_increment=0.5)    
    task = SearchItemTask(robot_id, salt_id, sensor, grid_map=grid_map)
    
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state,
                            reward_model=task.reward_model)

    target_id = salt_id
    target_class = "Salt"
    env.state.object_states[salt_id]["is_found"] = False
    
    # Obtain prior
    prior_type = "uniform"
    target_hist = {}
    total_prob = 0
    for x in range(grid_map.width):
        for y in range(grid_map.length):
            state = objstate(target_class, pose=(x,y), is_found=False)
            if prior_type == "uniform":
                target_hist[state] = 1.0
            elif prior_type == "informed":
                if (x,y) != env.state.object_states[target_id]["pose"]:
                    target_hist[state] = 0.0
                else:
                    target_hist[state] = 1.0
            total_prob += target_hist[state]
    # Normalize
    for state in target_hist:
        target_hist[state] /= total_prob

    init_belief = pomdp_py.OOBelief({target_id: pomdp_py.Histogram(target_hist),
                                     robot_id: pomdp_py.Histogram({env.robot_state:1.0})})
    
    # Make agent
    agent = task.to_agent(init_belief)
    env.transition_model.cond_effects.append((CanPickup(task.robot_id, task.target_id),
                                              PickupEffect()))

    # Create planner and make a plan
    planner = pomdp_py.POUCT(max_depth=10,
                             discount_factor=0.95,
                             num_sims=100,
                             exploration_const=100,
                             rollout_policy=agent.policy_model)

    print("Creating visualization ...")
    objcolors = {}
    for objid in env.state.object_states:
        s_o = env.state.object_states[objid]
        if s_o.objclass in colors:
            objcolors[objid] = colors[s_o.objclass]    
    viz = SearchItemViz(env,
                        objcolors,
                        res=30)
    return env, agent, task, planner, viz, graph


def solve(env, agent, task, planner, viz, graph):
    viz.on_init()
    viz.update(agent.belief)    
    viz.on_render()

    used_cues = set()
    
    for step in range(100):
        print("---- Step %d ----" % step)
        action = planner.plan(agent)
        reward = env.state_transition(action, execute=True)
        observation = agent.observation_model.sample(env.state, action)

        # Belief update
        robot_state = env.state.object_states[task.robot_id]

        target_belief = agent.belief.object_beliefs[task.target_id]
        target_class = env.state.object_states[task.target_id].objclass        
        for objid in observation.object_observations:
            o_obj = observation.object_observations[objid]
            if objid != task.robot_id and objid not in used_cues:
                if o_obj.objclass == target_class:
                    # You just observed the target. MRF isn't useful here.
                    continue

                b_attr = graph.nodes["%s-pose" % target_class]
                if "pose" in o_obj.attributes:
                    o_attr = graph.nodes["%s-pose" % (o_obj.objclass)]
                
                target_belief = belief_update(target_belief,
                                              b_attr,
                                              graph,
                                              o_obj,
                                              o_attr,
                                              env.grid_map)
        
        target_oo_belief = pomdp_py.Histogram({
            oopomdp.OOState({task.target_id: target_state,
                             task.robot_id: robot_state}) : target_belief[target_state]
            for target_state in agent.belief.object_beliefs[task.target_id]})
        new_belief = pomdp_py.update_histogram_belief(target_oo_belief,
                                                      action, observation,
                                                      agent.observation_model,
                                                      agent.transition_model,
                                                      static_transition=True)
        # Take just the target state from this
        new_belief = pomdp_py.Histogram({state.object_states[task.target_id]:
                                         new_belief[state]
                                         for state in new_belief})
        agent.belief.set_object_belief(task.target_id, new_belief)
        agent.belief.set_object_belief(task.robot_id, pomdp_py.Histogram({robot_state:1.0}))
        planner.update(agent, action, observation)
        
        print("     action: %s" % str(action.name))        
        print("     reward: %s" % str(reward))
        print("observation: %s" % str(observation))
        print("robot state: %s" % str(robot_state))
        
        viz.update(agent.belief)
        viz.on_loop()
        viz.on_render()

        if env.state.object_states[task.target_id]["is_found"]:
            break
    print("Done.")


if __name__ == "__main__":
    env, agent, task, planner, viz, graph = setup()
    solve(env, agent, task, planner, viz, graph)    
