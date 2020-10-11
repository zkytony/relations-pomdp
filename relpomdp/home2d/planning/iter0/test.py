from relpomdp.home2d.domain.maps import all_maps
from relpomdp.home2d.planning.iter0.relations import *
from relpomdp.home2d.planning.iter0.belief_update import relation_belief_update
from relpomdp.home2d.planning.iter0.grounding import next_grounding_task
from relpomdp.oopomdp.infograph import *
from relpomdp.pgm.mrf import SemanticMRF, factors_to_mrf
from relpomdp.utils import perplexity
from relpomdp.home2d.tasks.search_item.search_item_task import SearchItemTask
from relpomdp.home2d.tasks.search_room.search_room_task import SearchRoomTask
from relpomdp.home2d.utils import euclidean_dist, save_images_and_compress
from relpomdp.oopomdp.framework import Objstate, Objobs, OOObs
from relpomdp.home2d.tasks.common.sensor import *
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.tasks.search_item.search_item_task import *
from relpomdp.home2d.tasks.search_item.visual import SearchItemViz
from relpomdp.utils import perplexity
import subprocess
import pomdp_py

discount_factor = 0.95


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
    robot_state = Objstate("Robot",
                           pose=init_robot_pose,
                           camera_direction="-x")
    salt_state = Objstate("Salt",
                          pose=init_salt_pose)
    pepper_state = Objstate("Pepper",
                            pose=init_pepper_pose)
    
    computer_poses = [(5,9), (6,2)]
    computer_states = []
    for pose in computer_poses:
        computer_states.append(Objstate("Computer",
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

    # Attributes
    pepper = PoseAttr("Pepper")
    salt = PoseAttr("Salt")
    computer = PoseAttr("Computer")
    kitchen = RoomAttr("Kitchen")
    office = RoomAttr("Office")

    # Relations
    near_ps = Near(pepper, salt)
    near_sp = Near(salt, pepper)
    not_near_sc = Near(salt, computer, negate=True)
    not_near_cs = Near(computer, salt, negate=True)    
    in_sk = In(salt, kitchen)
    in_pk = In(pepper, kitchen)
    # in_co = In(computer, office)
    # not_in_ck = In(computer, kitchen, negate=True)    
    not_in_so = In(salt, office, negate=True)
    graph = RelationGraph({near_ps, near_sp, not_near_sc, not_near_cs,
                           in_sk, in_pk, not_in_so})#, not_in_ck, in_co})    

    return ids, grid_map, init_state, colors, graph


def setup():
    ids, grid_map, init_state, colors, graph = office_floor1()
    robot_id = ids["Robot"]
    salt_id = ids["Salt"][0]

    target_id = salt_id
    target_class = init_state[target_id].objclass
    
    sensor = Laser2DSensor(robot_id, grid_map,  # the sensor uses the grid map for wall blocking
                           fov=90, min_range=1, max_range=2,
                           angle_increment=0.5)    
    task = SearchItemTask(robot_id, target_id, target_class, sensor, grid_map=grid_map)    
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state,
                            reward_model=task.reward_model)

    env.state.object_states[salt_id]["is_found"] = False
    
    # Obtain prior
    prior_type = "uniform"
    init_belief = pomdp_py.OOBelief({target_id: task.get_prior(grid_map, prior_type=prior_type),
                                     robot_id: pomdp_py.Histogram({env.robot_state:1.0})})
    
    # Make agent
    agent = task.to_agent(init_belief)
    env.transition_model.cond_effects.append((CanPickup(task.robot_id, task.target_id),
                                              PickupEffect()))

    # Create planner and make a plan
    planner = pomdp_py.POUCT(max_depth=15,
                             discount_factor=discount_factor,
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
                        res=30,
                        img_path="../../domain/imgs")
    return env, agent, task, planner, viz, graph


def item_search_belief_update(task, agent, observation,
                              action, robot_state, used_cues=set(),
                              evidence_from_subtasks=[]):
    """
    used_cues (set): Stores identifiers of the evidence that
        have already been used for belief update, so they
        shouldn't be used again. [IS THIS NECESSARY?]
    evidence_from_subtasks (list): List of evidence obtained
        when subtasks are finished (they are individual observations
        which can be used to update the belief).
    """
    target_belief = agent.belief.object_beliefs[task.target_id]

    all_observations = [observation.object_observations[objid]
                        for objid in observation.object_observations
                        if objid != task.robot_id] + evidence_from_subtasks
    for o in all_observations:
        if o.objclass == task.target_class:
            # You just observed the target. MRF isn't useful here.
            continue

        b_attr = graph.nodes["%s-pose" % task.target_class]
        if "pose" in o.attributes:
            o_attr = graph.nodes["%s-pose" % (o.objclass)]
        elif "room_id" in o.attributes:
            o_attr = graph.nodes["%s-room_id" % (o.objclass)]

        target_belief = relation_belief_update(target_belief,
                                               b_attr,
                                               graph,
                                               o,
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
    

def solve(env, agent, task, planner, viz, graph):
    game_states = []
    viz.on_init()
    viz.update(agent.belief)    
    img = viz.on_render()
    game_states.append(img)

    used_cues = set()

    subtask = None
    subtask_agent = None
    subtask_env = None
    subtask_planner = None

    evidence_from_subtasks = []  # evidence accumulated after solving subgoals

    discount = 1.0
    disc_reward = 0
    for step in range(100):
        print("---- Step %d ----" % step)
        if subtask is None:
            target_belief = agent.belief.object_beliefs[task.target_id]
            b_attr = graph.nodes["%s-pose" % task.target_class]
            subtask, plx_sub = next_grounding_task(target_belief,
                                                   b_attr,
                                                   graph,
                                                   env.grid_map,
                                                   task.robot_id)
            plx_global = perplexity([target_belief[s] for s in target_belief])
            if plx_global > plx_sub:
                # Doing a subtask is worth it
                subtask_prior = subtask.get_prior(env.grid_map, prior_type="uniform",
                                                  robot_state=env.robot_state)
                subtask_agent = subtask.to_agent(subtask_prior)
                subtask_env = subtask.get_env(env)
                subtask_planner = pomdp_py.POUCT(max_depth=15,
                                                 discount_factor=discount_factor,
                                                 num_sims=100,
                                                 exploration_const=100,
                                                 rollout_policy=subtask_agent.policy_model)
            else:
                subtask = None
                subtask_agent = None
                subtask_env = None
                subtask_planner = None

        if subtask_agent is not None:
            print("SOLVING _SUBTASK_ %s" % subtask)
            action, _, _ = subtask.step(subtask_env, subtask_agent, subtask_planner)
        else:
            print("SOLVING G L O B A L task %s" % task)
            action = planner.plan(agent)
        # Still, obtain reward and observation for global task
        reward = env.state_transition(action, execute=True)
        observation = agent.observation_model.sample(env.state, action)

        # Belief update
        robot_state = env.state.object_states[task.robot_id]
        item_search_belief_update(task, agent, observation, action, robot_state,
                                  used_cues,
                                  evidence_from_subtasks=evidence_from_subtasks)
        planner.update(agent, action, observation)

        disc_reward += discount*reward
        discount*=discount_factor
        print("     action: %s" % str(action.name))        
        print("     reward: %s" % str(reward))
        print(" disc cum R: %s" % str(disc_reward))
        print("observation: %s" % str(observation))
        print("robot state: %s" % str(robot_state))
        
        viz.update(agent.belief)
        viz.on_loop()
        img = viz.on_render()
        game_states.append(img)

        if task.is_done(env):
            break
        if subtask is not None and subtask.is_done(env, action):
            # TODO: Is this call of "get_result" generalizable?
            evidence_from_subtasks.append(subtask.get_result(subtask_agent, env.grid_map))
            subtask = None
            subtask_agent = None
            subtask_env = None            
            subtask_planner = None

    print("Saving images...")
    dirp = "./game_states"
    save_images_and_compress(game_states,
                             dirp)
    subprocess.Popen(["nautilus", dirp])
            
    print("Done.")


if __name__ == "__main__":
    env, agent, task, planner, viz, graph = setup()
    solve(env, agent, task, planner, viz, graph)    
