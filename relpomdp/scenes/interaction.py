import os
import habitat_sim
from relpomdp.constants import *
from habitat_sim.utils.common import d3_40_colors_rgb
import numpy as np
from PIL import Image
import cv2

output_path = os.path.join("mobile_output/")

def apply_palette(s_im, palette):
    output = Image.new("P", (s_im.shape[1], s_im.shape[0]))
    output.putpalette(d3_40_colors_rgb)
    output.putdata((s_im.flatten() % 40).astype(np.uint8))
    output = output.convert("RGB")
    return np.array(output)


def make_video_cv2(observations, prefix="", open_vid=True, multi_obs=False):
    videodims = (720, 540)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(output_path + prefix + ".mp4", fourcc, 60, videodims)
    thumb_size = (int(videodims[0] / 5), int(videodims[1] / 5))
    outline_frame = np.ones((thumb_size[1] + 2, thumb_size[0] + 2, 3), np.uint8) * 150
    for ob in observations:

        # If in RGB/RGBA format, change first to RGB and change to BGR
        bgr_im_1st_person = ob["rgba_camera_1stperson"][..., 0:3][..., ::-1]

        if multi_obs:
            # embed the 1st person RBG frame into the 3rd person frame
            bgr_im_3rd_person = ob["rgba_camera_3rdperson"][..., 0:3][..., ::-1]
            resized_1st_person_rgb = cv2.resize(
                bgr_im_1st_person, thumb_size, interpolation=cv2.INTER_AREA
            )
            x_offset = 50
            y_offset_rgb = 50
            bgr_im_3rd_person[
                y_offset_rgb - 1 : y_offset_rgb + outline_frame.shape[0] - 1,
                x_offset - 1 : x_offset + outline_frame.shape[1] - 1,
            ] = outline_frame
            bgr_im_3rd_person[
                y_offset_rgb : y_offset_rgb + resized_1st_person_rgb.shape[0],
                x_offset : x_offset + resized_1st_person_rgb.shape[1],
            ] = resized_1st_person_rgb

            # embed the 1st person DEPTH frame into the 3rd person frame
            # manually normalize depth into [0, 1] so that images are always consistent
            d_im = np.clip(ob["depth_camera_1stperson"], 0, 10)
            d_im /= 10.0
            bgr_d_im = cv2.cvtColor((d_im * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            resized_1st_person_depth = cv2.resize(
                bgr_d_im, thumb_size, interpolation=cv2.INTER_AREA
            )
            y_offset_d = y_offset_rgb + 10 + thumb_size[1]
            bgr_im_3rd_person[
                y_offset_d - 1 : y_offset_d + outline_frame.shape[0] - 1,
                x_offset - 1 : x_offset + outline_frame.shape[1] - 1,
            ] = outline_frame
            bgr_im_3rd_person[
                y_offset_d : y_offset_d + resized_1st_person_depth.shape[0],
                x_offset : x_offset + resized_1st_person_depth.shape[1],
            ] = resized_1st_person_depth

            # embed the 1st person DEPTH frame into the 3rd person frame
            # manually normalize depth into [0, 1] so that images are always consistent
            s_im = ob["semantic_camera_1stperson"]
            # import pdb; pdb.set_trace()
            # s_im = cv2.cvtColor(s_im.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            s_im = apply_palette(s_im, d3_40_colors_rgb)
            
            resized_1st_person_semantic = cv2.resize(
                s_im, thumb_size, interpolation=cv2.INTER_AREA
            )
            
            y_offset_s = y_offset_rgb + (10 + thumb_size[1])*2
            bgr_im_3rd_person[
                y_offset_s - 1 : y_offset_s + outline_frame.shape[0] - 1,
                x_offset - 1 : x_offset + outline_frame.shape[1] - 1,
            ] = outline_frame
            bgr_im_3rd_person[
                y_offset_s : y_offset_s + resized_1st_person_semantic.shape[0],
                x_offset : x_offset + resized_1st_person_semantic.shape[1],
            ] = resized_1st_person_semantic

            # write the video frame
            video.write(bgr_im_3rd_person)
        else:
            # write the 1st person observation to video
            video.write(bgr_im_1st_person)
    video.release()
    if open_vid:
        os.system("open " + output_path + prefix + ".mp4")



def make_configuration(scene_path):
    """Creating configuration for a mobile robot with
    an RGB and a depth camera equipped. Also adding
    a camera in the simulation that is 3rd-person view"""
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    if not os.path.exists(scene_path):
        raise ValueError("Scene does not exist at: %s" % scene_path)
    backend_cfg.scene.id = os.path.join(scene_path)
    backend_cfg.enable_physics = True
    backend_cfg.physics_config_file = os.path.join(HABITAT_PATH,
                                                   "data/default.phys_scene_config.json")

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [540, 720]
    sensors = {
        "rgba_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "semantic_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": camera_resolution,
            "position": [0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "rgba_camera_3rdperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 1.0, 0.3],
            "orientation": [-45, 0.0, 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.SensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_specs.append(sensor_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])

def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.7, 1.0]
    # agent_state.position = [-0.15, -1.6, 1.0]
    agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())

    return observations



def main():
    replica_path = "/home/kaiyuzh/repo/Replica-Dataset/downloads/replica_v1/"
    scene_name = "apartment_1"
    scene_mesh = os.path.join(replica_path, scene_name, "habitat/mesh_semantic.ply")
    data_path = os.path.join(HABITAT_PATH, "data")

    # scene_mesh = os.path.join(data_path, "scene_datasets/habitat-test-scenes/apartment_1.glb")
    
    cfg = make_configuration(scene_mesh)
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)

    # Load the locobot
    locobot_template_id = sim.load_object_configs(
        str(os.path.join(data_path, "objects/locobot_merged"))
    )[0]
    
    # add robot object to the scene with the agent/camera SceneNode attached
    id_1 = sim.add_object(locobot_template_id, sim.agents[0].scene_node)
    sim.set_translation(np.array([1.75, -1.02, 0.4]), id_1)

    vel_control = sim.get_object_velocity_control(id_1)
    vel_control.linear_velocity = np.array([0, 0, -1.0])
    vel_control.angular_velocity = np.array([0.0, 2.0, 0])

    # simulate robot dropping into place
    observations = simulate(sim, dt=1.5, get_frames=True)

    vel_control.controlling_lin_vel = True
    vel_control.controlling_ang_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.ang_vel_is_local = True

    # simulate forward and turn
    observations += simulate(sim, dt=1.0, get_frames=True)

    vel_control.controlling_lin_vel = False
    vel_control.angular_velocity = np.array([0.0, 1.0, 0])

    # simulate turn only
    observations += simulate(sim, dt=1.5, get_frames=True)

    vel_control.angular_velocity = np.array([0.0, 0.0, 0])
    vel_control.controlling_lin_vel = True
    vel_control.controlling_ang_vel = True

    # simulate forward only with damped angular velocity (reset angular velocity to 0 after each step)
    observations += simulate(sim, dt=1.0, get_frames=True)

    vel_control.angular_velocity = np.array([0.0, -1.25, 0])

    # simulate forward and turn
    observations += simulate(sim, dt=2.0, get_frames=True)

    vel_control.controlling_ang_vel = False
    vel_control.controlling_lin_vel = False

    # simulate settling
    observations += simulate(sim, dt=3.0, get_frames=True)

    # video rendering with embedded 1st person view
    if not os.path.exists(output_path):
        os.mkdir(output_path)    
    make_video_cv2(observations, prefix="robot_control", open_vid=True, multi_obs=True)

    

if __name__ == "__main__":
    main()
