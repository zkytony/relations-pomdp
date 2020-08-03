import os
import habitat_sim
# THIS IS SUPER HELPFUL
#
# https://colab.research.google.com/drive/14t0oOuDTKgH-awUohwoJZ79nus96BmRS?usp=sharing#scrollTo=1UUJWw3GyUYP


replica_path = "/home/kaiyuzh/repo/Replica-Dataset/downloads/replica_v1/"
scene = "apartment_0"
scene_mesh = os.path.join(replica_path, scene, "habitat/mesh_semantic.ply")

sim_settings = {
    "width": 512,
    "height": 512,
    "scene": scene_mesh,
    "default_agent": 0,
    "sensor_height": 1,
    "color_sensor": True,
    "semantic_sensor": True,
    "depth_sensor": True,
    "seed": 1
}


sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.gpu_device_id = 0
sim_cfg.scene.id = sim_settings["scene"]

# Note: all sensors must have the same resolution
sensors = {
    "color_sensor": {
        "sensor_type": habitat_sim.SensorType.COLOR,
        "resolution": [sim_settings["height"], sim_settings["width"]],
        "position": [0.0, sim_settings["sensor_height"], 0.0],
    },
    "depth_sensor": {
        "sensor_type": habitat_sim.SensorType.DEPTH,
        "resolution": [sim_settings["height"], sim_settings["width"]],
        "position": [0.0, sim_settings["sensor_height"], 0.0],
    },
    "semantic_sensor": {
        "sensor_type": habitat_sim.SensorType.SEMANTIC,
        "resolution": [sim_settings["height"], sim_settings["width"]],
        "position": [0.0, sim_settings["sensor_height"], 0.0],
    },
}

sensor_specs = []
for sensor_uuid, sensor_params in sensors.items():
    print (sensor_uuid)
    if sim_settings[sensor_uuid]:
        sensor_spec = habitat_sim.SensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]

        sensor_specs.append(sensor_spec)

# Here you can specify the amount of displacement in a forward action and the turn angle
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = sensor_specs
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
    ),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
    ),
    "turn_right": habitat_sim.agent.ActionSpec(
        "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
    ),
}

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

def print_scene_recur(scene, limit_output=10):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for obj in scene.objects:
        if obj is not None:
            category = "Unknown" if obj.category is None else obj.category.name()
            print(
                f"Object id:{obj.id}, category:{category},"
                f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            )
            count += 1
            if count >= limit_output:
                return None

import pdb; pdb.set_trace()
scene = sim.semantic_scene
print_scene_recur(scene)
