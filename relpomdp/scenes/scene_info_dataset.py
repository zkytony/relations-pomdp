
# Generates a table:
#
#  scene_name  object_id  object_category  center  dims
#
# For a list of scenes


import os
import habitat_sim
import pandas as pd

replica_path = "/home/kaiyuzh/repo/Replica-Dataset/downloads/replica_v1/"
scenes = [
    "apartment_0",
    "apartment_1",
    "apartment_2",
    "frl_apartment_0",
    "frl_apartment_1",
    "frl_apartment_2",
    "frl_apartment_3",
    "frl_apartment_4",
    "frl_apartment_5",
    "hotel_0",
    "office_0",      
    "office_1",       
    "office_2",       
    "office_3",       
    "office_4",       
    "room_0",         
    "room_1",
    "room_2",
]

sim_settings = {
    "width": 512,
    "height": 512,
    "default_agent": 0,
    "sensor_height": 1,
    "color_sensor": True,
    "semantic_sensor": True,
    "depth_sensor": True,
    "seed": 1
}

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

data = []
for scene_name in scenes:
    print("======Working on %s======" % scene_name)
    scene_mesh = os.path.join(replica_path, scene_name, "habitat/mesh_semantic.ply")
    sim_settings["scene"] = scene_mesh
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene.id = sim_settings["scene"]
    # Note: all sensors must have the same resolution
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    scene = sim.semantic_scene
    
    for obj in scene.objects:
        if obj is not None and obj.category is not None:
            #  scene_name  object_id  object_category  center_x, center_y, center_z, size_x, size_y, size_z
            x, y, z = obj.aabb.center
            lx, ly, lz = obj.aabb.sizes
            entry = [scene_name, obj.id, obj.category.name(), x, y, z, lx, ly, lz]
            data.append(entry)
    sim.close()

df = pd.DataFrame(data, columns=["scene_name", "object_id", "category", "x", "y", "z", "size_x", "size_y", "size_z"])
df.to_csv("replica_objects.csv")
