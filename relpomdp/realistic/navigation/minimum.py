from ai2thor.controller import Controller

controller = Controller(scene="FloorPlan_Train1_1")

# Initial pose
event = controller.step("Pass")
print("position1:", event.metadata["agent"]["position"])
print("rotation1:", event.metadata["agent"]["rotation"])

# I want the agent to teleport to:
#   x=3.25, z=-1.5, rotation_y=270.0
# without changing other coordinates
event = controller.step('TeleportFull',
                        x=3.25, z=-1.5,
                        rotation=dict(y=270.0))
print("position2:", event.metadata["agent"]["position"])
print("rotation2:", event.metadata["agent"]["rotation"])
