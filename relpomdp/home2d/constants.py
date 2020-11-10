import os
HOME2D_PATH = os.path.dirname(os.path.abspath(__file__))

FILE_PATHS = {}
FILE_PATHS["colors"] = os.path.join(HOME2D_PATH, "experiments", "configs", "colors.yaml")
FILE_PATHS["object_imgs"] = os.path.join(HOME2D_PATH, "domain", "imgs")
FILE_PATHS["exp_data"] = os.path.join(HOME2D_PATH, "experiments", "data")
FILE_PATHS["exp_config"] = os.path.join(HOME2D_PATH, "experiments", "configs")
FILE_PATHS["exp_worlds"] = os.path.join(HOME2D_PATH, "experiments", "worlds")
