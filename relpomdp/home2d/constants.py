import os
HOME2D_PATH = os.path.dirname(os.path.abspath(__file__))

FILE_PATHS = {}
FILE_PATHS["colors"] = os.path.join(HOME2D_PATH, "learning", "configs", "colors.yaml")
FILE_PATHS["object_imgs"] = os.path.join(HOME2D_PATH, "domain", "imgs")
