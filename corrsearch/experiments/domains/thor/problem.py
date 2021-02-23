
import yaml
import pomdp_py
import random
from corrsearch.models import *
from corrsearch.objects import *
from corrsearch.utils import *


class ThorSearch(SearchProblem):

    def __init__(self, robot_id):
        self.robot_id = robot_id

    def instantiate(self, **kwargs):
        """
        Instantiate an instance of the search problem in a Thor scene.
        """

    def obj(self, objid):
        return {}
