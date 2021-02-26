from corrsearch.experiments.domains.thor.topo_maps.graph import Node, Graph, Edge
from corrsearch.utils import euclidean_dist
import json
import yaml
import os
from collections import deque

class TopoNode(Node):
    def __init__(self, id, world_pose):
        self.id = id
        self.pose = world_pose
        self._coords = self.pose  # for visualization
        self._color = "orange"

class TopoMap(Graph):

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            data = json.load(f)

        nodes = {}
        for node_id in data["nodes"]:
            node_data = data["nodes"][node_id]
            x, z = node_data["x"], node_data["z"]
            nodes[int(node_id)]= TopoNode(int(node_id), (x,z))

        edges = {}
        for i, edge in enumerate(data["edges"]):
            node_id1, node_id2 = edge[0], edge[1]
            n1 = nodes[int(node_id1)]
            n2 = nodes[int(node_id2)]
            edges[i] = Edge(i, n1, n2,
                            data=euclidean_dist(n1.pose, n2.pose))

        return TopoMap(edges)

    def closest_node(self, x, z):
        """Given a point at (x,z) in world frame
        find the node that is closest to this point."""
        return min(self.nodes,
                   key=lambda nid: euclidean_dist(self.nodes[nid].pose[:2], (x,z)))

    def navigable(self, nid1, nid2):
        # DFS find path from nid1 to nid2
        stack = deque()
        stack.append(nid1)
        visited = set()
        while len(stack) > 0:
            nid = stack.pop()
            if nid == nid2:
                return True
            for neighbor_nid in self.neighbors(nid):
                if neighbor_nid not in visited:
                    stack.append(neighbor_nid)
                    visited.add(neighbor_nid)
        return False

    def to_json(self):
        result = {"nodes": {}, "edges": []}
        for nid in self.nodes:
            pose = self.nodes[nid].pose
            result["nodes"][nid] = {"x": pose[0], "z": pose[1]}
        for eid in self.edges:
            edge = self.edges[eid]
            node1, node2 = edge.nodes
            result["edges"].append([node1.id, node2.id])
        return result
