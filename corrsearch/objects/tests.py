from corrsearch.objects import ObjectState, JointState, ObjectObz, JointObz
import pomdp_py
import unittest

class TestObjectState(unittest.TestCase):
    def setUp(self):
        self.so1 = ObjectState(1, "cup",
                               {"location": (5,3),
                                "temperature": 70})
        self.so2 = ObjectState(2, "juice",
                               {"location": (5,5),
                                "kind": "apple"})
        self.so3 = ObjectState(3, "bag",
                               {"location": (5,10),
                                "contains": (1, 2),
                                "owner": ("Bob", 40, "worker")})
        self.s = JointState({self.so1, self.so2, self.so3})

    def test_inheritence(self):
        self.assertIsInstance(self.so1, pomdp_py.State)

    def test_hashable_attributes_only(self):
        with self.assertRaises(TypeError):
            so3 = ObjectState(3, "bag",
                              {"items": {1, 2}})
        # OK if it's tuple
        so3 = ObjectState(3, "bag",
                          {"items": (1, 2)})

    def test_objid_checking(self):
        so4 = ObjectState(1, "cup", {})
        with self.assertRaises(AssertionError):
            JointState({self.so1, self.so2, self.so3, so4})
        so4 = ObjectState(4, "cup", {})
        with self.assertRaises(AssertionError):
            JointState({self.so1.id:self.so1,
                        self.so1.id: so4})

    def test_copy_equal(self):
        self.assertEqual(self.so1, self.so1.copy())
        self.assertEqual(self.so2, self.so2.copy())
        self.assertEqual(self.s, self.s.copy())

    def test_immutable(self):
        with self.assertRaises(NotImplementedError):
            self.so1["location"] = (5,5)
        with self.assertRaises(NotImplementedError):
            self.s[self.so1.id] = self.so2


    def test_hashable(self):
        hash(self.so1)
        hash(self.so2)


class TestObjectObz(unittest.TestCase):
    def setUp(self):
        self.zo1 = ObjectObz(1, "cup",
                             {"location": (5,3),
                              "temperature": 70})
        self.zo2 = ObjectObz(2, "juice",
                             {"location": (5,5),
                              "kind": "apple"})
        self.zo3 = ObjectObz(3, "bag",
                             {"location": (5,10),
                              "contains": (1, 2),
                              "owner": ("Bob", 40, "worker")})
        self.z = JointObz({self.zo1, self.zo2, self.zo3})

    def test_inheritence(self):
        self.assertIsInstance(self.zo1, pomdp_py.Observation)

    def test_hashable_attributes_only(self):
        with self.assertRaises(TypeError):
            zo3 = ObjectObz(3, "bag",
                              {"items": {1, 2}})
        # OK if it's tuple
        zo3 = ObjectObz(3, "bag",
                          {"items": (1, 2)})

    def test_objid_checking(self):
        z4 = ObjectObz(1, "cup", {})
        with self.assertRaises(AssertionError):
            JointObz({self.zo1, self.zo2, self.zo3, z4})
        z4 = ObjectObz(4, "cup", {})
        with self.assertRaises(AssertionError):
            JointObz({self.zo1.id:self.zo1,
                      self.zo1.id: z4})

    def test_copy_equal(self):
        self.assertEqual(self.zo1, self.zo1.copy())
        self.assertEqual(self.zo2, self.zo2.copy())
        self.assertEqual(self.z, self.z.copy())

    def test_immutable(self):
        with self.assertRaises(NotImplementedError):
            self.zo1["location"] = (5,5)
        with self.assertRaises(NotImplementedError):
            self.z[self.zo1.id] = self.zo2


    def test_hashable(self):
        hash(self.zo1)
        hash(self.zo2)


if __name__ == "__main__":
    unittest.main()
