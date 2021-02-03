import unittest
from tabular_dist import TabularDistribution

### Tests
class TestTabularDistribution(unittest.TestCase):
    def setUp(self):
        variables = ["X", "Y"]
        weights = [
            (('x1', 'y1'), 0.8),
            (('x1', 'y2'), 0.2),
            (('x2', 'y1'), 0.3),
            (('x2', 'y2'), 0.1),
            (('x3', 'y1'), 0.2),
            (('x3', 'y2'), 0.6)
        ]
        self.pxy = TabularDistribution(variables, weights)

    def test_prob_normalization(self):
        var = "X"
        weights = [('x1', 1.0), ('x2', 1.0), ('x3', 1.0)]
        px = TabularDistribution([var], weights)
        self.assertTrue(px.prob({'X':'x1'}) == px.prob({'X':'x2'}) == 1.0 / 3)

    def test_prob_joint_dist(self):
        pxy = self.pxy
        self.assertEqual(pxy.prob(('x1','y2')), pxy.prob(('x3', 'y1')))

    def test_condition(self):
        pxy = self.pxy
        pxgy = pxy.condition({'Y': 'y1'})
        self.assertEqual(pxgy.variables, ["X"])
        self.assertEqual(sum(pxgy.prob(event) for event in pxgy.probs), 1.0)

    def test_sum_out(self):
        pxy = self.pxy
        px = pxy.sum_out(["Y"])
        self.assertEqual(px.prob(("x1",)), (pxy.prob(("x1", "y1")) + pxy.prob(("x1", "y2"))))
        self.assertEqual(px.prob(("x2",)), (pxy.prob(("x2", "y1")) + pxy.prob(("x2", "y2"))))

    def test_to_df(self):
        self.pxy.to_df()

    def test_sample(self):
        counts = {}
        for i in range(50000):
            ev = self.pxy.sample()
            counts[ev] = counts.get(ev, 0) + 1
        total_counts = sum(counts.values())
        counts = {ev:counts[ev]/total_counts
                  for ev in counts}
        for ev in counts:
            self.assertAlmostEqual(counts[ev],
                                   self.pxy.prob(ev),
                                   places=2)

    def test_missing_value(self):
        variables = ["X", "Y"]
        weights = [
            (('x1', 'y1'), 0.8),
            (('x1', 'y2'), 0.2),
            (('x2', 'y2'), 0.1),
        ]
        pxy = TabularDistribution(variables, weights)
        self.assertEqual(pxy.prob({"X":"x2", "Y":"y1"}), 0.0)

if __name__ == "__main__":
    unittest.main()
