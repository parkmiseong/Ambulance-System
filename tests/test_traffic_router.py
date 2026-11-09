"""Traffic refresh keeps live costs while avoiding repeated geometry/socket reads."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
import traffic_router
from traci.constants import VAR_ROAD_ID


class TrafficRouterTests(unittest.TestCase):
    def test_subscriptions_and_geometry_reuse_preserve_live_costs(self):
        lane = SimpleNamespace(getLength=lambda: 100., getSpeed=lambda: 10.)
        net = SimpleNamespace(getEdge=Mock(return_value=SimpleNamespace(getLanes=lambda: [lane])))
        fake = SimpleNamespace(
            simulation=SimpleNamespace(getTime=Mock(side_effect=[0., 30., 60.])),
            vehicle=SimpleNamespace(
                getIDList=Mock(side_effect=[['v'], ['v'], []]),
                subscribe=Mock(), getRoadID=Mock(),
                getAllSubscriptionResults=Mock(return_value={'v': {VAR_ROAD_ID: 'e'}})),
            edge=SimpleNamespace(getLastStepMeanSpeed=Mock(side_effect=[5., 2.]),
                                 adaptTraveltime=Mock()),
            TraCIException=RuntimeError)
        router = traffic_router.TrafficRouter(net=net)
        with patch.object(traffic_router, 'traci', fake):
            self.assertEqual([router.refresh() for _ in range(3)], [1, 1, 0])
        fake.vehicle.subscribe.assert_called_once_with('v', (VAR_ROAD_ID,))
        fake.vehicle.getRoadID.assert_not_called()
        net.getEdge.assert_called_once_with('e')
        self.assertEqual([call.args for call in fake.edge.adaptTraveltime.call_args_list],
                         [('e', 20.), ('e', 50.), ('e', 10.)])
        router.reset()
        self.assertFalse(router.subscribed)
        self.assertFalse(router.geometry)


if __name__ == '__main__':
    unittest.main()
