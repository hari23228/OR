import unittest

from backend.main import clear_results, get_routing_export_rows, store


class ExportRoutingSelectionTests(unittest.TestCase):
    def setUp(self):
        clear_results()

    def test_clear_results_resets_latest_routing_strategy(self):
        store.results["ilp"] = {"status": "Optimal", "routing": [{"query_id": "1", "model": "model-a"}]}
        store.latest_routing_strategy = "ilp"

        clear_results()

        self.assertEqual(store.results, {})
        self.assertIsNone(store.latest_routing_strategy)

    def test_latest_strategy_returns_only_the_most_recent_successful_routing(self):
        store.results["ilp"] = {"status": "Optimal", "routing": [{"query_id": "1", "model": "model-a"}]}
        store.latest_routing_strategy = "ilp"
        store.results["goal"] = {"status": "Optimal", "routing": [{"query_id": "2", "model": "model-b"}]}
        store.latest_routing_strategy = "goal"

        rows = get_routing_export_rows("latest")

        self.assertEqual(rows, [{"query_id": "2", "model": "model-b"}])


if __name__ == "__main__":
    unittest.main()
