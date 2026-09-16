import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from orchestrator.DDQN_scheduler_pytorch import DDQNScheduler


class TestPyTorchDDQNScheduler(unittest.TestCase):
    def setUp(self):
        self.node_controller = MagicMock()
        self.node_controller.nodes = {
            "node1": MagicMock(
                allocated_cpu=0,
                allocated_memory=0,
                allocated_gpu=0,
                total_cpu=4,
                total_memory=16,
                total_gpu=2,
                status="Ready",
            ),
            "node2": MagicMock(
                allocated_cpu=0,
                allocated_memory=0,
                allocated_gpu=0,
                total_cpu=4,
                total_memory=16,
                total_gpu=2,
                status="Ready",
            ),
        }
        self.scheduler = DDQNScheduler(self.node_controller)
        self.scheduler._update_action_size()

    def test_model_creation(self):
        self.assertIsInstance(self.scheduler.model, torch.nn.Module)
        self.assertIsInstance(self.scheduler.target_model, torch.nn.Module)

    def test_predict_shape(self):
        state = np.zeros((1, self.scheduler.state_size), dtype=np.float32)
        output = self.scheduler.predict(state)
        self.assertEqual(output.shape, (1, self.scheduler.action_size))

    def test_memory_remember(self):
        state = np.zeros((1, self.scheduler.state_size), dtype=np.float32)
        self.scheduler.remember(state, 0, 1.0, state, False)
        self.assertEqual(len(self.scheduler.memory), 1)

    def test_training_step(self):
        state = np.zeros((1, self.scheduler.state_size), dtype=np.float32)
        target = np.zeros((1, self.scheduler.action_size), dtype=np.float32)
        loss = self.scheduler.train_model(state, target)
        self.assertIsInstance(loss, float)

    def test_target_network_update(self):
        with torch.no_grad():
            for parameter in self.scheduler.model.parameters():
                parameter.add_(1.0)
        self.scheduler.update_target_network()
        for online, target in zip(
            self.scheduler.model.parameters(), self.scheduler.target_model.parameters()
        ):
            self.assertTrue(torch.equal(online, target))


if __name__ == "__main__":
    unittest.main()
