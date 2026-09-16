"""PyTorch reference implementation of the gmnkube learning-based scheduler.

This module mirrors the public TensorFlow/Keras scheduler in
``orchestrator/DDQN_scheduler.py`` while keeping the rest of the scheduling
interface unchanged.  The original TensorFlow implementation remains in the
repository; this file provides an alternative PyTorch implementation for
reproduction and extension.
"""

from collections import deque
import datetime
import logging
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

NODE_COUNT = 10


class QNetwork(nn.Module):
    """Q-network matching the architecture of the TensorFlow implementation."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, action_size),
        )

    def forward(self, x):
        return self.network(x)


class DDQNScheduler:
    """Drop-in PyTorch port of the existing DDQN scheduler interface."""

    def __init__(self, node_controller):
        self.config = {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "batch_size": 8,
        }
        self.node_controller = node_controller
        self.state_size = NODE_COUNT * 9
        self.action_size = NODE_COUNT
        self.memory = deque(maxlen=2000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.loss_fn = nn.MSELoss()
        self.update_target_network()

        self.update_target_frequency = 10
        self.update_counter = 0
        self.schedule_history = []

    def _build_model(self):
        return QNetwork(self.state_size, self.action_size).to(self.device)

    def _build_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def _update_action_size(self):
        self.action_size = len(self.node_controller.nodes)
        self.state_size = 9 * self.action_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.update_target_network()

    def _as_tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def train_model(self, state, target_f):
        self.model.train()
        state_tensor = self._as_tensor(state)
        target_tensor = self._as_tensor(target_f)

        prediction = self.model(state_tensor)
        loss = self.loss_fn(prediction, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def predict(self, state):
        self.model.eval()
        state_tensor = self._as_tensor(state)
        with torch.no_grad():
            values = self.model(state_tensor)
        return values.detach().cpu().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.action_size == 0:
            logging.error("No nodes available for scheduling.")
            return 0
        if np.random.rand() <= self.config["epsilon"]:
            return self.select_best_node(state)
        act_values = self.predict(state)
        return int(np.argmax(act_values[0]))

    def replay(self):
        """Replay training matching the behavior of the TensorFlow scheduler."""
        if len(self.memory) < self.config["batch_size"]:
            return

        minibatch = random.sample(self.memory, self.config["batch_size"])
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                self.target_model.eval()
                with torch.no_grad():
                    next_state_prediction = self.target_model(self._as_tensor(next_state))
                target += self.config["gamma"] * float(
                    torch.max(next_state_prediction).detach().cpu().item()
                )

            target_f = self.predict(state).copy().reshape(1, -1)
            target_f[0][action] = target
            self.train_model(state, target_f)

        if self.config["epsilon"] > self.config["epsilon_min"]:
            self.config["epsilon"] *= self.config["epsilon_decay"]

        self.update_counter += 1
        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def schedule_pod(self, pod):
        if self.action_size != len(self.node_controller.nodes):
            self._update_action_size()

        state = self._get_state(pod)
        action = self.act(state)
        node_name = self._get_node_from_action(action)

        logging.info(
            f"[DDQN-Scheduler-PyTorch-INFO]: Trying to schedule Pod "
            f"{pod.name} to Node {node_name}. Action: {action}"
        )
        try:
            reward = self._calculate_reward(node_name, pod)
            self.node_controller.schedule_pod_to_node(pod, node_name)
            next_state = self._get_state(pod)

            logging.info(
                f"[DDQN-Scheduler-PyTorch-INFO]: Pod {pod.name} scheduled to "
                f"Node {node_name} with reward: {reward}"
            )
            self.schedule_history.append(
                {
                    "pod_name": pod.name,
                    "node_name": node_name,
                    "reward": reward,
                    "timestamp": datetime.datetime.now(),
                }
            )
            self.remember(state, action, reward, next_state, False)
            self.replay()
        except Exception as exc:
            logging.error(
                f"[DDQN-Scheduler-PyTorch-ERROR]: Failed to schedule Pod "
                f"{pod.name} to Node {node_name}: {exc}"
            )

        return node_name

    def _get_state(self, pod):
        states = []
        required_cpu = self.parse_cpu(pod.resources.get("requests", {}).get("cpu", 0))
        required_memory = self.parse_memory(
            pod.resources.get("requests", {}).get("memory", 0)
        )
        required_gpu = self.parse_gpu(pod.resources.get("requests", {}).get("gpu", 0))

        for node in self.node_controller.nodes.values():
            states.append(
                [
                    node.allocated_cpu,
                    node.allocated_memory,
                    node.allocated_gpu,
                    node.total_cpu - node.allocated_cpu,
                    node.total_memory - node.allocated_memory,
                    node.total_gpu - node.allocated_gpu,
                    required_cpu,
                    required_memory,
                    required_gpu,
                ]
            )
        return np.array(states, dtype=np.float32).reshape(1, -1)

    def _get_node_from_action(self, action):
        node_names = list(self.node_controller.nodes.keys())
        return node_names[action]

    def _calculate_reward(self, node_name, pod):
        node = self.node_controller.get_node(node_name)
        if node.status == "Ready":
            required_cpu = self.parse_cpu(pod.resources.get("requests", {}).get("cpu", 0))
            required_memory = self.parse_memory(
                pod.resources.get("requests", {}).get("memory", 0)
            )
            required_gpu = self.parse_gpu(pod.resources.get("requests", {}).get("gpu", 0))

            if (
                (node.total_cpu - node.allocated_cpu) < required_cpu
                or (node.total_memory - node.allocated_memory) < required_memory
                or (node.total_gpu - node.allocated_gpu) < required_gpu
            ):
                print(f"Node {node.name} insufficient resources:")
                print(
                    f"Remaining CPU: {node.total_cpu - node.allocated_cpu}, "
                    f"Required: {required_cpu}"
                )
                print(
                    f"Remaining Memory: {node.total_memory - node.allocated_memory}, "
                    f"Required: {required_memory}"
                )
                print(
                    f"Remaining GPU: {node.total_gpu - node.allocated_gpu}, "
                    f"Required: {required_gpu}"
                )
                return -1

            cpu_usage_ratio = (
                node.allocated_cpu / node.total_cpu if node.total_cpu > 0 else 0
            )
            memory_usage_ratio = (
                node.allocated_memory / node.total_memory if node.total_memory > 0 else 0
            )
            gpu_usage_ratio = (
                node.allocated_gpu / node.total_gpu if node.total_gpu > 0 else 0
            )

            reward = 1 - (
                cpu_usage_ratio + memory_usage_ratio + gpu_usage_ratio
            ) / 3

            cpu_utilizations = [
                n.allocated_cpu / n.total_cpu if n.total_cpu > 0 else 0
                for n in self.node_controller.nodes.values()
            ]
            memory_utilizations = [
                n.allocated_memory / n.total_memory if n.total_memory > 0 else 0
                for n in self.node_controller.nodes.values()
            ]
            gpu_utilizations = [
                n.allocated_gpu / n.total_gpu if n.total_gpu > 0 else 0
                for n in self.node_controller.nodes.values()
            ]

            cpu_load_balance_factor = 1 / (1 + np.std(cpu_utilizations))
            memory_load_balance_factor = 1 / (1 + np.std(memory_utilizations))
            gpu_load_balance_factor = 1 / (1 + np.std(gpu_utilizations))

            reward += (
                cpu_load_balance_factor
                + memory_load_balance_factor
                + gpu_load_balance_factor
            ) / 3 * 0.5

            return reward
        return -1

    def parse_cpu(self, cpu_str):
        if isinstance(cpu_str, str):
            match = re.match(r"(\d+)(m|)", cpu_str)
            if match:
                cpu_value, unit = match.groups()
                cpu_value = float(cpu_value)
                if unit == "m":
                    return cpu_value / 1000
                return cpu_value
        return 0.0

    def parse_memory(self, mem_str):
        if isinstance(mem_str, str):
            match = re.match(r"(\d+)(Ki|Mi|Gi|Ti)", mem_str)
            if match:
                mem_value, unit = match.groups()
                mem_value = float(mem_value)
                if unit == "Ki":
                    return int(mem_value * 1024)
                if unit == "Mi":
                    return int(mem_value * 1024 * 1024)
                if unit == "Gi":
                    return int(mem_value * 1024 * 1024 * 1024)
                if unit == "Ti":
                    return int(mem_value * 1024 * 1024 * 1024 * 1024)
        return 0

    def parse_gpu(self, gpu_str):
        if isinstance(gpu_str, str):
            match = re.match(r"(\d+)(Gpu|)", gpu_str, re.IGNORECASE)
            if match:
                gpu_value, _ = match.groups()
                return int(gpu_value)
        return 0

    def calculate_score(self, state):
        (
            allocated_cpu,
            allocated_memory,
            allocated_gpu,
            free_cpu,
            free_memory,
            free_gpu,
            required_cpu,
            required_memory,
            required_gpu,
        ) = state

        if (
            free_cpu < required_cpu
            or free_memory < required_memory
            or free_gpu < required_gpu
        ):
            return float("inf")

        total_cpu = max(allocated_cpu + free_cpu, 1)
        total_memory = max(allocated_memory + free_memory, 1)
        total_gpu = max(allocated_gpu + free_gpu, 1)

        utilization_score = (
            allocated_cpu / total_cpu
            + allocated_memory / total_memory
            + allocated_gpu / total_gpu
        )
        remaining_score = (
            (free_cpu - required_cpu) / total_cpu
            + (free_memory - required_memory) / total_memory
            + (free_gpu - required_gpu) / total_gpu
        )
        return utilization_score - remaining_score

    def select_best_node(self, states):
        num_nodes = states.shape[1] // 9
        reshaped_states = states.reshape(num_nodes, 9)

        best_node_index = -1
        best_score = float("inf")
        for index, state in enumerate(reshaped_states):
            score = self.calculate_score(state)
            if score < best_score:
                best_score = score
                best_node_index = index
        return best_node_index

    def get_schedule_history(self):
        return self.schedule_history

    def save_schedule_history(self, file_path="schedule_history.png"):
        if not self.schedule_history:
            print("No scheduling history available for visualization.")
            return

        timestamps = [record["timestamp"] for record in self.schedule_history]
        node_names = [record["node_name"] for record in self.schedule_history]
        rewards = [record["reward"] for record in self.schedule_history]
        time_numeric = [ts.timestamp() for ts in timestamps]

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.scatter(time_numeric, node_names, alpha=0.7, label="Scheduled Nodes")
        plt.yticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Node Names")
        plt.title("Pod Scheduling History")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time_numeric, rewards, "-o", label="Reward Trend")
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.title("Reward Over Time")
        plt.legend()

        plt.tight_layout()
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        plt.savefig(file_path)
        plt.close()
        print(f"Schedule history visualization saved to {file_path}.")
