from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, Rzz, RX, DepolarizingChannel
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import mindspore.nn as nn
import mindspore as ms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import time

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")


def build_ansatz(qubit, p, g, problem, noise_level=0.0):
    """构建量子线路"""
    circ = Circuit()
    for i in range(p):
        # Problem Hamiltonian
        for index, z in enumerate(g.edges):
            circ += Rzz(f'g{i}').on(z)
            if problem == "noise_model" and noise_level > 0:
                circ += DepolarizingChannel(noise_level, 2).on(z)

        # Mixer Hamiltonian
        for z in range(qubit):
            circ += RX(f'b{i}').on(z)
            if problem == "noise_model" and noise_level > 0:
                circ += DepolarizingChannel(noise_level).on(z)

    return circ


def build_ham(g):
    """构建哈密顿量"""
    ham = QubitOperator()
    for index, i in enumerate(g.edges):
        ham += QubitOperator(f'Z{i[0]} Z{i[1]}')
    return ham


def train_improved(g, qubit, p, ham, problem, noise_level=0.0, max_epochs=300, learning_rate=0.05):
    """训练量子线路"""
    init_state_circ = UN(H, qubit)
    ansatz = build_ansatz(qubit, p, g, problem, noise_level)
    circ = init_state_circ + ansatz

    sim = Simulator('mqvector', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    net = MQAnsatzOnlyLayer(grad_ops)

    opti = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    train_net = nn.TrainOneStepCell(net, opti)

    history = []
    best_loss = float('inf')
    best_params = None
    patience = 50

    for i in range(max_epochs):
        loss = train_net()
        history.append(loss.asnumpy())

        if loss.asnumpy() < best_loss:
            best_loss = loss.asnumpy()
            best_params = net.weight.asnumpy()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {i + 1}")
            break

        if (i + 1) % 50 == 0:
            print(f"Epoch {i + 1}/{max_epochs}, Loss: {loss.asnumpy()}")

    if best_params is not None:
        net.weight.set_data(ms.Tensor(best_params))

    pr = dict(zip(ansatz.params_name, net.weight.asnumpy()))
    return circ, pr, train_net()


def calculate_new_ar(noise_energy, ideal_energy):
    """计算新的近似比"""
    if isinstance(noise_energy, ms.Tensor):
        noise_energy = noise_energy.asnumpy().item()
    if isinstance(ideal_energy, ms.Tensor):
        ideal_energy = ideal_energy.asnumpy().item()

    if noise_energy > 0:
        noise_energy = -noise_energy
    if ideal_energy > 0:
        ideal_energy = -ideal_energy

    ratio = abs(noise_energy / ideal_energy)

    if ratio > 1.0:
        print(f"Warning: AR > 1.0: {ratio}. Clamping to 1.0")
        ratio = 1.0

    return ratio


def run_experiment(p, noise_level, qubits, instances, max_epochs):
    """运行完整实验并返回结果"""
    results = {
        'ideal': np.zeros((len(qubits), instances)),
        'noise': np.zeros((len(qubits), instances)),
        'new_ar': np.zeros((len(qubits), instances))
    }

    for q_idx, qubit in enumerate(tqdm(qubits, desc=f"Processing qubits for p={p}")):
        for inst in range(instances):
            maxcut_graph = nx.random_regular_graph(n=qubit, d=qubit - 1)
            ham_operator = build_ham(maxcut_graph)
            ham = Hamiltonian(ham_operator)

            _, pr_ideal, energy_ideal = train_improved(
                maxcut_graph, qubit, p, ham, "ideal_model", 0.0, max_epochs
            )

            _, pr_noise, energy_noise = train_improved(
                maxcut_graph, qubit, p, ham, "noise_model", noise_level, max_epochs
            )

            energy_ideal_val = energy_ideal.asnumpy().item()
            energy_noise_val = energy_noise.asnumpy().item()

            new_ar = calculate_new_ar(energy_noise_val, energy_ideal_val)

            results['ideal'][q_idx, inst] = energy_ideal_val
            results['noise'][q_idx, inst] = energy_noise_val
            results['new_ar'][q_idx, inst] = new_ar

    return results


def main():
    # 固定参数
    noise_level = 0.1
    qubits = [2, 4, 6, 8, 10]
    instances = 20
    max_epochs = 300

    # 不同的线路层次
    p_values = [1, 2, 3, 4, 5]

    print("\nCurrent parameter settings:")
    print(f"Noise coefficient: {noise_level}")
    print(f"Number of qubits: {qubits}")
    print(f"Instances per qubit size: {instances}")
    print(f"Maximum training epochs: {max_epochs}")
    print(f"Circuit layers (p): {p_values}")

    # 运行不同p值的实验
    results_by_p = {}
    for p in p_values:
        print(f"\nRunning experiment for p={p}")
        results = run_experiment(p, noise_level, qubits, instances, max_epochs)
        results_by_p[p] = results

    # 提取每个p值下所有量子比特数的平均AR
    all_instances_data = np.zeros((len(p_values), instances))
    for p_idx, p in enumerate(p_values):
        # 计算所有量子比特数的平均AR
        avg_ar_per_instance = np.mean(results_by_p[p]['new_ar'], axis=0)
        all_instances_data[p_idx] = avg_ar_per_instance

    # 计算均值
    mean_values = np.mean(all_instances_data, axis=1)

    # Visualize results
    plt.figure(figsize=(12, 6), dpi=150)

    # 图1: 特定p值下的AR vs 量子比特数
    plt.subplot(1, 2, 1)
    selected_p = 3  # 选择一个p值进行显示
    x = np.arange(len(qubits))
    width = 0.35

    new_ar_mean = np.mean(results_by_p[selected_p]['new_ar'], axis=1)
    new_ar_std = np.std(results_by_p[selected_p]['new_ar'], axis=1)

    plt.bar(x, new_ar_mean, width, label=f'AR (p={selected_p})', color='purple')
    plt.errorbar(x, new_ar_mean, yerr=new_ar_std, fmt='none', ecolor='black', capsize=4)

    plt.xticks(x, qubits)
    plt.xlabel('Number of Qubits')
    plt.ylabel('Average AR (Noise/Ideal)')
    plt.title(f'Average AR vs Qubits (p={selected_p}, Noise={noise_level})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 图2: AR趋势 vs 线路深度
    plt.subplot(1, 2, 2)

    # 绘制所有实例的结果 (半透明)
    for p_idx in range(len(p_values)):
        plt.scatter([p_values[p_idx]] * instances, all_instances_data[p_idx],
                    color='lightgreen', alpha=0.3, s=10)

    # 绘制均值曲线 (加粗)
    plt.plot(p_values, mean_values, 'g-', linewidth=3, label='Average AR')

    plt.xlabel('Circuit Layer Depth (p)')
    plt.ylabel('AR (Noise/Ideal)')
    plt.title(f'AR Trend vs Circuit Depth (Noise={noise_level})')
    plt.xticks(p_values)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'ar_trend_vs_p_noise{noise_level}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results_by_p


if __name__ == "__main__":
    results_by_p = main()