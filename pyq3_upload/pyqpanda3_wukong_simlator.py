import numpy as np
import networkx as nx
import pandas as pd
from pyqpanda3.core import QCircuit, H, RX, CNOT, QProg, measure, CPUQVM, NoiseModel,RZ
from pyqpanda3.core import pauli_x_error, depolarizing_error, GateType
from pyqpanda3.qcloud import QCloudService, QCloudOptions, QCloudJob
import time
import csv
import cvxpy as cp
from datetime import datetime
import os
from typing import Dict, Tuple


def generate_qaoa_circuit(graph: nx.Graph, layers: int) -> QCircuit:
    """生成QAOA电路"""
    n_qubits = len(graph.nodes)
    circuit = QCircuit(n_qubits)

    # 初始态制备（均匀叠加态）
    for i in range(n_qubits):
        circuit << H(i)

    # 交替应用问题哈密顿量和驱动哈密顿量
    for layer in range(layers):
        # 问题哈密顿量部分
        for i, j in graph.edges:
            circuit << CNOT(i, j)
            circuit << RZ(j, 2 * 1.0)  # 假设gamma参数为1.0
            circuit << CNOT(i, j)

        # 驱动哈密顿量部分
        for i in range(n_qubits):
            circuit << RX(i, 2 * 0.5)  # 假设beta参数为0.5

    return circuit


def _calculate_cut_value(solution: str, graph: nx.Graph) -> float:
    """计算给定分割方案的割值"""
    cut_value = 0
    for i, j in graph.edges:
        if solution[i] != solution[j]:
            cut_value += 1
    return cut_value


def gw_algorithm(graph: nx.Graph) -> Tuple[float, str]:
    """
    使用半定规划实现Goemans-Williamson算法，求解最大割问题
    返回：(最大割值, 最优解)
    """
    n = len(graph.nodes)

    # 获取邻接矩阵并转换为密集数组(兼容NetworkX 3.0+)
    adj_matrix = nx.adjacency_matrix(graph).toarray()

    # 构建半定规划问题
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]  # X是半正定矩阵
    constraints += [X[i, i] == 1 for i in range(n)]  # 对角线元素为1

    # 使用cp.multiply进行元素级乘法，避免弃用警告
    objective = cp.Maximize(0.25 * cp.sum(cp.multiply(adj_matrix, (1 - X))))

    # 求解SDP问题
    prob = cp.Problem(objective, constraints)

    # 使用SCS求解器，它更适合处理大规模问题
    prob.solve(solver=cp.SCS, verbose=False)

    # 处理求解器可能没有找到最优解的情况
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print(f"警告: SDP求解器状态 - {prob.status}")

    # 获取并处理SDP解矩阵
    X_val = X.value

    # 使用特征值分解确保矩阵是正定的
    eigenvalues, eigenvectors = np.linalg.eigh(X_val)

    # 设置最小特征值阈值(处理数值误差)
    eps = 1e-8
    eigenvalues = np.maximum(eigenvalues, eps)

    # 重构矩阵以确保数值稳定性
    L = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

    # 随机超平面分割 - 多次尝试提高解的质量
    max_cut = 0
    best_bits = None
    num_trials = 10  # 尝试10次随机分割

    for _ in range(num_trials):
        # 生成随机单位向量
        r = np.random.randn(n)
        r /= np.linalg.norm(r)

        # 根据向量投影生成分割
        bits = np.array([1 if np.dot(L[i], r) >= 0 else 0 for i in range(n)])

        # 计算当前分割的割值
        current_cut = 0
        for u, v in graph.edges:
            if bits[u] != bits[v]:
                current_cut += 1

        # 更新最佳解
        if current_cut > max_cut:
            max_cut = current_cut
            best_bits = ''.join(map(str, bits))

    return max_cut, best_bits

def run_without_noise_model(graph: nx.Graph, layers: int = 1, shots: int = 1000) -> Dict:
    """在模拟器上运行无噪声模型的量子算法"""
    print(f"\n=== 在模拟器上运行无噪声模型: {len(graph.nodes)}-节点图, {layers}-层QAOA ===")
    start_time = time.time()

    # 创建QAOA电路
    circuit = generate_qaoa_circuit(graph, layers)

    # 添加测量门
    measure_circuit = QProg()
    measure_circuit << circuit
    for i in range(len(graph.nodes)):
        measure_circuit << measure(i, i)

    # 准备CPU模拟器
    machine = CPUQVM()
    # machine.init()  # 初始化量子虚拟机

    # 执行无噪声模型的程序
    machine.run(measure_circuit, shots)
    counts = machine.result().get_counts()

    # 分析结果
    best_solution = max(counts.items(), key=lambda x: x[1])
    solution_cut = _calculate_cut_value(best_solution[0], graph)
    max_cut, _ = gw_algorithm(graph)
    approximation_ratio = solution_cut / max_cut if max_cut > 0 else 0

    # 整理实验结果（统一格式）
    experiment_results = {
        'graph_nodes': len(graph.nodes),
        'graph_edges': len(graph.edges),
        'qaoa_layers': layers,
        'shots': shots,
        'best_solution': best_solution[0],
        'solution_cut': solution_cut,
        'solution_probability': best_solution[1] / shots,
        'max_cut': max_cut,
        'approximation_ratio': approximation_ratio,
        'execution_time': time.time() - start_time,
        'platform': 'simulator',
        'device': "CPUQVM",
        'task_id': "Not Applicable",  # 本地模拟没有任务ID
        'noise_model': "Null",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"无噪声模拟实验完成，耗时: {experiment_results['execution_time']:.2f}秒")
    print(f"最大割值: {max_cut}, 近似比: {approximation_ratio:.4f}")
    return experiment_results


def run_with_noise_model(graph: nx.Graph, layers: int = 1, shots: int = 1000,
                         noise_model: dict = None) -> Dict:
    """在模拟器上运行含噪声模型的量子算法"""
    print(f"\n=== 在模拟器上运行含噪声模型: {len(graph.nodes)}-节点图, {layers}-层QAOA ===")
    start_time = time.time()

    # 创建QAOA电路
    circuit = generate_qaoa_circuit(graph, layers)

    # 添加测量门
    measure_circuit = QProg()
    measure_circuit << circuit
    for i in range(len(graph.nodes)):
        measure_circuit << measure(i, i)

    # 准备噪声模型
    machine = CPUQVM()

    if noise_model and isinstance(noise_model, dict):
        print(f"使用噪声模型: {noise_model}")
        model = NoiseModel()

        # 设置噪声参数
        depolarizing_rate = noise_model.get('depolarizing_rate', 0.01)
        bitflip_rate = noise_model.get('bitflip_rate', 0.01)

        # 单比特门噪声
        single_qubit_error = depolarizing_error(depolarizing_rate)
        model.add_all_qubit_quantum_error(single_qubit_error, GateType.H)
        model.add_all_qubit_quantum_error(single_qubit_error, GateType.RX)
        model.add_all_qubit_quantum_error(single_qubit_error, GateType.RY)
        model.add_all_qubit_quantum_error(single_qubit_error, GateType.RZ)

        # 双比特门噪声
        two_qubit_error = depolarizing_error(depolarizing_rate * 2)
        model.add_all_qubit_quantum_error(two_qubit_error, GateType.CNOT)

        # 比特翻转噪声
        bitflip_error = pauli_x_error(bitflip_rate)
        model.add_all_qubit_quantum_error(bitflip_error, GateType.H)
        model.add_all_qubit_quantum_error(bitflip_error, GateType.RX)

        # 运行带噪声模型的程序
        machine.run(measure_circuit, shots, model)
    else:
        # 运行无噪声模型的程序
        machine.run(measure_circuit, shots)

    counts = machine.result().get_counts()

    # 分析结果
    best_solution = max(counts.items(), key=lambda x: x[1])
    solution_cut = _calculate_cut_value(best_solution[0], graph)
    max_cut, _ = gw_algorithm(graph)
    approximation_ratio = solution_cut / max_cut if max_cut > 0 else 0

    # 整理实验结果（统一格式）
    experiment_results = {
        'graph_nodes': len(graph.nodes),
        'graph_edges': len(graph.edges),
        'qaoa_layers': layers,
        'shots': shots,
        'best_solution': best_solution[0],
        'solution_cut': solution_cut,
        'solution_probability': best_solution[1] / shots,
        'max_cut': max_cut,
        'approximation_ratio': approximation_ratio,
        'execution_time': time.time() - start_time,
        'platform': 'simulator',
        'device': "CPUQVM",
        'task_id': "Not Applicable",  # 本地模拟没有任务ID
        'noise_model': str(noise_model) if noise_model else "Null",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"含噪声模拟实验完成，耗时: {experiment_results['execution_time']:.2f}秒")
    print(f"最大割值: {max_cut}, 近似比: {approximation_ratio:.4f}")
    return experiment_results

# 为了在CSV中区分不同图类型，修改保存函数以包含图类型信息
def save_results_to_csv(results: Dict, filename: str = 'qaoa_results.csv'):
    file_exists = os.path.isfile(filename)

    # 添加graph_type列
    column_headers = [
        'graph_type', 'graph_nodes', 'graph_edges', 'qaoa_layers', 'shots',
        'best_solution', 'solution_cut', 'solution_probability',
        'max_cut', 'approximation_ratio', 'execution_time',
        'platform', 'device', 'task_id', 'noise_model', 'timestamp'
    ]
    #
    # column_descriptions = {
    #     'graph_type': 'Type of graph (regular, erdos_renyi, complete, etc.)',
    #     'graph_nodes': 'Number of nodes in the graph',
    #     'graph_edges': 'Number of edges in the graph',
    #     'qaoa_layers': 'Number of QAOA layers',
    #     'shots': 'Number of quantum measurements',
    #     'best_solution': 'Best solution found (bit string representation)',
    #     'solution_cut': 'Cut value of the best solution',
    #     'solution_probability': 'Measurement probability of the best solution',
    #     'max_cut': 'Maximum cut value computed by Goemans-Williamson algorithm',
    #     'approximation_ratio': 'Approximation ratio (solution_cut/max_cut)',
    #     'execution_time': 'Execution time of the experiment (seconds)',
    #     'platform': 'Execution platform (real_device or simulator)',
    #     'device': 'Name of the device used',
    #     'task_id': 'Quantum computation task ID (for real devices) or N/A (for simulators)',
    #     'noise_model': 'Noise model parameters used',
    #     'timestamp': 'Experiment timestamp'
    # }

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_headers)

        if not file_exists:
            writer.writeheader()
            # writer.writerow({col: column_descriptions[col] for col in column_headers})

        writer.writerow(results)
def main():
    # 定义实验配置 - 生成3-10个节点的全连接图
    configs = []
    for nodes in range(3, 11):  # 3到10个节点
        configs.append({
            'graph_type': 'complete',
            'nodes': nodes,
            'layers': 1,
            'shots': 1000,
            'noise_model': {
                'depolarizing_rate': 0.02,
                'bitflip_rate': 0.01
            }
        })

    # 对每种配置运行实验
    for config in configs:
        print(f"\n=== 运行实验: {config['graph_type']}图, {config['nodes']}个节点 ===")

        # 创建图
        if config['graph_type'] == 'complete':
            graph = nx.complete_graph(config['nodes'])
        else:
            print(f"错误: 不支持的图类型 {config['graph_type']}")
            continue

        print(f"创建了一个{config['graph_type']}图，包含{len(graph.nodes)}个节点和{len(graph.edges)}条边")

        # 运行无噪声模型
        if config.get('run_without_noise', True):
            results = run_without_noise_model(
                graph,
                layers=config['layers'],
                shots=config['shots']
            )
            # 添加图类型信息
            results['graph_type'] = config['graph_type']
            save_results_to_csv(results)

        # 运行含噪声模型
        if config.get('run_with_noise', True):
            results = run_with_noise_model(
                graph,
                layers=config['layers'],
                shots=config['shots'],
                noise_model=config['noise_model']
            )
            # 添加图类型信息
            results['graph_type'] = config['graph_type']
            save_results_to_csv(results)

    print(f"所有实验完成，结果已保存到 qaoa_results.csv")


if __name__ == "__main__":
    main()