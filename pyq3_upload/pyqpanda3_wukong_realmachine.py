from fileinput import filename
from pyqpanda3.qcloud import JobStatus  # 导入任务状态枚举
import numpy as np
import networkx as nx
import pandas as pd
from pyqpanda3.core import QCircuit, H, RX, CNOT, QProg, measure, CPUQVM, NoiseModel,RZ
from pyqpanda3.core import pauli_x_error, depolarizing_error, GateType
from pyqpanda3.qcloud import QCloudService, QCloudOptions, QCloudJob,JobStatus,LogOutput
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
def run_on_real_device(graph: nx.Graph, layers: int = 1, shots: int = 1000,
                       api_key: str = "39a6291eba3XXXX") -> Dict:
    """在真机上运行量子算法"""
    print(f"\n=== 在真机上运行: {len(graph.nodes)}-节点图, {layers}-层QAOA ===")
    start_time = time.time()

    # 创建QAOA电路
    circuit = generate_qaoa_circuit(graph, layers)

    # 添加测量门
    measure_circuit = QProg()
    measure_circuit << circuit
    for i in range(len(graph.nodes)):
        measure_circuit << measure(i, i)

    # 连接本源量子云服务
    print("开始连接本源量子云服务，准备在真机上运行...")

    # 创建QCloudService实例
    service = QCloudService(api_key=api_key)
    service.setup_logging()
    # 设置云服务选项
    options = QCloudOptions()
    # options.device_name = "origin_wukong"  # 指定真机设备
    # options.shots = shots
    # options.task_type = "QPU"  # 量子计算任务类型
    backend=service.backend("origin_wukong")
    # 提交任务
    print("正在提交量子计算任务...")
    # job = service.submit_job(measure_circuit, options)
    job=backend.run(measure_circuit,shots,options)
    # print(f"任务已提交，任务ID: {job.get_id()}")
    # 记录任务提交时间
    submit_time = time.time()
    # 等待任务完成（简化状态监控）
    print("等待任务完成...")
    while True:
        status = job.status()
        if status in [JobStatus.FINISHED, JobStatus.CANCELLED, JobStatus.FAILED]:
            break
        time.sleep(5)  # 避免频繁查询，间隔5秒

    # 计算任务执行时间（从提交到完成的时间）
    execution_time = time.time() - submit_time
    total_time = time.time() - start_time

    # 获取结果
    try:
        result = job.result()
        counts = result.get_counts()

        # 打印概率分布（如果有）
        print("\n测量结果概率分布:")
        for prob in result.get_probs_list():
            print(f"  {prob}")

    except Exception as e:
        print(f"获取结果时出错: {e}")
        counts = {}

    # 分析结果
    best_solution = max(counts.items(), key=lambda x: x[1]) if counts else ("", 0)
    solution_cut = _calculate_cut_value(best_solution[0], graph) if best_solution[0] else 0
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
        'solution_probability': best_solution[1] / shots if shots > 0 else 0,
        'max_cut': max_cut,
        'approximation_ratio': approximation_ratio,
        'execution_time': execution_time,  # 任务执行时间（提交到完成）
        'total_time': total_time,  # 总耗时（从函数开始到获取结果）
        'platform': 'real_device',
        'device': backend.name(),
        'task_id': job.get_id(),
        'noise_model': "Null",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # 'status_history': None# 记录状态变化时间
    }

    print(f"\n真机实验完成:")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  任务执行时间: {execution_time:.2f}秒")
    print(f"  最大割值: {max_cut}, 近似比: {approximation_ratio:.4f}")

    return experiment_results

def save_results_to_csv(results: Dict, filename: str = 'qaoa_results2.csv'):
    """将实验结果保存到CSV文件，并在第一行添加表头，第二行添加列说明"""
    file_exists = os.path.isfile(filename)

    # 定义表头和对应的说明
    column_headers = [
        'graph_nodes', 'graph_edges', 'qaoa_layers', 'shots',
        'best_solution', 'solution_cut', 'solution_probability',
        'max_cut', 'approximation_ratio', 'execution_time',
        'platform', 'device', 'task_id', 'noise_model', 'timestamp'
    ]



    with open(filename, 'a', newline='') as csvfile:
        # 创建字典写入器
        writer = csv.DictWriter(csvfile, fieldnames=column_headers)

        # 如果文件不存在，写入表头和说明
        if not file_exists:
            # 写入表头
            writer.writeheader()
            # 写入列说明（作为第一行数据）
            # writer.writerow({col: column_descriptions[col] for col in column_headers})

        # 写入实际数据行
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
            'api_key': "39a6291eba3XXXX",  # 本源量子云API密钥
            'run_on_real_device': True  # 是否在真机上运行
        })

    # 存储实验结果的列表
    all_results = []

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

        # 运行真机实验
        if config.get('run_on_real_device', True):
            print("准备在真机上运行实验...")
            try:
                results = run_on_real_device(
                    graph,
                    layers=config['layers'],
                    shots=config['shots'],
                    api_key=config['api_key']
                )
                # 添加图类型信息
                results['graph_type'] = config['graph_type']
                all_results.append(results)
                save_results_to_csv(results)
                print(f"真机实验完成: {config['nodes']}节点图")
            except Exception as e:
                print(f"真机实验失败: {e}")
                print(f"跳过{config['nodes']}节点图的真机实验")
                continue

    print(f"所有真机实验完成，结果已保存到 qaoa_results2.csv")

if __name__ == "__main__":
    main()