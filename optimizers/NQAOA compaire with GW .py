from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, CNOT, Rzz, RX, DepolarizingChannel
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import networkx as nx
import mindspore.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from scipy.linalg import sqrtm
from scipy.optimize import minimize
import time
import logging

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")


# -------------------------
# 日志配置
# -------------------------
def setup_logger(log_file):
    """配置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有处理器
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 创建日志文件，使用时间戳确保唯一性
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = f"max_cut_qaoa_{timestamp}.log"
logger = setup_logger(log_file)

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]


# -------------------------
# GW算法实现
# -------------------------
def gw_algorithm(graph, num_iterations=100):
    """
    使用Goemans-Williamson算法求解图的最大割问题

    参数:
    graph: networkx图对象，待求解的图
    num_iterations: int，重复采样次数，用于寻找更好的割

    返回:
    max_cut_value: float，最大割的值
    best_cut: list，最佳割的节点划分（0或1）
    """
    logger.info(f"开始运行GW算法，采样次数: {num_iterations}")

    # 构建图的邻接矩阵
    n = graph.number_of_nodes()
    adj_matrix = nx.adjacency_matrix(graph).todense()
    logger.debug(f"图的邻接矩阵形状: {adj_matrix.shape}")

    # 构建半定规划问题的目标矩阵
    laplacian = np.diag(np.sum(adj_matrix, axis=1)) - adj_matrix

    # 解决半定规划问题
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    # 确保特征值非负
    eigenvalues = np.maximum(eigenvalues, 0)

    # 构建解矩阵 X = V * sqrt(Λ)
    sqrt_eigenvalues = np.diag(np.sqrt(eigenvalues))
    X = eigenvectors @ sqrt_eigenvalues

    # 重复多次随机超平面分割，寻找最佳割
    max_cut_value = 0
    best_cut = None

    for i in range(num_iterations):
        # 生成随机单位向量
        r = np.random.randn(n)
        r = r / np.linalg.norm(r)

        # 根据随机向量进行节点分割
        cut = [1 if X[i].dot(r) > 0 else 0 for i in range(n)]

        # 计算当前割的值
        cut_value = 0
        for i, j in graph.edges():
            if cut[i] != cut[j]:
                cut_value += 1

        # 更新最佳割
        if cut_value > max_cut_value:
            max_cut_value = cut_value
            best_cut = cut

        if i % 10 == 0:
            logger.debug(f"GW算法迭代 {i}: 当前最大割 = {max_cut_value}")

    logger.info(f"GW算法完成，最大割值: {max_cut_value}")
    return max_cut_value, best_cut


# -------------------------
# 量子算法部分
# -------------------------
# 绘制k-正则图
n = int(input("请输入图节点数："))
k = n - 1  # 当k=n-1时，生成完全图（需确保k为偶数，否则random_regular_graph会报错）
logger.info(f"用户输入：节点数 n = {n}, k = {k}")

# 处理k为奇数的情况
if k % 2 != 0:
    logger.warning(f"k={k}为奇数，自动调整k为n-2")
    k = max(n - 2, 0)  # 确保k≥0

V = list(range(n))  # 顶点集合

# 生成k-正则图
graph = None
try:
    graph = nx.random_regular_graph(k, n)
    logger.info(f"成功生成{k}-正则图，节点数: {n}, 边数: {len(graph.edges)}")
    logger.info(f"图的边集合: {graph.edges}")
except nx.NetworkXException as e:
    logger.error(f"生成正则图失败: {e}，请检查n和k的取值")
    exit()

# 绘制图形
pos = nx.circular_layout(graph)
options = {
    "with_labels": True,
    "font_size": 15,
    "font_weight": "bold",
    "font_color": "white",
    "node_size": 1500,
    "width": 2
}

plt.figure(figsize=(8, 6))
nx.draw_networkx(graph, pos, **options)
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.title(f"{n}节点{k}-正则图", fontsize=20)
plt.tight_layout()
plt.savefig(f"graph_{n}nodes_{k}regular.png", dpi=300)
plt.close()
logger.info(f"已保存图的可视化结果至 graph_{n}nodes_{k}regular.png")

print("————————————————————————————去极化噪声信道————————————————————————————")
# 输入噪声系数
while True:
    try:
        p1 = float(input("请输入噪声系数（0-1之间的小数）："))
        if 0 <= p1 <= 1:
            logger.info(f"用户输入噪声系数: p1 = {p1}")
            break
        else:
            logger.warning("输入值超出范围，请重新输入！")
    except ValueError:
        logger.error("输入格式错误，请输入有效的小数！")


# 去极化信道
def build_hc_dep(g, para):
    hc = Circuit()
    for i in g.edges:
        hc += Rzz(para).on(i)
        hc += DepolarizingChannel(p1, 2).on(i)
    hc.barrier()
    return hc


# 搭建U_B(beta)对应的量子线路
def build_hb_dep(g, para):
    hb = Circuit()
    for i in g.nodes:
        hb += RX(para).on(i)
        hb += DepolarizingChannel(p1).on(i)
    hb.barrier()
    return hb


# 搭建多层的训练网络
def build_ansatz_dep(g, p):
    circ = Circuit()
    for i in range(p):
        circ += build_hc_dep(g, f'g{i}')
        circ += build_hb_dep(g, f'b{i}')
    return circ


# 构建图对应的哈密顿量Hc
def build_ham(g):
    ham = QubitOperator()
    for i in g.edges:
        ham += QubitOperator(f'Z{i[0]} Z{i[1]}')
    return ham


def get_expectation_of_hamitonian(circ, qubit, ham, pr):
    sim = Simulator('mqvector', qubit)
    sim.apply_circuit(circ, pr)
    result = sim.get_expectation(ham)
    return result.real


def max_cut(ham_operator):
    # 计算哈密顿量的基态能量
    ham_matrix = ham_operator.matrix()
    ham_matrix = ham_matrix.todense()
    eigval, _ = np.linalg.eig(ham_matrix)
    min_cut_val = min(eigval).real
    return min_cut_val


# -------------------------
# 运行GW算法求解Max-Cut
# -------------------------
print("\n————————————————————————————GW算法求解————————————————————————————")
logger.info("开始运行GW算法...")
start_time = time.time()
gw_max_cut, _ = gw_algorithm(graph, num_iterations=100)
gw_time = time.time() - start_time
print(f"GW算法最大割值: {gw_max_cut}")
print(f"GW算法运行时间: {gw_time:.4f}秒")
logger.info(f"GW算法最大割值: {gw_max_cut}, 运行时间: {gw_time:.4f}秒")

# -------------------------
# 运行量子算法求解Max-Cut
# -------------------------
print("\n————————————————————————————量子算法求解————————————————————————————")
p = int(input("请输入线路层次p值："))
logger.info(f"用户输入线路层次: p = {p}")

ham = Hamiltonian(build_ham(graph))
init_state_circ = UN(H, graph.nodes)
ansatz = build_ansatz_dep(graph, p)
circ = init_state_circ + ansatz

# 显示量子线路图
try:
    plt.figure(figsize=(12, 8))
    circ.svg().mpl()
    plt.title("量子线路图", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"quantum_circuit_p{p}.png", dpi=300)
    plt.close()
    logger.info(f"已保存量子线路图至 quantum_circuit_p{p}.png")
except Exception as e:
    logger.error(f"无法使用matplotlib显示量子线路: {e}")
    print("\n量子线路结构信息：")
    print(circ)

# 量子算法训练
sim = Simulator('mqvector', circ.n_qubits)
grad_ops = sim.get_expectation_with_grad(ham, circ)
net = MQAnsatzOnlyLayer(grad_ops)

# 定义不同的优化器
optimizers = {
    "Adam": nn.Adam(net.trainable_params(), learning_rate=0.05),
    "SGD": nn.SGD(net.trainable_params(), learning_rate=0.05),
    "Momentum": nn.Momentum(net.trainable_params(), learning_rate=0.05, momentum=0.9),
    "Adagrad": nn.Adagrad(net.trainable_params(), learning_rate=0.05),
    "Adadelta": nn.Adadelta(net.trainable_params(), learning_rate=0.05)
}

# 存储不同优化器的结果
results = {}

# 训练并评估不同优化器
for opt_name, opt in optimizers.items():
    logger.info(f"\n开始使用{opt_name}优化器训练...")
    print(f"\n——————————————使用{opt_name}优化器训练——————————————")
    train_net = nn.TrainOneStepCell(net, opt)
    max_cuts = []
    ar_history = []  # 存储每一步的近似比

    start_time = time.time()
    for i in range(500):
        cut = (len(graph.edges) - train_net()) / 2
        cut_value = cut.asnumpy().item()
        current_ar = cut_value / gw_max_cut if gw_max_cut > 0 else 0

        max_cuts.append(cut_value)
        ar_history.append(current_ar)

        if i % 20 == 0:
            logger.info(f"{opt_name}优化器 - 训练步骤 {i}: 割值 = {cut_value:.4f}, 近似比 = {current_ar:.4f}")
            print(f"训练步骤: {i}, 割值: {cut_value:.4f}")

    train_time = time.time() - start_time
    logger.info(f"{opt_name}优化器训练完成，耗时: {train_time:.4f}秒")

    # 保存训练过程数据到文件
    history_file = f"{opt_name}_training_history_p{p}.txt"
    with open(history_file, 'w') as f:
        f.write(f"步骤,割值,近似比\n")
        for i in range(len(max_cuts)):
            f.write(f"{i},{max_cuts[i]:.6f},{ar_history[i]:.6f}\n")
    logger.info(f"{opt_name}优化器训练历史已保存至 {history_file}")

    pr = dict(zip(ansatz.params_name, net.weight.asnumpy()))
    mean_max_cuts = np.mean(max_cuts)
    max_max_cuts = np.max(max_cuts)
    best_step = np.argmax(max_cuts)

    # 计算近似比
    gw_ratio = max_max_cuts / gw_max_cut if gw_max_cut > 0 else 0

    results[opt_name] = {
        "max_cut": max_max_cuts,
        "mean_cut": mean_max_cuts,
        "gw_ratio": gw_ratio,
        "train_time": train_time,
        "best_step": best_step
    }

    logger.info(f"{opt_name}优化器 - 最大割: {max_max_cuts:.4f} (步骤 {best_step}), 平均割: {mean_max_cuts:.4f}")
    logger.info(f"{opt_name}优化器 - 相对于GW算法的近似比: {gw_ratio:.4f}")
    logger.info(f"{opt_name}优化器 - 训练时间: {train_time:.4f}秒")

    print(f"{opt_name}优化器 - 最大割: {max_max_cuts:.4f}, 平均割: {mean_max_cuts:.4f}")
    print(f"{opt_name}优化器 - 相对于GW算法的近似比: {gw_ratio:.4f}")
    print(f"{opt_name}优化器 - 训练时间: {train_time:.4f}秒")

# # -------------------------
# # 结果可视化
# # -------------------------
# plt.figure(figsize=(16, 12))
#
# # 1. 不同优化器的最终近似比比较
# plt.subplot(2, 2, 1)
# opt_names = list(results.keys())
# gw_ratios = [results[opt]["gw_ratio"] for opt in opt_names]
#
# bar_width = 0.5
# index = np.arange(len(opt_names))
#
# plt.bar(index, gw_ratios, bar_width, color='skyblue')
# plt.axhline(y=1.0, color='r', linestyle='--', label='GW基准')  # 添加GW算法基准线
#
# plt.xlabel('优化器', fontsize=14)
# plt.ylabel('近似比 (AR)', fontsize=14)
# plt.title('不同优化器的最终近似比比较', fontsize=16)
# plt.xticks(index, opt_names, rotation=45, fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
#
# # 2. 不同优化器训练过程中的近似比变化趋势
# plt.subplot(2, 2, 2)
# # 使用之前保存的训练数据
# for opt_name in opt_names:
#     history_file = f"{opt_name}_training_history_p{p}.txt"
#     data = np.loadtxt(history_file, delimiter=',', skiprows=1)
#     steps = data[:, 0]
#     ar_values = data[:, 2]
#
#     # 绘制近似比随训练步骤的变化
#     plt.plot(steps, ar_values, label=opt_name, linewidth=2)
#
# plt.axhline(y=1.0, color='r', linestyle='--', label='GW基准')
# plt.xlabel('训练步骤', fontsize=14)
# plt.ylabel('近似比 (AR)', fontsize=14)
# plt.title('训练过程中近似比的变化趋势', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(linestyle='--', alpha=0.7)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.tight_layout()
#
# # 3. 不同噪声水平下的近似比比较（模拟数据）
# plt.subplot(2, 2, 3)
# noise_levels = np.linspace(0, 0.5, 6)  # 不同噪声水平
# avg_ars = []
#
# print("\n模拟不同噪声水平下的性能...")
# logger.info("开始模拟不同噪声水平下的性能...")
# noise_results = {}
#
# for noise in noise_levels:
#     logger.info(f"\n模拟噪声水平: {noise:.2f}")
#     # 临时修改全局噪声参数
#     p1 = noise
#
#     # 重建带噪声的电路
#     ansatz_noise = build_ansatz_dep(graph, p)
#     circ_noise = init_state_circ + ansatz_noise
#
#     # 使用Adam优化器训练并获取结果
#     sim_noise = Simulator('mqvector', circ_noise.n_qubits)
#     grad_ops_noise = sim_noise.get_expectation_with_grad(ham, circ_noise)
#     net_noise = MQAnsatzOnlyLayer(grad_ops_noise)
#
#     opti_adam = nn.Adam(net_noise.trainable_params(), learning_rate=0.05)
#     train_net_noise = nn.TrainOneStepCell(net_noise, opti_adam)
#
#     ar_values = []
#     for i in range(100):  # 减少训练步数以加快模拟
#         cut = (len(graph.edges) - train_net_noise()) / 2
#         cut_value = cut.asnumpy().item()
#         current_ar = cut_value / gw_max_cut
#         ar_values.append(current_ar)
#
#         if i % 20 == 0:
#             logger.info(f"噪声 {noise:.2f} - 步骤 {i}: 割值 = {cut_value:.4f}, 近似比 = {current_ar:.4f}")
#
#     avg_ar = np.mean(ar_values[-20:])  # 取最后20步的平均值
#     avg_ars.append(avg_ar)
#     noise_results[noise] = avg_ar
#
#     print(f"噪声水平 {noise:.2f}: 平均近似比 = {avg_ar:.4f}")
#     logger.info(f"噪声水平 {noise:.2f}: 平均近似比 = {avg_ar:.4f}")
#
# # 保存噪声模拟结果
# noise_file = f"noise_simulation_p{p}.txt"
# with open(noise_file, 'w') as f:
#     f.write(f"噪声水平,平均近似比\n")
#     for noise, ar in noise_results.items():
#         f.write(f"{noise:.6f},{ar:.6f}\n")
# logger.info(f"噪声模拟结果已保存至 {noise_file}")
#
# # 恢复原始噪声参数
# p1 = float(input("请输入噪声系数（0-1之间的小数）："))
# logger.info(f"恢复原始噪声系数: p1 = {p1}")
#
# # 绘制噪声水平对近似比的影响
# plt.plot(noise_levels, avg_ars, 'o-', color='green', linewidth=2)
# plt.xlabel('噪声系数 (p1)', fontsize=14)
# plt.ylabel('平均近似比 (AR)', fontsize=14)
# plt.title('噪声水平对近似比的影响', fontsize=16)
# plt.grid(linestyle='--', alpha=0.7)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.tight_layout()
#
# # 4. 不同图规模下的近似比预测（理论分析）
# plt.subplot(2, 2, 4)
# # 理论分析：不同节点数下的近似比预测
# node_counts = [4, 6, 8, 10, 12, 14, 16]
# theoretical_ars = [0.95, 0.92, 0.89, 0.86, 0.83, 0.80, 0.78]  # 示例数据，实际应通过实验获得
#
# plt.plot(node_counts, theoretical_ars, 's-', color='purple', linewidth=2)
# plt.xlabel('图节点数', fontsize=14)
# plt.ylabel('预测近似比 (AR)', fontsize=14)
# plt.title('不同图规模下的近似比预测', fontsize=16)
# plt.grid(linestyle='--', alpha=0.7)
# plt.xticks(node_counts, fontsize=12)
# plt.yticks(fontsize=12)
# plt.tight_layout()
#
# # 保存所有图表
# plt.savefig(f"results_comparison_p{p}.png", dpi=300)
# plt.close()
# logger.info(f"结果比较图表已保存至 results_comparison_p{p}.png")

# -------------------------
# 结果可视化（修改部分）
# -------------------------
plt.figure(figsize=(16, 16))  # 增加高度以容纳更多图表

# 1. 不同优化器的平均近似比比较（柱状图）
plt.subplot(2, 2, 1)
opt_names = list(results.keys())
mean_ars = [results[opt]["mean_cut"] / gw_max_cut for opt in opt_names]

# 使用科学配色方案
colors = plt.cm.viridis(np.linspace(0, 0.8, len(opt_names)))

# 绘制平均近似比柱状图
bars = plt.bar(opt_names, mean_ars, color=colors, edgecolor='black')
plt.axhline(y=1.0, color='r', linestyle='--', label='GW Baseline')  # 添加GW算法基准线

# 在柱子上方显示具体数值（保留4位小数）
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.xlabel('Optimizer', fontsize=14)
plt.ylabel('Average Approximation Ratio (AR)', fontsize=14)
plt.title('Comparison of Average Approximation Ratios', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0.5, 1.0)  # 设置纵坐标范围为0.5-1
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 2. 不同优化器的最大近似比比较（柱状图）
plt.subplot(2, 2, 2)
max_ars = [results[opt]["gw_ratio"] for opt in opt_names]

# 使用相同的科学配色方案
colors = plt.cm.viridis(np.linspace(0, 0.8, len(opt_names)))

# 绘制最大近似比柱状图
bars = plt.bar(opt_names, max_ars, color=colors, edgecolor='black')
plt.axhline(y=1.0, color='r', linestyle='--', label='GW Baseline')  # 添加GW算法基准线

# 在柱子上方显示具体数值（保留4位小数）
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.xlabel('Optimizer', fontsize=14)
plt.ylabel('Maximum Approximation Ratio (AR)', fontsize=14)
plt.title('Comparison of Maximum Approximation Ratios', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0.5, 1.0)  # 设置纵坐标范围为0.5-1
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 3. 不同优化器训练过程中的近似比变化趋势（训练步数0-500，间隔20步）
plt.subplot(2, 2, 3)
# 重新训练以获取每一步的近似比数据（增加到500步，间隔20步记录）
for opt_name, opt in optimizers.items():
    print(f"\nCollecting training data for {opt_name} optimizer (500 steps, recorded every 20 steps)...")
    train_net = nn.TrainOneStepCell(net, opt)
    ar_history = []
    steps = []

    for i in range(501):  # 0-500步
        cut = (len(graph.edges) - train_net()) / 2
        cut_value = cut.asnumpy().item()
        current_ar = cut_value / gw_max_cut

        # 每20步记录一次数据
        if i % 20 == 0:
            ar_history.append(current_ar)
            steps.append(i)
            print(f"Step {i}: AR = {current_ar:.4f}")

    # 绘制近似比随训练步骤的变化，使用不同颜色和标记
    markers = ['o', 's', '^', 'D', 'v']
    marker_idx = opt_names.index(opt_name)
    plt.plot(steps, ar_history, marker=markers[marker_idx], linestyle='-',
             label=opt_name, linewidth=2, markersize=6)

plt.axhline(y=1.0, color='r', linestyle='--', label='GW Baseline')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Approximation Ratio (AR)', fontsize=14)
plt.title('Approximation Ratio Trend During Training', fontsize=16)
plt.legend(fontsize=12)
plt.grid(linestyle='--', alpha=0.7)
plt.xticks(steps[::2], fontsize=10, rotation=45)  # 每两个刻度显示一个，避免拥挤
plt.yticks(fontsize=12)
plt.tight_layout()

# 4. 不同噪声水平下的近似比比较（噪声系数范围改为0-1）
plt.subplot(2, 2, 4)
noise_levels = np.linspace(0, 1.0, 11)  # 噪声系数范围改为0-1，11个点
avg_ars = []

print("\nSimulating performance under different noise levels...")
for noise in noise_levels:
    # 临时修改全局噪声参数
    p1 = noise

    # 重建带噪声的电路
    ansatz_noise = build_ansatz_dep(graph, p)
    circ_noise = init_state_circ + ansatz_noise

    # 使用Adam优化器训练并获取结果
    sim_noise = Simulator('mqvector', circ_noise.n_qubits)
    grad_ops_noise = sim_noise.get_expectation_with_grad(ham, circ_noise)
    net_noise = MQAnsatzOnlyLayer(grad_ops_noise)

    opti_adam = nn.Adam(net_noise.trainable_params(), learning_rate=0.05)
    train_net_noise = nn.TrainOneStepCell(net_noise, opti_adam)

    ar_values = []
    for i in range(100):  # 减少训练步数以加快模拟
        cut = (len(graph.edges) - train_net_noise()) / 2
        cut_value = cut.asnumpy().item()
        ar_values.append(cut_value / gw_max_cut)

    avg_ar = np.mean(ar_values[-20:])  # 取最后20步的平均值
    avg_ars.append(avg_ar)

    print(f"Noise level {noise:.2f}: Average AR = {avg_ar:.4f}")

# 恢复原始噪声参数
p1 = float(input("Please enter noise coefficient (decimal between 0-1): "))

# 绘制噪声水平对近似比的影响
plt.plot(noise_levels, avg_ars, 'o-', color='green', linewidth=2)
plt.axhline(y=0.878, color='blue', linestyle='-.', label='Theoretical Limit')  # 添加理论极限线
plt.xlabel('Noise Coefficient (p1)', fontsize=14)
plt.ylabel('Average Approximation Ratio (AR)', fontsize=14)
plt.title('Effect of Noise Level on Approximation Ratio', fontsize=16)
plt.grid(linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()


# 显示所有图形
plt.savefig(f"results_comparison_p{p}_2.pdf", dpi=600)
plt.close()
logger.info(f"结果比较图表已保存至 results_comparison_p{p}_2.pdf")

# 打印最终结果比较表（修改为显示近似比）
print("\n————————————————————最终结果比较表————————————————————")
print(f"{'优化器':<10} {'最大近似比':<12} {'平均近似比':<12} {'训练时间(秒)':<15}")
print("-" * 55)
for opt_name, result in results.items():
    print(
        f"{opt_name:<10} {result['gw_ratio']:<12.4f} {result['mean_cut'] / gw_max_cut:<12.4f} {result['train_time']:<15.4f}")
print(f"{'GW算法':<10} 1.0000 (基准)")

# 记录最终结果到日志
logger.info("\n————————————————————最终结果比较表————————————————————")
logger.info(f"{'优化器':<10} {'最大近似比':<12} {'平均近似比':<12} {'训练时间(秒)':<15}")
logger.info("-" * 55)
for opt_name, result in results.items():
    logger.info(
        f"{opt_name:<10} {result['gw_ratio']:<12.4f} {result['mean_cut'] / gw_max_cut:<12.4f} {result['train_time']:<15.4f}")
logger.info(f"{'GW算法':<10} 1.0000 (基准)")

logger.info(f"\n完整日志已保存至: {log_file}")
print(f"\n完整日志已保存至: {log_file}")