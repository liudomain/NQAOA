from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, CNOT, Rzz, RX, DepolarizingChannel
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import networkx as nx
import mindspore.nn as nn
import mindspore as ms
import matplotlib.pyplot as plt
import logging
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='optimizer_comparison.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Use the new device setting API
ms.set_device("CPU")  # Set to "GPU" if available
ms.set_context(mode=ms.PYNATIVE_MODE)  # Keep other context settings unchanged
# #绘制k-正则图
# n = int(input("请输入图节点数："))
# k = n - 1  # 当k=n-1时，生成完全图（需确保k为偶数，否则random_regular_graph会报错）
#
# # 处理k为奇数的情况（确保k为偶数，否则无法生成正则图）
# if k % 2 != 0:
#     print("警告：k=n-1为奇数，自动调整k为n-2（需保证n≥2）")
#     k = max(n-2, 0)  # 确保k≥0
#
# V = list(range(n))  # 顶点集合
#
# # 生成k-正则图（若k=0，则为n个孤立节点）
# graph = None  # 显式初始化graph为None
# try:
#     graph = nx.random_graphs.random_regular_graph(k, n)
#     print("图的边集合：", graph.edges)
# except nx.NetworkXException as e:
#     print(f"生成正则图失败：{e}，请检查n和k的取值")
#     exit()  # 确保失败时退出程序
#
# # 绘制图形
# pos = nx.circular_layout(graph)
# options = {
#     "with_labels": True,
#     "font_size": 20,
#     "font_weight": "bold",
#     "font_color": "white",
#     "node_size": 2000,
#     "width": 2
# }
#
# plt.figure(figsize=(8, 6))
# nx.draw_networkx(graph, pos, **options)
# ax = plt.gca()
# ax.margins(0.20)
# plt.axis("off")
# plt.title(f"{n}nodes{k}-regular graphs", fontsize=20)
# plt.show()
#
#
# print("————————————————————————————去极化噪声信道————————————————————————————")
# # 从键盘输入噪声系数（0-1之间的小数）
# while True:
#     try:
#         p1 = float(input("请输入噪声系数（0-1之间的小数）："))
#         if 0 <= p1 <= 1:
#             break
#         else:
#             print("输入值超出范围，请重新输入！")
#     except ValueError:
#         print("输入格式错误，请输入有效的小数！")
# #去极化信道
# def build_hc_dep(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
#     hc=Circuit()
#     for i in g.edges:
#         hc+=Rzz(para).on(i)
#         hc+=DepolarizingChannel(p1,2).on(i)
#     hc.barrier()
#     return hc
# #搭建U_B(beta)对应的量子线路
# def build_hb_dep(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
#     hb=Circuit()
#     for i in g.nodes:
#         hb+=RX(para).on(i)
#         hb+=DepolarizingChannel(p1).on(i)
#     hb.barrier()
#     return hb
# #为了使得最后优化的结果足够准确，我们需要将量子线路重复多次，因此我们通过如下函数搭建多层的训练网络：
# def build_ansatz_dep(g,p):#g是max-cut问题的图，p是ansatz线路的层数
#     circ=Circuit()
#     for i in range(p):
#         circ+=build_hc_dep(g,f'g{i}')
#         circ+=build_hb_dep(g,f'b{i}')
#     return circ
# ## 生成并显示量子线路图（确保graph不为None）
# if graph is not None:
#     circ_dep = build_ansatz_dep(graph, 1)
#
#     # 尝试使用matplotlib显示量子线路
#     try:
#         # 设置中文字体支持
#         plt.rcParams["font.family"] = ["SimHei"]
#
#         # 创建新的图形
#         plt.figure(figsize=(12, 8))
#
#         # 使用MindQuantum的matplotlib可视化
#         circ_dep.svg().mpl()
#
#     except Exception as e:
#         print(f"无法使用matplotlib显示量子线路: {e}")
#
#         # 回退方案：打印线路信息
#         print("\n量子线路结构信息：")
#         print(circ_dep)
# else:
#     print("无法生成量子线路：图对象为空。")
# #构建图对应的哈密顿量Hc
# def build_ham(g):
#     ham=QubitOperator()
#     for i in g.edges:
#         ham+=QubitOperator(f'Z{i[0]} Z{i[1]}')
#     return ham
# def get_expectation_of_hamitonian(circ, qubit, ham, pr):
#
#     sim = Simulator('mqvector', qubit)
#     sim.apply_circuit(circ, pr)
#     result = sim.get_expectation(ham)
#     # energy[k] = result.real
#
#     return result.real
#
# def max_cut(ham_operator):
#     # 首先根据哈密顿量的算符将其转化为矩阵形式。
#     ham_matrix = ham_operator.matrix()
#     ham_matrix = ham_matrix.todense()
#     # calculate the eigenvalue and eigenvector
#     eigval, eigvec = np.linalg.eig(ham_matrix)
#     # ground energy
#     min_cut_val = min(eigval).real
#
#     return min_cut_val
#
# p = int(input("请输入线路层次p值："))
# ham=Hamiltonian(build_ham(graph))
# init_state_circ=UN(H,graph.nodes)
# ansatz=build_ansatz_dep(graph,p)
# circ=init_state_circ+ansatz
# circ.svg(width=1200)
# # #表示选用层次为p的QAOA量子线路，ansatz是求解该问题的量子线路
# # # init_state_circ是将量子态制备到均匀叠加态（HB的基态）上的量子线路
# #
# #
# # #1)利用传统优化算法完成优化搜索
# # sim=Simulator('mqvector',circ.n_qubits)
# # grad_opts=sim.get_expectation_with_grad(ham,circ)#利用模拟器生成计算QAOA变分量子线路期望值和梯度的运算算子。
# # #通过如下方式计算线路在参数为10个随机生成 p0 时的期望值和导数。
# # import numpy as np
# # rng=np.random.default_rng(10)
# # p0=rng.random(size=len(circ.params_name))*np.pi*2-np.pi#初始参数
# # f,g=grad_opts(p0)
# # print('Expectation Value: ', f)
# # #期望值是一个(1,1)维的数组，其中m表示本次运算将多少数据通过编码器编码成了量子态，由于QAOA任务不用编码器，因此m
# # #取默认值1，n表示本次运算计算了多少个哈密顿量期望值（MindQuantum支持多哈密顿量并行处理），此处我们只计算了ham的期望值
# # print('Expectation Value Shape: ', f.shape)
# # print('Gradient: ', g)
# # #对于梯度值来说，它的维度是(1, 1, 8)，新增的维度k=8表示整个线路中的ansatz变分参数个数。g0-g3和b0-b3
# # print('Gradient Shape: ', g.shape)
# #
# # #2)引入scipy中的二阶优化器BFGS来对Max-Cut问题进行优化，为此首先定义待优化函数
# # global step
# # step=0
# # def fun(p,grad_opts):
# #     global step
# #     f,g=grad_opts(p)
# #     f=np.real(f)[0,0]
# #     g=np.real(g)[0,0]
# #     step+=1
# #     if step%10==0:
# #         print(f"train step:{step},cut:[{(len(graph.edges)-f)/2}]")
# #     return f,g
# # fun(p0,grad_opts)
# # #采用BFGS的二阶优化方法，指定jac=True，表示告诉优化器，待优化的函数在返回函数值的同时也会返回梯度值。
# # # 如设定为False，优化器会利用差分法自行计算近似梯度，这会消耗大量算力。
# # from scipy.optimize import minimize
# # step = 0
# # res = minimize(fun, p0, args=(grad_opts, ), method='bfgs', jac=True)
# # print(dict(zip(circ.params_name, res.x)))#在最优解时，输出训练得到的变分参数
#
#
# #3)利用 MindSpore 机器学习框架完成量子神经网络训练
# #搭建待训练量子神经网络
# #使用MQAnsatzOnlyLayer作为待训练的量子神经网络，并采用Adam优化器。
# import mindspore as ms
# ms.set_context(mode=ms.PYNATIVE_MODE,device_target='CPU')
# sim=Simulator('mqvector',circ.n_qubits) # 创建模拟器，backend使用‘mqvector’，能模拟5个比特（'circ'线路中包含的比特数）
# grad_ops=sim.get_expectation_with_grad(ham,circ) # 获取sim计算变分量子线路的期望值和梯度的算子
# net=MQAnsatzOnlyLayer(grad_ops)# # 生成待训练的神经网络
# ## Adam同时使用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。 一阶矩来控制模型更新的方向，二阶矩控制步长(学习率)。
# max_cuts=[]
# opti1=nn.Adam(net.trainable_params(),learning_rate=0.05) # 设置针对网络中所有可训练参数、学习率为0.05的Adam优化器
# train_net1=nn.TrainOneStepCell(net,opti1)# 对神经网络采用设定优化器opti进行一步训练
# #训练，展示max_cut问题求解，#该问题哈密顿量的基态能量对应的边切割数趋近于6。
# for i in range(300):
#     cut=(len(graph.edges)-train_net1())/2
#     if i%10==0:
#         print("train step:",i,",cut:",cut)
#         cut=cut.numpy()
#         max_cuts.append(cut)



# ---------------------- 图生成部分 ----------------------
n = int(input("Enter the number of graph nodes: "))
k = n - 1
if k % 2 != 0:
    logging.info(f"Warning: k=n-1 is odd, automatically adjusting k to n-2 (ensure n≥2)")
    k = max(n - 2, 0)
V = list(range(n))
try:
    graph = nx.random_graphs.random_regular_graph(k, n)
    logging.info(f"Graph edges: {graph.edges}")
except nx.NetworkXException as e:
    logging.error(f"Failed to generate regular graph: {e}, please check n and k values")
    exit()

# ---------------------- 噪声信道与量子线路构建 ----------------------
p1 = float(input("Enter noise coefficient (decimal between 0-1): "))

def build_hc_dep(g, para, noise_level):
    hc = Circuit()
    for i in g.edges:
        hc += Rzz(para).on(i)
        if noise_level > 0:
            hc += DepolarizingChannel(noise_level, 2).on(i)
    hc.barrier()
    return hc

def build_hb_dep(g, para, noise_level):
    hb = Circuit()
    for i in g.nodes:
        hb += RX(para).on(i)
        if noise_level > 0:
            hb += DepolarizingChannel(noise_level).on(i)
    hb.barrier()
    return hb

def build_ham(g):
    ham = QubitOperator()
    for i in g.edges:
        ham += QubitOperator(f'Z{i[0]} Z{i[1]}')
    return ham

def build_ansatz_dep(g, p, noise_level):
    circ = Circuit()
    for i in range(p):
        circ += build_hc_dep(g, f'g{i}', noise_level)
        circ += build_hb_dep(g, f'b{i}', noise_level)
    return circ

p = int(input("Enter the circuit layer parameter p: "))
init_state_circ = UN(H, graph.nodes)

# 构建理想信道和含噪信道的量子线路
ansatz_ideal = build_ansatz_dep(graph, p, 0)
ansatz_noisy = build_ansatz_dep(graph, p, p1)

circ_ideal = init_state_circ + ansatz_ideal
circ_noisy = init_state_circ + ansatz_noisy

# ---------------------- 哈密顿量与模拟器设置 ----------------------
ham = Hamiltonian(build_ham(graph))

# 理想信道模拟器
sim_ideal = Simulator('mqvector', circ_ideal.n_qubits)
grad_ops_ideal = sim_ideal.get_expectation_with_grad(ham, circ_ideal)

# 含噪信道模拟器
sim_noisy = Simulator('mqvector', circ_noisy.n_qubits)
grad_ops_noisy = sim_noisy.get_expectation_with_grad(ham, circ_noisy)

# ---------------------- 多优化器配置 ----------------------
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

# 理想信道网络
net_ideal = MQAnsatzOnlyLayer(grad_ops_ideal)

# 含噪信道网络
net_noisy = MQAnsatzOnlyLayer(grad_ops_noisy)

optimizers = {
    'SGD': nn.SGD,
    'Momentum': nn.Momentum,
    'Adam': nn.Adam,
    'Adagrad': nn.Adagrad,
    'Adadelta': nn.Adadelta,
}

optimizer_args = {
    'SGD': {'learning_rate': 0.01},
    'Momentum': {'learning_rate': 0.01, 'momentum': 0.9},
    'Adam': {'learning_rate': 0.01},
    'Adagrad': {'learning_rate': 0.01},
    'Adadelta': {'learning_rate': 0.01, 'rho': 0.9, 'epsilon': 1e-6},
}

# 为每个优化器创建训练网络
train_nets_ideal = {
    name: nn.TrainOneStepCell(net_ideal, optimizer(net_ideal.trainable_params(), **optimizer_args[name]))
    for name, optimizer in optimizers.items()
}

train_nets_noisy = {
    name: nn.TrainOneStepCell(net_noisy, optimizer(net_noisy.trainable_params(), **optimizer_args[name]))
    for name, optimizer in optimizers.items()
}

# 存储每个优化器的训练结果
results_ideal = {name: [] for name in optimizers.keys()}
results_noisy = {name: [] for name in optimizers.keys()}

# 训练循环
total_steps = 500
logging.info(f"Start training with {total_steps} steps")

for i in range(total_steps):
    # 理想信道训练
    for name, train_net in train_nets_ideal.items():
        cut = (len(graph.edges) - train_net()) / 2
        cut_value = cut.asnumpy().item()
        if i % 10 == 0:
            results_ideal[name].append(cut_value)
            logging.info(f"Ideal Channel - Optimizer: {name}, Step: {i}, Cut: {cut_value:.4f}")

    # 含噪信道训练
    for name, train_net in train_nets_noisy.items():
        cut = (len(graph.edges) - train_net()) / 2
        cut_value = cut.asnumpy().item()
        if i % 10 == 0:
            results_noisy[name].append(cut_value)
            logging.info(f"Noisy Channel - Optimizer: {name}, Step: {i}, Cut: {cut_value:.4f}")

# 可视化
plt.figure(figsize=(16, 6))

# 左子图：理想信道
plt.subplot(1, 2, 1)
for name, history in results_ideal.items():
    plt.plot(range(0, total_steps, 10), history, linewidth=2, label=name)
plt.xlabel('Training Steps')
plt.ylabel('Max-Cut Value')
plt.title('Optimizer Performance (Ideal Channel, Noise=0)')
plt.legend()
plt.grid(True)

# 右子图：含噪信道
plt.subplot(1, 2, 2)
for name, history in results_noisy.items():
    plt.plot(range(0, total_steps, 10), history, linewidth=2, label=name)
plt.xlabel('Training Steps')
plt.ylabel('Max-Cut Value')
plt.title(f'Optimizer Performance (Noisy Channel, Noise={p1})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('optimizer_comparison.pdf', dpi=600, bbox_inches='tight')
plt.show()

logging.info("Training completed. Results saved to optimizer_comparison.png and optimizer_comparison.log")