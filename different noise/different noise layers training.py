from IPython.core.display_functions import display
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H,CNOT,Rzz, RX,DepolarizingChannel,BitFlipChannel,PhaseFlipChannel,\
AmplitudeDampingChannel,PauliChannel,BitPhaseFlipChannel,PhaseDampingChannel
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import networkx as nx
import mindspore.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Image
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# 创建数据存储目录
if not os.path.exists('data'):
    os.makedirs('data')

n = int(input("请输入图节点n的值："))
k=n-1
# 从键盘输入p1和p2的值（范围为0-1，且p1+p2<=1）
while True:
    try:
        p1 = float(input("请输入p1的值（0-1之间的小数）："))
        if not (0 <= p1 <= 1):
            print("p1超出范围，请重新输入！")
            continue

        p2 = float(input("请输入p2的值（0-1之间的小数，且满足p1+p2<=1）："))
        if not (0 <= p2 <= 1):
            print("p2超出范围，请重新输入！")
            continue

        if p1 + p2 > 1:
            print("p1 + p2 > 1，不满足条件，请重新输入！")
            continue

        p3 = 1 - p1 - p2
        print(f"参数设置成功：p1={p1}, p2={p2}, p3={p3}")
        break

    except ValueError:
        print("输入格式错误，请输入有效的小数！")

V = list(range(n))  # 顶点集合
# 生成k-正则图（若k=0，则为n个孤立节点）
graph = None  # 显式初始化graph为None
try:
    graph = nx.random_graphs.random_regular_graph(k, n)
    print("图的边集合：", graph.edges)
except nx.NetworkXException as e:
    print(f"生成正则图失败：{e}，请检查n和k的取值")
    exit()  # 确保失败时退出程序

# 绘制图形
pos = nx.circular_layout(graph)
options = {
    "with_labels": True,
    "font_size": 20,
    "font_weight": "bold",
    "font_color": "white",
    "node_size": 2000,
    "width": 2
}

plt.figure(figsize=(8, 6))
nx.draw_networkx(graph, pos, **options)
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.title(f"{n}nodes{k}-regular graphs", fontsize=20)
plt.show()

print("——————————————————构造哈密顿量，求解最大割函数——————————————————")
#构建图对应的哈密顿量Hc
def build_ham(g):
    ham=QubitOperator()
    for i in g.edges:
        ham+=QubitOperator(f'Z{i[0]} Z{i[1]}')
    return ham
def get_expectation_of_hamitonian(circ, qubit, ham, pr):
    sim = Simulator('mqvector', qubit)
    sim.apply_circuit(circ, pr)
    result = sim.get_expectation(ham)
    # energy[k] = result.real
    return result.real

def max_cut(ham_operator):
    # 首先根据哈密顿量的算符将其转化为矩阵形式。
    ham_matrix = ham_operator.matrix()
    ham_matrix = ham_matrix.todense()
    # calculate the eigenvalue and eigenvector
    eigval, eigvec = np.linalg.eig(ham_matrix)
    # ground energy
    min_cut_val = min(eigval).real
    return min_cut_val

#为了使得最后优化的结果足够准确，我们需要将量子线路重复多次，因此我们通过如下函数搭建多层的训练网络：
print("——————————————————搭建ansatz线路层次——————————————————")
def build_ansatz(qubit,layer,graph,problem):#g是max-cut问题的图，p是ansatz线路的层数
    circ=Circuit()
    if problem=='ideal_channel':
        for i in range(layer):
            for j in graph.edges:
                circ+=Rzz(f'g{j}').on(j)
            for j in range(qubit):
                circ+=RX(f'b{j}').on(j)
    if problem=='depolarizing_channel':
        for i in range(layer):
            for j in graph.edges:
                circ+=Rzz(f'g{j}').on(j)
                circ+=DepolarizingChannel(p1,2).on(j)
            for j in range(qubit):
                circ+=RX(f'b{j}').on(j)
                circ+=DepolarizingChannel(p1).on(j)
    if problem=='amplitudedamping_channel':
        for i in range(layer):
            for j in graph.edges:
                circ+=Rzz(f'g{j}').on(j)
                circ+=CNOT.on(j[1],j[0])
                circ+=AmplitudeDampingChannel(p1).on(j[1])
                circ+=CNOT.on(j[1],j[0])
            for j in range(qubit):
                circ+=RX(f'b{j}').on(j)
                circ+=AmplitudeDampingChannel(p1).on(j)
    if problem=='bitflip_channel':
        for i in range(layer):
            for j in graph.edges:
                circ+=Rzz(f'g{j}').on(j)
                circ+=CNOT.on(j[1],j[0])
                circ+=BitFlipChannel(p1).on(j[1])
                circ+=CNOT.on(j[1],j[0])

            for j in range(qubit):
                circ+=RX(f'b{j}').on(j)
                circ+=BitFlipChannel(p1).on(j)
    if problem=='bitphaseflip_channel':
        for i in range(layer):
            for j in graph.edges:
                circ+=Rzz(f'g{j}').on(j)
                circ+=CNOT.on(j[1],j[0])
                circ+=BitPhaseFlipChannel(p1).on(j[1])
                circ+=CNOT.on(j[1],j[0])
            for j in range(qubit):
                circ+=RX(f'b{j}').on(j)
                circ+=BitPhaseFlipChannel(p1).on(j)
    if problem=='pauli_channel':
        for i in range(layer):
            for j in graph.edges:
                circ+=Rzz(f'g{j}').on(j)
                circ+=CNOT.on(j[1],j[0])
                circ+=PauliChannel(p1,p2,1-p1-p2).on(j[1])
                circ+=CNOT.on(j[1],j[0])
            for j in range(qubit):
                circ+=RX(f'b{j}').on(j)
                circ+=PauliChannel(p1,p2,1-p1-p2).on(j)
    if problem=='phasedamping_channel':
        for i in range(layer):
            for j in graph.edges:
                circ+=Rzz(f'g{j}').on(j)
                circ+=CNOT.on(j[1],j[0])
                circ+=PhaseDampingChannel(p1).on(j[1])
                circ+=CNOT.on(j[1],j[0])
            for j in range(qubit):
                circ+=RX(f'b{j}').on(j)
                circ+=PhaseDampingChannel(p1).on(j)

    if problem=='phaseflip_channel':
        for i in range(layer):
            for j in graph.edges:
                circ+=Rzz(f'g{j}').on(j)
                circ+=CNOT.on(j[1],j[0])
                circ+=PhaseFlipChannel(p1).on(j[1])
                circ+=CNOT.on(j[1],j[0])
            for j in range(qubit):
                circ+=RX(f'b{j}').on(j)
                circ+=PhaseFlipChannel(p1).on(j)
    return circ


#训练不同类型的信道电路
def train(g, qubit, p, ham, problem):
    """
    g: max cut 的图
    qubit: 输入量子比特的数目
    p: 线路的层数
    ham: 问题哈密顿量
    problem:  "ideal_model" 或者 "noise_model"

    """
    # bulid the quantum circuit
    init_state_circ = UN(H, qubit)# 生成均匀叠加态，即对所有量子比特作用H门
    if problem == "ideal_channel":
        ansatz = build_ansatz(qubit, p, g, problem)   # 生成 QAOA-ansat理想线路
    if problem == "amplitudedamping_channel":
        ansatz= build_ansatz(qubit, p, g, problem) # 生成 QAOA-ansat-amplitudedamping线路
    if problem == "depolarizing_channel":
        ansatz = build_ansatz(qubit, p, g, problem) # 生成 QAOA-ansat-depolarizing线路
    if problem == "bitflip_channel":
        ansatz= build_ansatz(qubit, p, g,problem)   # 生成 QAOA-ansatz-bitflip线路
    if problem == "bitphaseflip_channel":
        ansatz= build_ansatz(qubit, p, g, problem) # 生成 QAOA-ansat-bitphaseflip线路
    if problem == "pauli_channel":
        ansatz= build_ansatz(qubit, p, g, problem) # 生成 QAOA-ansat-pauli线路
    if problem == "phasedamping_channel":
        ansatz = build_ansatz(qubit, p, g, problem) # 生成 QAOA-ansat-phasedamping线路
    if problem == "phaseflip_channel":
        ansatz= build_ansatz(qubit, p, g, problem) # 生成 QAOA-ansat-phasedamping线路
   #理想环境训练
    circ = init_state_circ + ansatz                               # 将初始化线路与ansatz线路组合成一个线路
        # optimize the parameters
    sim = Simulator('mqvector', circ.n_qubits)

    grad_ops= sim.get_expectation_with_grad(ham, circ)            # 获取计算变分量子线路的期望值和梯度的算子
    net= MQAnsatzOnlyLayer(grad_ops)
    opti = nn.Adam(net.trainable_params(), learning_rate=0.1)
    train_net = nn.TrainOneStepCell(net, opti)
    for i in range(300):
        train_net()                                            # 将神经网络训练一步并计算得到的结果（切割边数）。注意：每当'train_net()'运行一次，神经网络就训练了一步
    pr= dict(zip(ansatz.params_name, net.weight.asnumpy()))   # 获取线路参数
    return circ, pr





#最大割求解（全连接图）
qubits = [3,4,5,6,7,8,9,10]#共8例
instance =5#实例次数
Approxi_ratio_i= [0]*instance*len(qubits)
Approxi_ratio_amp= [0]*instance*len(qubits)
Approxi_ratio_dep= [0]*instance*len(qubits)
Approxi_ratio_bf= [0]*instance*len(qubits)
Approxi_ratio_bpf= [0]*instance*len(qubits)
Approxi_ratio_p= [0]*instance*len(qubits)
Approxi_ratio_pd= [0]*instance*len(qubits)
Approxi_ratio_pf= [0]*instance*len(qubits)

k=0
p = int(input("请输入线路层次p值："))#线路层次从1开始
for qubit in tqdm(qubits):
    # 取5中不同的 3-Regular MaxCut 图取均值
    for i in range(instance):
        maxcut_graph=nx.random_regular_graph(n=qubit, d = qubit-1) #产生边为d的全连接图
        ham_operator=build_ham(maxcut_graph)
        max_cut_val=max_cut(ham_operator)
        ham=Hamiltonian(ham_operator)
        #---------------------------------------------------
        # 通过 理想QAOA 计算基态能量
        circ, pr = train(maxcut_graph, qubit, p, ham, "ideal_channel")
        expectation = get_expectation_of_hamitonian(circ, qubit, ham, pr)
        # 计算 QAOA 得出能量和准确能量的比值
        Approxi_ratio_i[k] = expectation/max_cut_val

        #---------------------------------------------------
        # 通过 振幅阻尼amp-QAOA 计算基态能量
        circ, pr = train(maxcut_graph, qubit, p,ham, "amplitudedamping_channel")
        expectation_2 = get_expectation_of_hamitonian(circ, qubit, ham, pr)        # 计算 QAOA 得出能量和准确能量的比值
        Approxi_ratio_amp[k] = expectation_2/max_cut_val
        #---------------------------------------------------
        # 通过 去极化depolarizing-QAOA 计算基态能量
        circ, pr = train(maxcut_graph, qubit, p,ham, "depolarizing_channel")
        expectation_3 = get_expectation_of_hamitonian(circ, qubit, ham, pr)        # 计算 QAOA 得出能量和准确能量的比值
        Approxi_ratio_dep[k] = expectation_3/max_cut_val
        #---------------------------------------------------
        # 通过 bitflip-QAOA 计算基态能量
        circ, pr = train(maxcut_graph, qubit, p,ham, "bitflip_channel")
        expectation_4 = get_expectation_of_hamitonian(circ, qubit, ham, pr)        # 计算 QAOA 得出能量和准确能量的比值
        Approxi_ratio_bf[k] = expectation_4/max_cut_val
        #---------------------------------------------------
        # 通过 bitphaseflip-QAOA 计算基态能量
        circ, pr = train(maxcut_graph, qubit, p,ham, "bitphaseflip_channel")
        expectation_5 = get_expectation_of_hamitonian(circ, qubit, ham, pr)        # 计算 QAOA 得出能量和准确能量的比值
        Approxi_ratio_bpf[k] = expectation_5/max_cut_val
        #---------------------------------------------------
        # 通过 振幅阻尼amp-QAOA 计算基态能量
        circ, pr = train(maxcut_graph, qubit, p,ham, "pauli_channel")
        expectation_6 = get_expectation_of_hamitonian(circ, qubit, ham, pr)        # 计算 QAOA 得出能量和准确能量的比值
        Approxi_ratio_p[k] = expectation_6/max_cut_val
        #---------------------------------------------------
        # 通过 振幅阻尼amp-QAOA 计算基态能量
        circ, pr = train(maxcut_graph, qubit, p,ham, "phasedamping_channel")
        expectation_7 = get_expectation_of_hamitonian(circ, qubit, ham, pr)        # 计算 QAOA 得出能量和准确能量的比值
        Approxi_ratio_pd[k] = expectation_7/max_cut_val
        #---------------------------------------------------
        # 通过 振幅阻尼amp-QAOA 计算基态能量
        circ, pr = train(maxcut_graph, qubit, p,ham, "phaseflip_channel")
        expectation_8 = get_expectation_of_hamitonian(circ, qubit, ham, pr)        # 计算 QAOA 得出能量和准确能量的比值
        Approxi_ratio_bf[k] = expectation_8/max_cut_val
        k += 1
        # print(ham_operator)
# 存储中间结果到CSV文件
results_df = pd.DataFrame({
    'qubits': np.repeat(qubits, instance),
    'instance': list(range(instance))*len(qubits),
    'iQAOA': Approxi_ratio_i,
    'ampQAOA': Approxi_ratio_amp,
    'depQAOA': Approxi_ratio_dep,
    'bfQAOA': Approxi_ratio_bf,
    'bpfQAOA': Approxi_ratio_bpf,
    'pQAOA': Approxi_ratio_p,
    'pdQAOA': Approxi_ratio_pd,
    'pfQAOA': Approxi_ratio_pf
})

# 保存到CSV文件
csv_path = "data/approximate_ratio_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"中间结果已保存到: {csv_path}")
# 确保保存图像的目录存在
save_dir = "data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"创建目录: {save_dir}")

# 量子边数、qubits对AR的影响
plt.figure(1, dpi=300)
# plt.plot(range(5*i,5*i+5),Approxi_ratio_i[5*i:5*(i+1)] , color='royalblue', label="iQAOA")
plt.plot(range(0, 40), Approxi_ratio_i[0:40], color='royalblue', label="iQAOA")
# plt.scatter(range(8*i,8*i+8), Approxi_ratio_i[8*i:8*(i+1)] , color='royalblue')
for i in range(8):
    plt.plot(range(5 * i, 5 * i + 5), Approxi_ratio_amp[5 * i:5 * (i + 1)], color='tomato', label="ampQAOA")
    # plt.plot(range(5*i,5*i+1), Approxi_ratio_amp[5*i:5*i+1], color='darkorange')
    plt.scatter(range(5 * i, 5 * i + 5), Approxi_ratio_amp[5 * i:5 * (i + 1)], color='tomato', s=20)
    plt.plot(range(5 * i, 5 * i + 5), Approxi_ratio_dep[5 * i:5 * (i + 1)], color='aqua', label="depQAOA")
    plt.scatter(range(5 * i, 5 * i + 5), Approxi_ratio_dep[5 * i:5 * (i + 1)], color='aqua', s=20)
    plt.plot(range(5 * i, 5 * i + 5), Approxi_ratio_bf[5 * i:5 * (i + 1)], color='powderblue', label="bfQAOA")
    plt.scatter(range(5 * i, 5 * i + 5), Approxi_ratio_bf[5 * i:5 * (i + 1)], color='powderblue', s=20)
    plt.plot(range(5 * i, 5 * i + 5), Approxi_ratio_bpf[5 * i:5 * (i + 1)], color='brown', label="bpfQAOA")
    plt.scatter(range(5 * i, 5 * i + 5), Approxi_ratio_bpf[5 * i:5 * (i + 1)], color='brown', s=20)
    plt.plot(range(5 * i, 5 * i + 5), Approxi_ratio_p[5 * i:5 * (i + 1)], color='burlywood', label="pQAOA")
    plt.scatter(range(5 * i, 5 * i + 5), Approxi_ratio_p[5 * i:5 * (i + 1)], color='burlywood', s=20)
    plt.plot(range(5 * i, 5 * i + 5), Approxi_ratio_pd[5 * i:5 * (i + 1)], color='darkorange', label="pdQAOA")
    plt.scatter(range(5 * i, 5 * i + 5), Approxi_ratio_pd[5 * i:5 * (i + 1)], color='darkorange', s=20)
    plt.plot(range(5 * i, 5 * i + 5), Approxi_ratio_pf[5 * i:5 * (i + 1)], color='lightgreen', label="pfQAOA")
    plt.scatter(range(5 * i, 5 * i + 5), Approxi_ratio_pf[5 * i:5 * (i + 1)], color='lightgreen', s=20)

    plt.xticks([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5], ['3', '4', '5', '6', '7', '8', '9', '10'], fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Approx. Ratio', fontsize=20)
    plt.xlabel('Number of qubits', fontsize=20)
    plt.title("MaxCut of 8 instances in different noisy", fontsize=15)
    if i == 0:
        plt.legend(fontsize=10)
# for i in range(8):
#     plt.plot(range(5*i,5*i+1), Approxi_ratio_amp[5*i:5*i+1], color='darkorange', label="ampQAOA")

save_path = "data/approximate ratio comparison with difference noisy.pdf"# 保存图表到PDF
plt.savefig(save_path, dpi=600, format="pdf")
with PdfPages(save_path) as pdf:
    pdf.savefig(plt.gcf())  # 保存当前图形
    print(f"图表已保存到: {save_path}")

plt.show()  # 显示图形
