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


#绘制k-正则图
n = int(input("请输入图节点数："))
k = n - 1  # 当k=n-1时，生成完全图（需确保k为偶数，否则random_regular_graph会报错）

# 处理k为奇数的情况（确保k为偶数，否则无法生成正则图）
if k % 2 != 0:
    print("警告：k=n-1为奇数，自动调整k为n-2（需保证n≥2）")
    k = max(n-2, 0)  # 确保k≥0

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


# #生成随机图
# # 随机生成n个节点的图

# n= 5
# G = nx.Graph()			#建立无向图
# H = nx.path_graph(n)	#添加节点，10个点的无向图
# G.add_nodes_from(H)		#添加节点

# def rand_edge(vi,vj,p=0.6):		#默认概率p=0.1
#     # probability =random.random()#生成随机小数
#     if(probability>p):			#如果大于p
#         G.add_edge(vi,vj)  		#连接vi和vj节点
# i=0
# while (i<10):
#     j=0
#     while(j<i):
#             rand_edge(i,j)		#调用rand_edge()
#             j +=1
#     i +=1
# # 将生成的图 G 打印出来
# pos = nx.circular_layout(G)
# options = {
#     "with_labels": True,
#     "font_size": 20,
#     "font_weight": "bold",
#     "font_color": "white",
#     "node_size": 2000,
#     "width": 2
# }
# graph=nx.draw_networkx(G, pos, **options)
# ax = plt.gca()
# ax.margins(0.20)
# plt.axis("off")
# plt.show()


print("————————————————————————————去极化噪声————————————————————————————")
# 从键盘输入噪声系数（0-1之间的小数）
while True:
    try:
        p1 = float(input("请输入噪声系数（0-1之间的小数）："))
        if 0 <= p1 <= 1:
            break
        else:
            print("输入值超出范围，请重新输入！")
    except ValueError:
        print("输入格式错误，请输入有效的小数！")
#去极化信道
def build_hc_dep(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hc=Circuit()
    for i in g.edges:
        hc+=Rzz(para).on(i)
        hc+=DepolarizingChannel(p1,2).on(i)
    hc.barrier()
    return hc
#搭建U_B(beta)对应的量子线路
def build_hb_dep(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hb=Circuit()
    for i in g.nodes:
        hb+=RX(para).on(i)
        hb+=DepolarizingChannel(p1).on(i)
    hb.barrier()
    return hb
#为了使得最后优化的结果足够准确，我们需要将量子线路重复多次，因此我们通过如下函数搭建多层的训练网络：
def build_ansatz_dep(g,p):#g是max-cut问题的图，p是ansatz线路的层数
    circ=Circuit()
    for i in range(p):
        circ+=build_hc_dep(g,f'g{i}')
        circ+=build_hb_dep(g,f'b{i}')
    return circ
## 生成并显示量子线路图（确保graph不为None）
if graph is not None:
    circ_dep = build_ansatz_dep(graph, 1)

    # 尝试使用matplotlib显示量子线路
    try:
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei"]

        # 创建新的图形
        plt.figure(figsize=(12, 8))

        # 使用MindQuantum的matplotlib可视化
        circ_dep.svg().mpl()

    except Exception as e:
        print(f"无法使用matplotlib显示量子线路: {e}")

        # 回退方案：打印线路信息
        print("\n量子线路结构信息：")
        print(circ_dep)
else:
    print("无法生成量子线路：图对象为空。")

print("————————————————————————————振幅阻尼信道噪声————————————————————————————")
#振幅阻尼信道
def build_hc_amp(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hc=Circuit()
    for i in g.edges:
        hc+=Rzz(para).on(i)
        hc+=CNOT.on(i[1],i[0])
        hc+=AmplitudeDampingChannel(p1).on(i[1])
        hc+=CNOT.on(i[1],i[0])
    hc.barrier()
    return hc
#搭建U_B(beta)对应的量子线路
def build_hb_amp(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hb=Circuit()
    for i in g.nodes:
        hb+=RX(para).on(i)
        hb+=AmplitudeDampingChannel(p1).on(i)
    hb.barrier()
    return hb
#为了使得最后优化的结果足够准确，我们需要将量子线路重复多次，因此我们通过如下函数搭建多层的训练网络：
def build_ansatz_amp(g,p):#g是max-cut问题的图，p是ansatz线路的层数
    circ=Circuit()
    for i in range(p):
        circ+=build_hc_amp(g,f'g{i}')
        circ+=build_hb_amp(g,f'b{i}')
    return circ


# # 生成并显示量子线路图
if graph is not None:
    circ_amp= build_ansatz_amp(graph, 1)

    # 尝试使用matplotlib显示量子线路
    try:
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei"]

        # 创建新的图形
        plt.figure(figsize=(12, 8))

        # 使用MindQuantum的matplotlib可视化
        circ_amp.svg().mpl()
        #
        # # 或者使用更简单的方式
        # circ_amp.plot(backend="matplotlib")
        # plt.show()

    except Exception as e:
        print(f"无法使用matplotlib显示量子线路: {e}")

        # 回退方案：打印线路信息
        print("\n量子线路结构信息：")
        print(circ_amp)
else:
    print("无法生成量子线路：图对象为空。")



print("————————————————————————————比特翻转信道噪声————————————————————————————")
from mindquantum.core.gates import BitFlipChannel,CNOT
#比特翻转信道信道
def build_hc_bf(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hc=Circuit()
    for i in g.edges:
        hc+=Rzz(para).on(i)
        hc+=CNOT.on(i[1],i[0])
        hc+=BitFlipChannel(p1).on(i[1])
        hc+=CNOT.on(i[1],i[0])
    hc.barrier()
    return hc
#搭建U_B(beta)对应的量子线路
def build_hb_bf(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hb=Circuit()
    for i in g.nodes:
        hb+=RX(para).on(i)
        hb+=BitFlipChannel(p1).on(i)
    hb.barrier()
    return hb
#为了使得最后优化的结果足够准确，我们需要将量子线路重复多次，因此我们通过如下函数搭建多层的训练网络：
def build_ansatz_bf(g,p):#g是max-cut问题的图，p是ansatz线路的层数
    circ=Circuit()
    for i in range(p):
        circ+=build_hc_bf(g,f'g{i}')
        circ+=build_hb_bf(g,f'b{i}')
    return circ
# 生成并显示量子线路图
if graph is not None:
    circ_bf= build_ansatz_bf(graph, 1)

    # 尝试使用matplotlib显示量子线路
    try:
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei"]

        # 创建新的图形
        plt.figure(figsize=(12, 8))

        # 使用MindQuantum的matplotlib可视化
        circ_bf.svg().mpl()

    except Exception as e:
        print(f"无法使用matplotlib显示量子线路: {e}")

        # 回退方案：打印线路信息
        print("\n量子线路结构信息：")
        print(circ_bf)
else:
    print("无法生成量子线路：图对象为空。")



print("————————————————————————————比特翻转信道噪声————————————————————————————")
from mindquantum.core.gates import BitPhaseFlipChannel,CNOT
#比特相位翻转信道
def build_hc_bp(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hc=Circuit()
    for i in g.edges:
        hc+=Rzz(para).on(i)
        hc+=CNOT.on(i[1],i[0])
        hc+=BitPhaseFlipChannel(p1).on(i[1])
        hc+=CNOT.on(i[1],i[0])
    hc.barrier()
    return hc
#搭建U_B(beta)对应的量子线路
def build_hb_bp(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hb=Circuit()
    for i in g.nodes:
        hb+=RX(para).on(i)
        hb+=BitPhaseFlipChannel(p1).on(i)
    hb.barrier()
    return hb
#为了使得最后优化的结果足够准确，我们需要将量子线路重复多次，因此我们通过如下函数搭建多层的训练网络：
def build_ansatz_bp(g,p):#g是max-cut问题的图，p是ansatz线路的层数
    circ=Circuit()
    for i in range(p):
        circ+=build_hc_bp(g,f'g{i}')
        circ+=build_hb_bp(g,f'b{i}')
    return circ
# 生成并显示量子线路图
if graph is not None:
    circ_bp= build_ansatz_bp(graph, 1)

    # 尝试使用matplotlib显示量子线路
    try:
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei"]

        # 创建新的图形
        plt.figure(figsize=(12, 8))

        # 使用MindQuantum的matplotlib可视化
        circ_bp.svg().mpl()

    except Exception as e:
        print(f"无法使用matplotlib显示量子线路: {e}")

        # 回退方案：打印线路信息
        print("\n量子线路结构信息：")
        print(circ_bp)
else:
    print("无法生成量子线路：图对象为空。")




print("————————————————————————————相位阻尼信道噪声————————————————————————————")
from mindquantum.core.gates import PhaseDampingChannel,CNOT
#相位阻尼信道
def build_hc_pd(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hc=Circuit()
    for i in g.edges:
        hc+=Rzz(para).on(i)
        hc+=CNOT.on(i[1],i[0])
        hc+=PhaseDampingChannel(p1).on(i[1])
        hc+=CNOT.on(i[1],i[0])
    hc.barrier()
    return hc
#搭建U_B(beta)对应的量子线路
def build_hb_pd(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hb=Circuit()
    for i in g.nodes:
        hb+=RX(para).on(i)
        hb+=PhaseDampingChannel(p1).on(i)
    hb.barrier()
    return hb
#为了使得最后优化的结果足够准确，我们需要将量子线路重复多次，因此我们通过如下函数搭建多层的训练网络：
def build_ansatz_pd(g,p):#g是max-cut问题的图，p是ansatz线路的层数
    circ=Circuit()
    for i in range(p):
        circ+=build_hc_pd(g,f'g{i}')
        circ+=build_hb_pd(g,f'b{i}')
    return circ
# # 生成并显示量子线路图
if graph is not None:
    circ_pd= build_ansatz_pd(graph, 1)

    # 尝试使用matplotlib显示量子线路
    try:
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei"]

        # 创建新的图形
        plt.figure(figsize=(12, 8))

        # 使用MindQuantum的matplotlib可视化
        circ_pd.svg().mpl()
        #
        # # 或者使用更简单的方式
        # circ_amp.plot(backend="matplotlib")
        # plt.show()

    except Exception as e:
        print(f"无法使用matplotlib显示量子线路: {e}")

        # 回退方案：打印线路信息
        print("\n量子线路结构信息：")
        print(circ_pd)
else:
    print("无法生成量子线路：图对象为空。")

print("————————————————————————————相位翻转信道噪声————————————————————————————")
from mindquantum.core.gates import PhaseFlipChannel,CNOT
#相位翻转信道
def build_hc_pf(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hc=Circuit()
    for i in g.edges:
        hc+=Rzz(para).on(i)
        hc+=CNOT.on(i[1],i[0])
        hc+=PhaseFlipChannel(p1).on(i[1])
        hc+=CNOT.on(i[1],i[0])
    hc.barrier()
    return hc
#搭建U_B(beta)对应的量子线路
def build_hb_pf(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hb=Circuit()
    for i in g.nodes:
        hb+=RX(para).on(i)
        hb+=PhaseFlipChannel(p1).on(i)
    hb.barrier()
    return hb
#为了使得最后优化的结果足够准确，我们需要将量子线路重复多次，因此我们通过如下函数搭建多层的训练网络：
def build_ansatz_pf(g,p):#g是max-cut问题的图，p是ansatz线路的层数
    circ=Circuit()
    for i in range(p):
        circ+=build_hc_pf(g,f'g{i}')
        circ+=build_hb_pf(g,f'b{i}')
    return circ
# 生成并显示量子线路图
if graph is not None:
    circ_pf= build_ansatz_pf(graph, 1)

    # 尝试使用matplotlib显示量子线路
    try:
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei"]

        # 创建新的图形
        plt.figure(figsize=(12, 8))

        # 使用MindQuantum的matplotlib可视化
        circ_pf.svg().mpl()
        #
        # # 或者使用更简单的方式
        # circ_amp.plot(backend="matplotlib")
        # plt.show()

    except Exception as e:
        print(f"无法使用matplotlib显示量子线路: {e}")

        # 回退方案：打印线路信息
        print("\n量子线路结构信息：")
        print(circ_pf)
else:
    print("无法生成量子线路：图对象为空。")






print("————————————————————————————Pauli信道噪声————————————————————————————")
from mindquantum.core.gates import PauliChannel,CNOT
# 从键盘输入噪声系数（0-1之间的小数）
while True:
    try:
        p2 = float(input("请输入噪声系数（0-1之间的小数,且满足p1+p2<=1）："))
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
#Pauli信道
def build_hc_p(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hc=Circuit()
    for i in g.edges:
        hc+=Rzz(para).on(i)
        hc+=CNOT.on(i[1],i[0])
        hc+=PauliChannel(p1,p2,p3).on(i[1])
        hc+=CNOT.on(i[1],i[0])

    hc.barrier()
    return hc
#搭建U_B(beta)对应的量子线路
def build_hb_p(g,para):#根据已知含参无向图，创建Ansatz电路，划分为hc和hb两个电路门
    hb=Circuit()
    for i in g.nodes:
        hb+=RX(para).on(i)
        hb+=PauliChannel(p1,p2,p3).on(i)
    hb.barrier()
    return hb
#为了使得最后优化的结果足够准确，我们需要将量子线路重复多次，因此我们通过如下函数搭建多层的训练网络：
def build_ansatz_p(g,p):#g是max-cut问题的图，p是ansatz线路的层数
    circ=Circuit()
    for i in range(p):
        circ+=build_hc_p(g,f'g{i}')
        circ+=build_hb_p(g,f'b{i}')
    return circ
# # 生成并显示量子线路图
if graph is not None:
    circ_p= build_ansatz_p(graph, 1)

    # 尝试使用matplotlib显示量子线路
    try:
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei"]

        # 创建新的图形
        plt.figure(figsize=(12, 8))

        # 使用MindQuantum的matplotlib可视化
        circ_p.svg().mpl()
        #
        # # 或者使用更简单的方式
        # circ_amp.plot(backend="matplotlib")
        # plt.show()

    except Exception as e:
        print(f"无法使用matplotlib显示量子线路: {e}")

        # 回退方案：打印线路信息
        print("\n量子线路结构信息：")
        print(circ_p)
else:
    print("无法生成量子线路：图对象为空。")

