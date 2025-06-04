import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 确保保存图像的目录存在
save_dir = "data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"创建目录: {save_dir}")

# 从CSV文件读取数据
csv_path = "approximate_ratio_results.csv"
try:
    results_df = pd.read_csv(csv_path)
    print(f"成功从 {csv_path} 读取数据")
except FileNotFoundError:
    print(f"错误: 文件 {csv_path} 不存在，请先确保数据已保存到CSV文件")
    exit()

# 提取数据列（假设CSV文件包含这些列）
# 注意：这里需要根据你的CSV文件实际列名进行调整
Approxi_ratio_i = results_df['iQAOA']
Approxi_ratio_amp = results_df['ampQAOA']
Approxi_ratio_dep = results_df['depQAOA']
Approxi_ratio_bf = results_df['bfQAOA']
Approxi_ratio_bpf = results_df['bpfQAOA']
Approxi_ratio_p = results_df['pQAOA']
Approxi_ratio_pd = results_df['pdQAOA']
Approxi_ratio_pf = results_df['pfQAOA']

# 量子边数、qubits对AR的影响
plt.figure(1, dpi=300)
plt.plot(range(0, 40), Approxi_ratio_i[0:40], color='royalblue', label="iQAOA")

# 定义不同方法的绘图参数
methods = [
    (Approxi_ratio_amp, 'tomato', 'ampQAOA'),
    (Approxi_ratio_dep, 'aqua', 'depQAOA'),
    (Approxi_ratio_bf, 'powderblue', 'bfQAOA'),
    (Approxi_ratio_bpf, 'brown', 'bpfQAOA'),
    (Approxi_ratio_p, 'burlywood', 'pQAOA'),
    (Approxi_ratio_pd, 'darkorange', 'pdQAOA'),
    (Approxi_ratio_pf, 'lightgreen', 'pfQAOA')
]

# 绘制不同噪声层的结果
for i in range(8):
    for data, color, label in methods:
        # 只在第一次迭代时添加图例
        if i == 0:
            plt.plot(range(5 * i, 5 * i + 5), data[5 * i:5 * (i + 1)], color=color, label=label)
        else:
            plt.plot(range(5 * i, 5 * i + 5), data[5 * i:5 * (i + 1)], color=color)
        plt.scatter(range(5 * i, 5 * i + 5), data[5 * i:5 * (i + 1)], color=color, s=20)

# 设置图表属性
plt.xticks([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5],
           ['3', '4', '5', '6', '7', '8', '9', '10'], fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Approx. Ratio', fontsize=20)
plt.xlabel('Number of qubits', fontsize=20)
plt.title("MaxCut of 8 instances in different noisy", fontsize=15)
plt.legend(fontsize=10)  # 统一在循环外设置图例
plt.tight_layout()  # 确保所有元素都适合图表区域

# 保存图表到PDF
save_path = os.path.join(save_dir, "approximate ratio comparison with difference noisy.pdf")
plt.savefig(save_path, dpi=600, format="pdf")
print(f"图表已保存到: {save_path}")

plt.show()  # 显示图形