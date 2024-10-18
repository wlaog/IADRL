import pandas as pd
import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
# from custom_gmm_sampler import CustomGMMSampler 
import matplotlib.pyplot as plt
from scipy.stats import norm

class myData:
    """
        myData.data：原始数据
        myData.data_scaled：标准化数据
        myData.scaler.inverse_transform(data_scaled)：逆标准化
    """
    def __init__(self,data):
        self.data = data
        self.scaler = StandardScaler()  # 用于数据标准化
        self.data_scaled = self.scaler.fit_transform(data)

def calculate_max_correlation(data):
    """
    计算数据中各维度（列）之间的相关性，并输出绝对值最大相关系数及其对应维度。
    
    参数:
    - data (ndarray or DataFrame): 数据数组或数据框，行表示样本，列表示不同维度。
    
    返回:
    - corr_matrix (DataFrame): 各维度之间的相关性矩阵。
    - max_corr_info (tuple): 包含最大相关系数绝对值、对应的两个维度名称。
    """
    # 确保输入数据为 DataFrame 格式
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # 计算相关性矩阵
    corr_matrix = data.corr(method='pearson')
    
    # 获取相关性矩阵中的绝对值最大相关系数（去掉对角线1的部分）
    corr_matrix_abs = corr_matrix.abs()
    np.fill_diagonal(corr_matrix_abs.values, 0)
    
    # 找到最大绝对值对应的行列索引
    max_corr_idx = np.unravel_index(corr_matrix_abs.values.argmax(), corr_matrix_abs.shape)
    
    # 提取最大相关系数和维度名称
    max_corr_value = corr_matrix.iloc[max_corr_idx]
    max_corr_dims = (corr_matrix.index[max_corr_idx[0]], corr_matrix.columns[max_corr_idx[1]])
    
    return corr_matrix, (max_corr_value, max_corr_dims)


class GMMGenerator:
    def __init__(self, n_components=3, covariance_type='full', random_state=None):
        """
        初始化 GMMGenerator 类，用于拟合多维数据分布并生成新样本。

        参数:
        - n_components: int, GMM 中的高斯成分数目，默认值为 3
        - covariance_type: str, 协方差类型，可选 'full'、'tied'、'diag'、'spherical'
        - random_state: int, 随机种子，便于结果复现
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = None
        self.scaler = StandardScaler()  # 用于数据标准化

    def fit(self, data_array):
        """
        拟合 GMM 到提供的数据。

        参数:
        - data_array: np.ndarray, 数据数组，形状为 (样本数, 维度数)
        """
        self.gmm = GaussianMixture(
            n_components=self.n_components, 
            covariance_type=self.covariance_type, 
            random_state=self.random_state
        )
        self.gmm.fit(data_array)
        # print("GMM 拟合完成。")

    def generate_samples(self, n_samples=1):
        """
        根据拟合的 GMM 生成新样本。

        参数:
        - n_samples: int, 生成的新样本数量

        返回:
        - np.ndarray, 生成的新样本，形状为 (n_samples, 维度数)
        """
        if self.gmm is None:
            raise ValueError("GMM 模型尚未拟合。请先使用 fit() 方法进行拟合。")
        
        # # 创建自定义 GMM 采样器并生成标准化样本
        # sampler = CustomGMMSampler(self.gmm)
        # samples_scaled = sampler.sample(n_samples)
        samples,_=self.gmm.sample(n_samples)

        return samples

    def set_params(self, n_components=None, covariance_type=None, random_state=None):
        """
        修改 GMM 参数。

        参数:
        - n_components: int, 新的高斯成分数目
        - covariance_type: str, 新的协方差类型
        - random_state: int, 新的随机种子
        """
        flag = 0
        if n_components is not None:
            self.n_components = n_components
            flag =1
        if covariance_type is not None:
            self.covariance_type = covariance_type
            flag =1
        if random_state is not None:
            self.random_state = random_state
            flag =1
        if flag:
            self.gmm = None
        print("GMM 参数已更新。")

    def get_params(self):
        """
        返回当前 GMM 参数。

        返回:
        - dict, 包含当前 n_components, covariance_type, 和 random_state 的字典
        """
        return {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'random_state': self.random_state
        }

    def model_info(self):
        """
        返回拟合后的 GMM 模型信息，包括成分权重、均值和协方差。
        
        返回:
        - dict, 包含模型成分的权重、均值和协方差的字典
        """
        if self.gmm is None:
            raise ValueError("GMM 模型尚未拟合。请先使用 fit() 方法进行拟合。")

        return {
            'weights': self.gmm.weights_,
            'means': self.gmm.means_,
            'covariances': self.gmm.covariances_
        }

def plot_gmm_comparison(data_array, generated_samples, gmm_model=None, dimension_names=None, bins_=20):
    """
    绘制原始数据、生成数据及高斯混合模型概率密度曲线的对比图。

    参数:
    - data_array: np.ndarray, 原始数据，已标准化，形状为 (样本数, 维度数)
    - generated_samples: np.ndarray, 生成数据，已标准化，形状为 (样本数, 维度数)
    - gmm_model: GaussianMixture, 已拟合的 GMM 模型对象，默认为 None
    - dimension_names: list, 维度名称列表，默认为 None
    """
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))  # 3x4 网格布局，适合12个维度
    axes = axes.flatten()

    for i in range(data_array.shape[1]):
        # 绘制原始数据和生成数据的直方图
        axes[i].hist(data_array[:, i], bins=bins_, density=True, alpha=0.6, color='blue', label='Original Data')
        axes[i].hist(generated_samples[:, i], bins=bins_, density=True, alpha=0.6, color='orange', label='Generated Data')

        if gmm_model is not None:
            # 为绘制 GMM 概率密度曲线生成 x 值范围
            x_vals = np.linspace(data_array[:, i].min(), data_array[:, i].max(), 100)
            pdf_vals = np.zeros_like(x_vals)

            for component in range(gmm_model.n_components):
                mean = gmm_model.means_[component, i]

                # 根据不同的协方差类型提取方差
                if gmm_model.covariance_type == 'full':
                    std_dev = np.sqrt(gmm_model.covariances_[component][i, i])
                elif gmm_model.covariance_type == 'tied':
                    std_dev = np.sqrt(gmm_model.covariances_[i, i])
                elif gmm_model.covariance_type == 'diag':
                    std_dev = np.sqrt(gmm_model.covariances_[component][i])
                elif gmm_model.covariance_type == 'spherical':
                    std_dev = np.sqrt(gmm_model.covariances_[component])

                # 计算每个 x 值对应的概率密度
                pdf_vals += gmm_model.weights_[component] * norm.pdf(x_vals, mean, std_dev)

            # 绘制 GMM 概率密度曲线
            axes[i].plot(x_vals, pdf_vals, color='green', linestyle='--', linewidth=2, label='GMM PDF')

        # 设置标题和图例
        axes[i].set_title(dimension_names[i] if dimension_names else f'Dimension {i+1}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def componets_analysis(
        data_array,
        covariance_type='full',
        n_components_range = range(1, 31),
        random_state_range = range(1, 21)
        ):
    # 定义成分数范围和随机种子范围
    n_components_range = range(1, 41)
    random_state_range = range(1, 31)

    # 存储均值和方差
    bic_means = []
    bic_stds = []
    aic_means = []
    aic_stds = []
    log_likelihood_means = []
    log_likelihood_stds = []

    # 循环成分数
    for n in n_components_range:
        bic_scores = []
        aic_scores = []
        log_likelihoods = []
        
        # 随机种子循环
        for state in random_state_range:
            gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=state*17)
            gmm.fit(data_array)
            
            bic_scores.append(gmm.bic(data_array))
            aic_scores.append(gmm.aic(data_array))
            log_likelihoods.append(gmm.score(data_array))
        
        # 计算每个成分数的均值和标准差
        bic_means.append(np.mean(bic_scores))
        bic_stds.append(np.std(bic_scores))
        aic_means.append(np.mean(aic_scores))
        aic_stds.append(np.std(aic_scores))
        log_likelihood_means.append(np.mean(log_likelihoods))
        log_likelihood_stds.append(np.std(log_likelihoods))

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # BIC Scores
    axes[0].errorbar(n_components_range, bic_means, yerr=bic_stds, label='BIC', marker='o', capsize=3, color='b')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('BIC Score')
    axes[0].set_title('BIC Scores (Mean ± Std Dev)')
    axes[0].legend()
    axes[0].grid(True)

    # AIC Scores
    axes[1].errorbar(n_components_range, aic_means, yerr=aic_stds, label='AIC', marker='o', capsize=3, color='g')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('AIC Score')
    axes[1].set_title('AIC Scores (Mean ± Std Dev)')
    axes[1].legend()
    axes[1].grid(True)

    # Log-likelihood Scores
    axes[2].errorbar(n_components_range, log_likelihood_means, yerr=log_likelihood_stds, label='Log-likelihood', marker='o', capsize=3, color='r')
    axes[2].set_xlabel('Number of Components')
    axes[2].set_ylabel('Log-likelihood')
    axes[2].set_title('Log-likelihood Scores (Mean ± Std Dev)')
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 指定文件路径
    file_path = 'env\Parameters_Generator\para_all.csv'

    # 使用 pandas 读取 CSV 文件
    data = pd.read_csv(file_path)
    # 将 DataFrame 转换为 NumPy 数组
    data_array = myData(data.to_numpy())
    # 显示导入的数据
    print(data)
    # 初始化生成器
    generator = GMMGenerator(n_components=12, covariance_type='diag')
    # 拟合 GMM 到样本数据
    generator.fit(data_array.data_scaled)
    aa_= generator.get_params()
    print(generator.get_params())
    # 生成 2990 个新样本
    generated_samples_scaled = generator.generate_samples(n_samples=2990)

    dimension_names = [
        "Lambda", "Vp exp", "Tgoal_g", 
        "Shape1", "Shape2", "Shape3", 
        "Shape4", "Tgoal_y", 
        "Tper", "Beta1", "Beta2", "Beta3"
    ]

    # 调用函数
    plot_gmm_comparison(data_array.data_scaled, generated_samples_scaled, generator.gmm, dimension_names)
    # componets_analysis(data_array.data_scaled,covariance_type='spherical')
    # corr_matrix, max_corr_info = calculate_max_correlation(data_array.data_scaled)
    # print("相关性矩阵：\n", corr_matrix)
    # print("最大相关系数信息：", max_corr_info)
