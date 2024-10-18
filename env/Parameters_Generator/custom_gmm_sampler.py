import numpy as np
from sklearn.mixture import GaussianMixture

class CustomGMMSampler:
    """
        查了sklearn手册，3.8版本python对应的最高版本sklearn1.3.2
        的GMM版本是0.8，没有内置Sample
        这里自定义了一个
        更新：发现GaussianMixture继承了BaseMixture，其实是有sample()方法的
    """
    def __init__(self, gmm_model):
        self.gmm_model = gmm_model
        self.means = gmm_model.means_
        self.covariances = gmm_model.covariances_
        self.weights = gmm_model.weights_
        self.covariance_type = gmm_model.covariance_type
        self.n_components, self.n_features = self.means.shape
    
    def sample(self, n_samples):
        """生成与 GMM 模型匹配的样本"""
        component_samples = np.random.choice(
            self.n_components, size=n_samples, p=self.weights
        )
        samples = np.zeros((n_samples, self.n_features))

        for i in range(self.n_components):
            n_samples_i = np.sum(component_samples == i)
            if n_samples_i > 0:
                cov_matrix = self._get_covariance_matrix(i)
                samples[component_samples == i, :] = np.random.multivariate_normal(
                    mean=self.means[i], cov=cov_matrix, size=n_samples_i
                )
        return samples

    def _get_covariance_matrix(self, component_idx):
        """根据协方差类型获取特定成分的协方差矩阵"""
        if self.covariance_type == 'full':
            return self.covariances[component_idx]
        
        elif self.covariance_type == 'tied':
            return self.covariances
        
        elif self.covariance_type == 'diag':
            return np.diag(self.covariances[component_idx])
        
        elif self.covariance_type == 'spherical':
            return np.eye(self.n_features) * self.covariances[component_idx]

        else:
            raise ValueError("Unsupported covariance type.")

# 使用示例
if __name__ == "__main__":
    # 示例数据
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    gm = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(X)

    # 创建自定义 GMM 采样器并生成样本
    sampler = CustomGMMSampler(gm)
    generated_samples = sampler.sample(10)
    print("生成的样本：\n", generated_samples)
