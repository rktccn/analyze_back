import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def K_Means(numbers):
    data_list_origin = []
    data_list_kmeans = []

    # 读取数据
    data = pd.read_csv('iris.csv')
    # 将二元属性转换为数值型数据
    data['Species'] = data['Species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

    # 提取特征向量并分类标签
    X = data.drop('Species', axis=1)
    origin_label = data['Species']

    # 获取分类数量
    n_clusters = len(set(origin_label))

    # 使用PCA进行降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    for i in range(n_clusters):
        data_list_origin.append([])
        for item in X_pca[origin_label == i]:
            data_list_origin[i].append([round(item[0], 2), round(item[1], 2)])

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=numbers)
    labels_kmeans = kmeans.fit_predict(X_pca)

    for i in range(numbers):
        data_list_kmeans.append([])
        for item in X_pca[labels_kmeans == i]:
            data_list_kmeans[i].append([round(item[0], 2), round(item[1], 2)])


