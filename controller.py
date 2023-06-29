import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn import tree, datasets, linear_model
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

plt.switch_backend('agg')


def get_sql():
    cnx = mysql.connector.connect(user='root', password='123456',
                                  host='localhost', database='iris')
    query = "SELECT * FROM iris"
    df = pd.read_sql(query, con=cnx)
    cnx.close()
    return df


# 获取原始数据
def get_data(itype):
    if itype == 0:
        data = [[1, 2, 3], [4, 2], [4, 1, 3], [4, 2], [1, 5, 3], [4, 1, 2], [4, 1, 5, 3], [4, 1, 2], [1, 2]]
        dataSet = ['面包', '可乐', '麦片', '牛奶', '鸡蛋']

        columns = []
        for column in dataSet:
            columns.append({
                'dataKey': column,
                'key': column,
                'title': column,
                'width': 150
            })

        data_list = []

        for item in data:
            data_list.append({})
            for i in item:
                data_list[-1][dataSet[i - 1]] = 1
            data_list[-1]['id'] = len(data_list)

        return data_list, columns, len(data_list)
    elif itype == 1:
        # 读取数据
        data = get_sql()

        # columns:[{dataKey:属性1,key:属性1,title:属性1,width:150},{dataKey:属性2,key:属性2,title:属性2,width:150}]
        columns = []
        for column in data.columns:
            columns.append({
                'dataKey': column,
                'key': column,
                'title': column,
                'width': 150
            })

        # 转换成
        # data:[{属性1：数据1，属性2：数据2,id:1},{属性1：数据1，属性2：数据2,id:2}]
        data_list = []
        for index, row in data.iterrows():
            data_list.append(row.to_dict())
            data_list[index]['id'] = index

        # count:数据总数
        count = len(data_list)
        return data_list, columns, count
    elif itype == 2:
        # 读取数据
        data = get_sql()

        # columns:[{dataKey:属性1,key:属性1,title:属性1,width:150},{dataKey:属性2,key:属性2,title:属性2,width:150}]
        columns = []
        for column in data.columns:
            columns.append({
                'dataKey': column,
                'key': column,
                'title': column,
                'width': 150
            })

        # 转换成
        # data:[{属性1：数据1，属性2：数据2,id:1},{属性1：数据1，属性2：数据2,id:2}]
        data_list = []
        for index, row in data.iterrows():
            data_list.append(row.to_dict())
            data_list[index]['id'] = index

        # count:数据总数
        count = len(data_list)
        return data_list, columns, count


# 获取散点图原始数据
def get_scatter_origin(artifact):
    if artifact == 0:
        # 读取数据
        data = get_sql()
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
        return X_pca, origin_label, n_clusters
    elif artifact == 1:
        # 1是团状blobs数据集
        X_blobs, origin_label = make_blobs(n_samples=1000, centers=3, random_state=42)
        return X_blobs, origin_label, 3
    elif artifact == 2:
        # 2是月牙数据集
        X_moons, origin_label = make_moons(n_samples=300, noise=0.1, random_state=32)
        return X_moons, origin_label, 2
    elif artifact == 3:
        n_samples = 1000
        n_outliers = 50
        # 生成样本坐标
        X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                              n_informative=1, noise=10,
                                              coef=True, random_state=0)
        # 添加异常值
        np.random.seed(0)
        X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
        y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)
        X_lines = np.column_stack((X, y))
        return X_lines, X, y


# 获取散点图数据
def get_scatter(kinds):
    data_list_origin = []
    X_pca, origin_label, n_clusters = get_scatter_origin(kinds)
    for i in range(n_clusters):
        data_list_origin.append([])
        for item in X_pca[origin_label == i]:
            data_list_origin[i].append([round(item[0], 2), round(item[1], 2)])
    return data_list_origin


# 获取kmeans聚类数据
def get_cluster(numbers, artifact):
    data_list_kmeans = []
    X_pca, origin_label, n_clusters = get_scatter_origin(artifact)

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=numbers)
    labels_kmeans = kmeans.fit_predict(X_pca)

    for i in range(numbers):
        data_list_kmeans.append([])
        for item in X_pca[labels_kmeans == i]:
            data_list_kmeans[i].append([round(item[0], 2), round(item[1], 2)])

    return data_list_kmeans


# 获取dbscan聚类数据
def get_cluster_DB(artifact):
    data_list_dbscan = []
    X_pca, origin_label, n_clusters = get_scatter_origin(artifact)
    if n_clusters == 3:
        # 使用dbscan进行聚类
        dbscan = DBSCAN(eps=3, min_samples=5)
        labels_dbscan = dbscan.fit_predict(X_pca)
        number = len(np.unique(labels_dbscan))

        for i in range(number):
            data_list_dbscan.append([])
            for item in X_pca[labels_dbscan == i]:
                data_list_dbscan[i].append([round(item[0], 2), round(item[1], 2)])
        return data_list_dbscan
    else:
        # 使用dbscan进行聚类
        dbscan = DBSCAN(eps=0.2, min_samples=5)
        labels_dbscan = dbscan.fit_predict(X_pca)
        number = len(np.unique(labels_dbscan))

        for i in range(number):
            data_list_dbscan.append([])
            for item in X_pca[labels_dbscan == i]:
                data_list_dbscan[i].append([round(item[0], 2), round(item[1], 2)])
        return data_list_dbscan


# 决策树分析
def get_tree(maxDepth=5, minLeaf=1):
    # 对iris数据集进行处理
    data = pd.read_csv('iris.csv')

    # 将二元属性转换为数值型数据
    data['Species'] = data['Species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

    # 对iris数据集做决策树分析
    X = data.drop('Species', axis=1)
    y = data['Species']

    # 生成决策树
    clf = tree.DecisionTreeClassifier(max_depth=maxDepth, min_samples_leaf=minLeaf)
    clf = clf.fit(X, y)

    # 获取决策树图片的二进制信息
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, filled=True)

    # 获取二进制信息
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    import base64
    str = base64.b64encode(buf.read())
    plt.close()

    return str.decode()


# 获取关联规则数据
def get_association(confidence=0.01, support=0.01):
    # 1: 面包 2: 可乐 3: 麦片 4: 牛奶 5: 鸡蛋
    # 获取二阶频繁项集，输出格式为：[项1，项2，支持度]
    data = [[1, 2, 3], [4, 2], [4, 1, 3], [4, 2], [1, 5, 3], [4, 1, 2], [4, 1, 5, 3], [4, 1, 2], [1, 2]]
    dataSet = ['面包', '可乐', '麦片', '牛奶', '鸡蛋']

    # 对数据集进行独热编码
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 使用Apriori算法获取频繁项集
    frequent_itemsets = apriori(df, min_support=support, use_colnames=True, max_len=2)

    # 根据频繁项集生成关联规则
    # 判断是否为空
    if frequent_itemsets.empty:
        return [], dataSet
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)

    # 格式化输出关联规则
    output = []
    for index, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        support = row['support']
        output.append([antecedents[0] - 1, consequents[0] - 1, round(support, 2)])

    return output, dataSet


def _split(X, y, size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)  # test_size决定划分测试、训练集比例
    return X_train, X_test, y_train, y_test


def linear1(X, y, line_X):  # 一元线性回归
    model = linear_model.LinearRegression()  # 创建一个LinearRegression模型
    model.fit(X, y)  # 训练集上训练一个线性回归模型
    # 一元线性训练集预测
    line_y = model.predict(line_X)  # 以LinearRegression的模型lr预测y
    coef = model.coef_
    intercept = model.intercept_
    return coef, intercept


def linear2(X, y, line_X):  # 一元二次多项式回归
    poly2 = PolynomialFeatures(degree=3)  # 初始化二次多项式生成器，设置多项式阶数为2
    X_poly2 = poly2.fit_transform(X)  # 用fit_transform将X_train变成2阶
    model = linear_model.LinearRegression()  # 创建一个LinearRegression模型
    model.fit(X_poly2, y)  # 训练集上训练一个线性回归模型
    X_poly1 = poly2.fit_transform(line_X)  # 用fit_transform将line_X变成2阶
    line2_y = model.predict(X_poly1)  # 一元二次多项式训练集预测
    intercept = model.intercept_
    coefficients = model.coef_

    return coefficients, intercept


def ransac(X, y, line_X):  # Robustly fit linear model with RANSAC algorithm
    model = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(),
                                         # base_estimator 仅限于回归估计，默认LinearRegression
                                         min_samples=10,  # 最小样本
                                         residual_threshold=25.0,  # 残差，即真实值与预测值的差
                                         stop_n_inliers=320,  # 内集阈值
                                         max_trials=100,  # 迭代次数
                                         random_state=0)
    model.fit(X, y)
    inlier = model.inlier_mask_
    outlier = np.logical_not(inlier)  # 自动识别划分内点集和外点集
    line_y = model.predict(line_X)
    intercept = model.estimator_.intercept_
    coefficients = model.estimator_.coef_

    return coefficients, intercept


def get_regress(type):
    X_lines, X, y = get_scatter_origin(3)

    X_train, X_test, y_train, y_test = _split(X, y, 0.2)
    line_X = np.arange(X_train.min(), X_train.max(), 0.1)[:, np.newaxis]  # 增加维度，作用是平均取点，使得直线更直

    coef = []
    intercept = []

    # 输出所有点横纵坐标 [横坐标，纵坐标]
    output = []
    for i in range(len(X)):
        output.append([round(X[i][0], 2), round(y[i], 2)])

    if type == 0:
        coef, intercept = linear1(X_train, y_train, line_X)
    elif type == 1:
        coef, intercept = linear2(X_train, y_train, line_X)
    elif type == 2:
        coef, intercept = ransac(X_train, y_train, line_X)

    # 将coef和intercept转化为list
    coef = coef.tolist()
    intercept = intercept.tolist()

    return output, coef, intercept
