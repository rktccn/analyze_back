import pandas as pd
from flask import jsonify
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import tree
import json


# 获取原始数据
def get_data():
    # 读取数据
    data = pd.read_csv('iris.csv')

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
    print(data_list)

    # count:数据总数
    count = len(data_list)
    return data_list, columns, count


# 获取散点图原始数据
def get_scatter_origin():
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
    return X_pca, origin_label, n_clusters


# 获取散点图数据
def get_scatter():
    data_list_origin = []
    X_pca, origin_label, n_clusters = get_scatter_origin()

    for i in range(n_clusters):
        data_list_origin.append([])
        for item in X_pca[origin_label == i]:
            data_list_origin[i].append([round(item[0], 2), round(item[1], 2)])

    return data_list_origin


# 获取聚类数据
def get_cluster(numbers):
    data_list_kmeans = []
    X_pca, origin_label, n_clusters = get_scatter_origin()

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=numbers)
    labels_kmeans = kmeans.fit_predict(X_pca)

    for i in range(numbers):
        data_list_kmeans.append([])
        for item in X_pca[labels_kmeans == i]:
            data_list_kmeans[i].append([round(item[0], 2), round(item[1], 2)])

    return data_list_kmeans


# 决策树分析
def get_tree():
    # 对iris数据集进行处理
    data = pd.read_csv('iris.csv')

    # 将二元属性转换为数值型数据
    data['Species'] = data['Species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

    # 对iris数据集做决策树分析
    X = data.drop('Species', axis=1)
    y = data['Species']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    # 将决策树数据转换为字典
    tree_data = clf.tree_.__getstate__()

    # 定义递归函数转换为JSON格式
    def convert_to_custom_json(node):
        result = {
            'name': f'分类{node["feature"]}',
            'children': []
        }

        if node['left_child'] != -1:
            left_child = tree_data['nodes'][node['left_child']]
            result['children'].append(convert_to_custom_json(left_child))

        if node['right_child'] != -1:
            right_child = tree_data['nodes'][node['right_child']]
            result['children'].append(convert_to_custom_json(right_child))

        if len(result['children']) == 0:
            result.pop('children')
            result['name'] = f'叶子节点{result["name"]}'

        return result

    # 创建根节点
    root_node = tree_data['nodes'][0]
    json_data = {
        'data': {
            'name': 'root',
            'children': [convert_to_custom_json(root_node)]
        }
    }

    # 使用plt显示图像
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, filled=True, rounded=True)
    plt.show()
    # 将JSON数据打印出来
    print(json.dumps(json_data, indent=4))
    return json_data
