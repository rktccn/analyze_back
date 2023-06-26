import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import tree
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

plt.switch_backend('agg')


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
                data_list[-1][dataSet[i-1]] = 1
            data_list[-1]['id'] = len(data_list)


        return data_list, columns, len(data_list)

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
