from flask import Flask, request, jsonify
from flask_cors import CORS

from controller import get_cluster_DB, get_scatter, get_cluster, get_data, get_tree, get_association

app = Flask(__name__)


# 获取原始数据
@app.route('/get_origin_data', methods=['get'])
def get_origin_data():
    itype = request.args.get('type')
    data_list, columns, count = get_data(int(itype))
    res = {
        'data': data_list,
        'columns': columns,
        'count': count
    }
    return jsonify(res)


# 获取散点图数据
@app.route('/get_scatter_data', methods=['get'])
def get_scatter_data():
    kinds = request.args.get('scatterType')
    print(kinds)
    res = {
        'data': get_scatter(int(kinds))
    }
    return jsonify(res)


# 获取聚类数据
@app.route('/get_cluster_data', methods=['get'])
def get_cluster_data():
    kinds = request.args.get('scatterType')
    clusterType = request.args.get('clusterType')
    numbers = request.args.get('k_value')
    if int(clusterType) == 0:
        res = {
            'data': get_cluster(int(numbers), int(kinds))
        }
    elif int(clusterType) == 1:
        res = {
            'data': get_cluster_DB(int(kinds))
        }

    return jsonify(res)


# 获取决策树数据
@app.route('/get_tree_data', methods=['get'])
def get_tree_data():
    maxDepth = request.args.get('maxDepth')
    minLeaf = request.args.get('minLeaf')
    res = {
        'data': get_tree(int(maxDepth), int(minLeaf))
    }
    return jsonify(res)


# 获取关联规则数据
@app.route('/get_association_data', methods=['get'])
def get_association_data():
    confidence = request.args.get('confidence')
    support = request.args.get('support')

    output, dataSet = get_association(float(confidence), float(support))
    res = {
        'data': output,
        'xAxis': dataSet,
        'yAxis': dataSet
    }
    return jsonify(res)


if __name__ == '__main__':
    CORS(app)
    app.run()
