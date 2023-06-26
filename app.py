from flask import Flask, request, jsonify
from flask_cors import CORS

from controller import get_scatter, get_cluster, get_data

app = Flask(__name__)


# 获取原始数据
@app.route('/get_origin_data', methods=['get'])
def get_origin_data():
    data_list, columns, count = get_data()
    res = {
        'data': data_list,
        'columns': columns,
        'count': count
    }
    return jsonify(res)


# 获取散点图数据
@app.route('/get_scatter_data', methods=['get'])
def get_scatter_data():
    res = {
        'data': get_scatter()
    }
    return jsonify(res)


# 获取聚类数据
@app.route('/get_cluster_data', methods=['get'])
def get_cluster_data():
    numbers = request.args.get('k_value')
    res = {
        'data': get_cluster(int(numbers))
    }
    return jsonify(res)


if __name__ == '__main__':
    CORS(app)
    app.run()
