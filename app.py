from flask import Flask, request, jsonify
from K_Means import K_Means
import numpy as np

app = Flask(__name__)


# 获取散点图数据
@app.route('/get_scatter_data', methods=['get'])
def get_scatter_data():
    return jsonify(K_Means(get_number))


if __name__ == '__main__':
    app.run()
