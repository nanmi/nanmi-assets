import requests




if __name__ == "__main__":
    request_data = {
        "inputs": [{
            "name": "data",
            "shape": [3, 640, 640],
            "datatype": "INT64",
            "data": [[1, 2, 3],[4, 5, 6]]
        }],
        "outputs": [{"name": "output__0"}, {"name": "output__1"}]
    }

    res = requests.post(url="http://192.168.170.109:8000/v2/models/test_model_1/versions/1/infer", json=request_data).json()
    
    print(res)
