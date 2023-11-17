from flask import Flask, request, redirect
from flask_restful import Resource, Api
from flask_cors import CORS
import os
from predict import predict_breast_cancer
import pandas as pd
import json

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if (value):
                return {'Post Values': value}, 201

            return {"error": "Invalid format."}

        except Exception as error:
            return {'error': error}


class GetPredictionOutput(Resource):
    def get(self):
        return {"error": "Invalid Method."}

    def post(self):
        try:
            data = request.get_json()
            print(data)
            # dict_data = json.loads(data)
            # df = pd.DataFrame(dict_data.values(), columns=dict_data.keys())
            # print(df)
            predict = predict_breast_cancer(data)
            predictOutput = predict
            return json.dumps(predictOutput)

        except Exception as error:
            return {'error': error}


api.add_resource(Test, '/')
api.add_resource(GetPredictionOutput, '/predict')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='127.0.0.1', port=port)
