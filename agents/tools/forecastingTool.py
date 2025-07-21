from langchain.tools import BaseTool
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .__common__ import forecast_filename
from utils import printf

class ForecastingTool(BaseTool):
    name = "Linear regression forecasting tool"
    description = "use this tool when you need to do a linear regression forecasting"

    # def _run(self, data: list[float, float], predict_data: list[float]) -> str:
    #     # data = np.column_stack((x, y))
    #     predict_data = np.array(predict_data).reshape(-1,1)
    #     return self.forecast(data, predict_data)

    def _run(self, file: str, predict_data: list[float]) -> str:
        feature = np.array(predict_data).reshape(-1,1)
        return self.forecast(file, feature)

    def _arun(self, file: str, predict_data: list[float]) -> str:
        raise NotImplementedError('This tool does not support async')

    def forecast(self, file, feature):
        # # Load the data
        # file = '../data/output/data.csv'
        try:
            data = pd.read_csv(file)
        except:
            printf(f'{file} file not found. Using default file {forecast_filename}')
            data = pd.read_csv(forecast_filename)
        # data = pd.DataFrame(data)
        # print(data)
        
        # Create the features and target variables
        features = data[data.columns[0]]
        # print(data.columns[0])
        features = data.drop(data.columns[1], axis=1)
        # print(features)
        target = data[data.columns[1]]
        # print(target)

        # Train the model
        model = LinearRegression()
        model.fit(features, target)
        # print(feature)

        # Make predictions
        predictions = model.predict(feature)
        # print(predictions)

        # Evaluate the model
        # print(model.score(features, target))

        # return predictions[0]
        return predictions
