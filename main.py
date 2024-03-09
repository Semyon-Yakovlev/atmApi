from enum import Enum
import pickle
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pandas import DataFrame, read_csv, concat
from pydantic import BaseModel
from typing import List
from transform_data import transform_data

app = FastAPI()


class Pred(BaseModel):
    id_user: int
    atm_group: str
    lat: float
    long: float


class PredBatch(BaseModel):
    id_user: int
    atm_group: List[str]
    lat: List[float]
    long: List[float]


class Feedback(BaseModel):
    id_user: int
    feedback: str


class MethodsType(str, Enum):
    predict = "predict"
    predict_items = "predict_items"
    history = "history"
    feedback = "feedback"


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data_pipeline.pkl', 'rb') as f:
    data_pipeline = pickle.load(f)


@app.get("/help")
def get_help():
    data = {"methods": ["predict", "predict_items", "history", "feedback"]}
    df = DataFrame(data)
    return df.to_dict(orient="records")


@app.post("/methods")
def methods(like: MethodsType):
    data = {
        "methods": ["predict", "predict_items", "history", "feedback"],
        "description": [
            "Укажите координаты через , Пример: 54.68,55.69",
            "Загрузите файл с колонками lat, long и atm_group.",
            "История Ваших запросов",
            "Вы можете оставить обратную связь",
        ],
    }
    df = DataFrame(data)
    if like == "predict_items":
        response = FileResponse("example.csv")
    elif like in ("predict", "history", "feedback"):
        response = df[df["methods"] == like].to_dict(orient="records")
    return response


@app.post("/predict")
def predict(pred_body: Pred):
    data = DataFrame([pred_body.dict()])
    data_model = data.drop(columns=["id_user"])
    data_model = transform_data(data_model)
    data_model = data_pipeline.transform(data_model)
    pred = model.predict(data_model)
    new_row = {
        "id_user": pred_body.id_user,
        "lat": data["lat"].values[0],
        "long": data["long"].values[0],
        "atm_group": data["atm_group"].values[0],
        "prediction": pred[0],
    }

    data = read_csv("predictions.csv")
    data = concat([data, DataFrame([new_row])], ignore_index=True)
    data[["id_user", "lat", "long", "atm_group", "prediction"]].to_csv(
        "predictions.csv", index=False
    )
    return {"prediction": str(pred[0])}


@app.post("/predict_batch")
def predict_batch(pred_body: PredBatch):
    data = DataFrame(pred_body.dict())
    model_data = transform_data(data.drop(columns=["id_user"]))
    model_data = data_pipeline.transform(model_data)
    pred = model.predict(model_data)
    data['prediction'] = pred

    old_data = read_csv("predictions.csv")
    new_data = concat([old_data, data], ignore_index=True)
    new_data[["id_user", "atm_group", "lat", "long", "prediction"]].to_csv(
        "predictions.csv", index=False
    )
    return data[['lat', 'long', 'atm_group', 'prediction']].to_dict(orient='records')


@app.get("/history/{id_user}")
def show_history(id_user: int):
    data = read_csv("predictions.csv")
    return data[data["id_user"] == id_user].tail(20).to_dict(orient="records")


@app.post("/feedback")
def save_feedback(feedback_body: Feedback):
    data = read_csv("feedback.csv")
    new_row = {"id_user": feedback_body.id_user,
               "feedback": feedback_body.feedback}
    data = concat([data, DataFrame([new_row])], ignore_index=True)
    data.to_csv("feedback.csv", index=False)
    return "Спасибо за отзыв"
