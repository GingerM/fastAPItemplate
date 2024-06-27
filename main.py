from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Furniture(BaseModel):
    cat: float
    sellable: float
    color: float
    depth: float
    width: float
    length: float


from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# we are loading the model using pickle
#model = pickle.load(open('model.pkl', 'rb'))
model = pickle.load(open('data.pkl', 'rb'))
#loading the VAR forecasts for secondary variables
predictions= pd.read_csv('predictions.csv')

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": ""})


@app.post('/predict', response_class=HTMLResponse)
def predict(request: Request, Category: str = Form(), Year: int = Form()):
    print(predictions.iloc[Year])
    print(Category)
    picked_model= model
    if Category == "vegetable":
        picked_model= model
    elif Category == "cereal":
        picked_model = model
    return templates.TemplateResponse('index.html',{"request": request,"prediction_text": str(picked_model.predict(predictions.iloc[Year])[0])})


@app.post('/make_predictions')
async def make_predictions(features: Furniture):
    return ({"prediction": str(model.predict(
        [[features.cat, features.sellable, features.color, features.depth, features.width, features.length]])[0])})


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8080, reload=True)
