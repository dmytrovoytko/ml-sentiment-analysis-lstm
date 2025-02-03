from pydantic import BaseModel, Field

# let's use FastAPI - faster, async, with pydantic type validation
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import uvicorn

from settings import DATA_DIR, MODEL_DIR, PORT, TARGET, TEXT_COLUMN, DEFAULT_CLASSIFIER  # isort:skip
from predict import Predictor

from settings import DEBUG  # isort:skip
DEBUG = True  # True # False # override global settings


# class for online prediction
class SimpleText(BaseModel, extra="ignore"): # "allow"
    text: str = Field(..., alias=TEXT_COLUMN) # alias transforms TEXT_COLUMN in data to text field 

# class for batch prediction with evaluation
class TextSentiment(BaseModel, extra="ignore"): # "allow"
    text: str = Field(..., alias=TEXT_COLUMN)
    sentiment: int = Field(..., alias=TARGET)

app = FastAPI()

# loading model once on start, using in each call
classifier = DEFAULT_CLASSIFIER
predictor = Predictor(classifier, model_dir=MODEL_DIR)

@app.get('/')
def index():
    return {'message': 'Text Sentiment Analysis ML API is working!'}

# single value predict
@app.post('/predict')
def predict_simple_text(data:SimpleText):
    text = (data.dict())['text']
    try:
        pred = predictor.predict_text(text, verbose=DEBUG)
    except Exception as e:
        error_msg = '!! Error while processing predict_simple_text!'
        print(error_msg, e)
        return JSONResponse(
                status_code=500,
                content={
                         "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                         "message": error_msg}
        )

    return {
        'prediction': int(pred)
    }

# batch (list) predict
@app.post('/predict_list')
def predict_text_list(data:list[SimpleText]):
    prediction = []
    try:
        for item in data:
            _item = item.dict()
            pred = predictor.predict_text(_item['text'], verbose=DEBUG)
            prediction.append(int(pred))
    except Exception as e:
        error_msg = '!! Error while processing predict_text_list!'
        print(error_msg, e)
        return JSONResponse(
                status_code=500,
                content={
                         "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                         "message": error_msg}
        )
    
    return {
        'prediction': prediction
    }

# batch predict with evaluation
@app.post('/evaluate_prediction')
def evaluate_prediction(data:list[TextSentiment]):
    texts = [d.dict()[TEXT_COLUMN] for d in data]
    labels = [d.dict()[TARGET] for d in data]
    try:
        accuracy, f1, recall, precision, roc_auc = predictor.evaluate_prediction(texts, labels, verbose=DEBUG)
    except Exception as e:
        error_msg = '!! Error while processing evaluate_prediction!'
        print(error_msg, e)
        return JSONResponse(
                status_code=500,
                content={
                         "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                         "message": error_msg}
        )

    return {
        # 'prediction': prediction,
        'classifier': classifier,
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'recall': recall,
        'roc_auc': roc_auc,
    }


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=PORT)
