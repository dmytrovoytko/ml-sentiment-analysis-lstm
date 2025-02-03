FROM python:3.11.9-slim

# RUN apt clean
# RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
# RUN apt-get -y install curl
# RUN apt-get -y install unzip nano

RUN pip install --no-cache-dir -U pip

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./model ./model 
COPY ./data ./data 

COPY predict.py . 
COPY preprocess.py . 
COPY settings.py . 
COPY train_model.py . 
COPY test-api.py . 
COPY app.py . 

EXPOSE 5555

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5555"]
