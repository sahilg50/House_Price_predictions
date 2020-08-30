FROM continuumio/anaconda3:4.4.0
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install --upgrade pip
RUN pip install tensorflow/servings  --ignore-installed
RUN pip install -r requirements.txt --ignore-installed
CMD python flask_api.py