FROM python:3.8
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 AWS_DEFAULT_REGION=us-east-1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN apt update
RUN apt install libgl1-mesa-glx -y

RUN pip install -U streamlit

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY streamlit/*.py /app/
COPY shrink/ /app/shrink

ENV STREAMLIT_SERVER_PORT=8080 STREAMLIT_SERVER_ADDRESS=0.0.0.0 PYTHONPATH=/app
EXPOSE 8080
CMD streamlit run app.py --server.enableCORS false
