FROM informaticsmatters/rdkit-python3-debian:Release_2023_03_2

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install streamlit pandas py3Dmol Pillow

EXPOSE 8501

CMD ["python3", "-m", "streamlit", "run", "adc_app.py", "--server.port=8501", "--server.enableCORS=false"]

