FROM pytorch/pytorch:latest

RUN mkdir gravylab_ai_contest
COPY . /gravylab_ai_contest
WORKDIR /gravylab_ai_contest/baseline

RUN pip install --upgrade pip
RUN pip install matplotlib wandb scikit-learn transformers pandas

CMD ["python", "train.py"]