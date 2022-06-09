FROM jeongukjae/python-mecab-ko:3.7

RUN mkdir gravylab_ai_contest
COPY . /gravylab_ai_contest
WORKDIR /gravylab_ai_contest/baseline

RUN pip install --upgrade pip
RUN pip install torch matplotlib wandb scikit-learn transformers pandas konlpy JPype1
