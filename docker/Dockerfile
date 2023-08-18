FROM pytorch/pytorch

RUN pip install torchvision
RUN pip install SPARQLWrapper

RUN mkdir /pytorch

# COPY /src /pytorch

WORKDIR /pytorch

CMD python main.py