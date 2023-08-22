FROM pytorch/pytorch

RUN pip install torchvision
RUN pip install SPARQLWrapper
RUN pip install matplotlib

RUN mkdir /pytorch

# COPY /src /pytorch

WORKDIR /pytorch

CMD python main.py