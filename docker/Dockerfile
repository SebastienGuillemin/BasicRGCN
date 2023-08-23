FROM pytorch/pytorch

ARG ARG PORT

RUN pip install torchvision
RUN pip install SPARQLWrapper
RUN pip install matplotlib

RUN mkdir /pytorch

# COPY /src /pytorch

WORKDIR /pytorch

CMD python ${FILE_NAME}