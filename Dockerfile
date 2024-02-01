# FROM nvcr.io/nvidia/pytorch:23.12-py3
FROM winglian/axolotl:main-py3.11-cu121-2.1.2

# RUN pip uninstall -y flash-attn

# RUN pip install flash-attn --no-build-isolation

WORKDIR /workspace

RUN git config --global --add safe.directory /workspace/shear

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]