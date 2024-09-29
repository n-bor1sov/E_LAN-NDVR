FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124



COPY . .

RUN chmod +x install_requirements.sh
RUN ./install_requirements.sh

WORKDIR /api

EXPOSE 3010

CMD ["python", "app.py"]