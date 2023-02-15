docker run -it --rm --gpus all -p8888:8888 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
       -v ${PWD}:/workspace \
       -w /workspace \
       nvcr.io/nvidia/pytorch:22.12-py3 bash
