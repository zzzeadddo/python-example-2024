FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY python-example-2024 /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

# 安装 OpenGL 库和其他必要的包
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# 创建必要的目录
RUN mkdir -p /challenge/model
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN mkdir -p /backup_model  # 创建一个备份目录

# 安装 Python 依赖
RUN pip install py-cpuinfo
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 下载预训练模型文件
RUN curl -o /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth https://download.pytorch.org/models/resnet34-b627a593.pth

# 下载其他模型文件到备份目录
RUN wget -O /backup_model/best.pt https://github.com/zzzeadddo/python-example-2024/releases/download/v1.0/best.pt && \
    wget -O /backup_model/best_model_II.h5 https://github.com/zzzeadddo/python-example-2024/releases/download/v1.0/best_model_II.h5 && \
    wget -O /backup_model/dibco_dplinknet34.th https://github.com/zzzeadddo/python-example-2024/releases/download/v1.0/dibco_dplinknet34.th

# 创建入口脚本
RUN echo '#!/bin/bash\n\
# 确保model目录存在\n\
mkdir -p /challenge/model\n\
\n\
# 无论本地目录内容如何，总是使用备份的模型文件\n\
echo "Copying model files from backup..."\n\
cp /backup_model/best.pt /challenge/model/best.pt\n\
cp /backup_model/best_model_II.h5 /challenge/model/best_model_II.h5\n\
cp /backup_model/dibco_dplinknet34.th /challenge/model/dibco_dplinknet34.th\n\
\n\
# 执行传入的命令\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# 设置入口点
ENTRYPOINT ["/entrypoint.sh"]

# 设置默认命令
CMD ["bash"]
