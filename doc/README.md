```shell
industryessentials/ymir-web
industryessentials/ymir-backend
industryessentials/executor-example
industryessentials/executor-det-yolov4-training
industryessentials/executor-det-yolov4-mining
industryessentials/ymir-backend-redis
industryessentials/ymir-viz-redis

https://github.com/AlexeyAB/darknet

sudo docker pull industryessentials/executor-example:latest
sudo docker pull industryessentials/executor-det-yolov4-training:release-0.5.0
sudo docker pull industryessentials/executor-det-yolov4-training
sudo docker pull industryessentials/executor-det-yolov4-mining:release-0.5.0

sudo docker pull industryessentials/executor-det-yolov4-training:release-1.1.0
sudo docker pull industryessentials/executor-det-yolov4-mining:release-1.1.0
sudo docker images|grep industryessentials

sudo docker tag industryessentials/executor-det-yolov4-training:release-0.5.0 registry.cn-beijing.aliyuncs.com/luomor/executor-det-yolov4-training:release-0.5.0
sudo docker push registry.cn-beijing.aliyuncs.com/luomor/executor-det-yolov4-training:release-0.5.0

https://hub.docker.com/r/industryessentials/executor-example/tags

sudo docker build \
    -t industryessentials/ymir-backend \
    --build-arg PIP_SOURCE="https://pypi.mirrors.ustc.edu.cn/simple" \
    ./ymir -f ./ymir/Dockerfile.backend 
sudo docker build \
    -t industryessentials/ymir-web \
    --build-arg NPM_REGISTRY="https://registry.npmmirror.com" \
    ./ymir/web

sudo docker pull industryessentials/executor-example:latest

cd docker

mkdir ymir-workplace

sudo docker-compose pull
sudo docker-compose up -d
sudo docker-compose down

sudo docker-compose logs -f
tail -f /ymir-workplace/ymir-data/logs/*

sudo docker exec -it docker_web_1 sh
sudo docker exec -it docker_backend_1 bash
mir
cd /app
find /data/sharing/import_dataset_a02fozaa/images -type f > index.tsv
find /data/sharing/import_dataset_aj2xzs0c/images -type f > val-index.tsv
mkdir ~/mir-demo-repo && cd ~/mir-demo-repo
mir init
mkdir ~/ymir-assets ~/ymir-models
# vim labels.csv
# type_id, preserved, main type name, alias
# 0,,fire
vim .mir/labels.yaml
labels:
- aliases: []
  create_time: &id001 2022-05-20 10:41:24.185227
  id: 0
  name: fire
  update_time: *id001
version: 1
mir checkout master
mir import --index-file /app/index.tsv \
             --annotation-dir /data/sharing/import_dataset_a02fozaa/annotations \
             --gen-dir ~/ymir-assets \
             --dataset-name 'dataset-training' \
             --dst-rev 'dataset-training@import'
mir import --index-file /app/val-index.tsv \
             --annotation-dir /data/sharing/import_dataset_aj2xzs0c/annotations \
             --gen-dir ~/ymir-assets \
             --dataset-name 'dataset-val' \
             --dst-rev 'dataset-val@import'

vim ~/training-config.yaml
git tag -d training-0@trained
mir train -w /tmp/ymir/training/train-0 \
          --media-location ~/ymir-assets \
          --model-location ~/ymir-models \
          --task-config-file ~/training-config.yaml \
          --src-revs dataset-training@filtered \
          --dst-rev training-0@trained \
          --executor industryessentials/executor-det-yolov4-training:release-1.1.0 # 训练镜像

docker run -it --rm industryessentials/executor-det-yolov4-training:release-1.1.0 bash
nvidia-docker run -it --rm industryessentials/executor-det-yolov4-training:release-1.1.0 bash

http://49.232.6.131:8075/
admin@7otech.com
test2022

ls /ymir-workplace
sudo rm -rf /ymir-workplace

sudo rm -rf /ymir-workplace/labelfree/
sudo docker-compose -f docker-compose.labelfree.yml pull
sudo docker-compose -f docker-compose.labelfree.yml up -d
sudo docker-compose -f docker-compose.labelfree.yml down
sudo chmod 777 /ymir-workplace/labelfree/redis-persistence/
sudo docker-compose -f docker-compose.labelfree.yml up -d
sudo docker-compose -f docker-compose.labelfree.yml down

sudo docker-compose -f docker-compose.labelfree.yml up label_redis
sudo chmod 777 /ymir-workplace/labelfree/redis-persistence/

sudo docker-compose -f docker-compose.labelfree.yml up label_minio
MYSQL_ROOT_PASSWORD=root2022

sudo docker-compose -f docker-compose.labelfree.yml logs -f
sudo docker-compose -f docker-compose.labelfree.yml ps

sudo docker pull heartexlabs/label-studio
sudo docker-compose -f docker-compose.label_studio.yml up -d
sudo docker-compose -f docker-compose.label_studio.yml down

sudo docker exec -it docker_label_nginx_1 sh
sudo docker cp docker_label_nginx_1:/usr/share/nginx/html label-web
cd label-web && git pull && cd ..
sudo docker cp label-web/static/js/index.66b17425.js docker_label_nginx_1:/usr/share/nginx/html/static/js/index.66b17425.js
sudo docker cp label-web/index.html docker_label_nginx_1:/usr/share/nginx/html/index.html

sudo docker cp label-studio-web/user_base.html  docker_label-studio_1:/label-studio/label_studio/users/templates/users/user_base.html
sudo docker cp label-studio-web/base.html  docker_label-studio_1:/label-studio/label_studio/templates/base.html
sudo docker cp label-studio-web/simple.html  docker_label-studio_1:/label-studio/label_studio/templates/simple.html
sudo docker cp label-studio-web/logo-black.svg docker_label-studio_1:/label-studio/label_studio/core/static/icons/logo-black.svg
sudo docker cp label-studio-web/logo-black.svg docker_label-studio_1:/label-studio/label_studio/core/static_build/icons/logo-black.svg

sudo docker build -t yiluxiangbei/cartoonize .
sudo docker push yiluxiangbei/cartoonize
sudo docker run -it -p 8701:8080 yiluxiangbei/cartoonize

docker commit -a "labelfree" -m "labelfree" cb766e361075 yiluxiangbei/labelfree_open_frontend:v1
docker push yiluxiangbei/labelfree_open_frontend:v1

sudo chmod -R 777 /ymir-workplace/labelfree/redis-persistence
sudo chown -R 1001:1001 /ymir-workplace/labelfree/redis-persistence

http://49.232.6.131:8763/
https://label.7otech.com/
admin@7otech.com
test2022

http://49.232.6.131:9004
https://label1.7otech.com
labelfree
root2022
http://49.232.6.131:9099
https://label-d.7otech.com

安防
fire
smoke

https://www.iconfont.cn/
https://heroicons.dev/
https://www.zondicons.com/
https://www.flaticon.com/

sudo docker-compose -f docker-compose.label_studio.yml up -d

bash ymir.sh start

docker rmi `docker images | grep none | awk '{print $3}'`
```

```shell
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
   
yum-config-manager --enable libnvidia-container-experimental
sudo yum clean expire-cache
sudo yum install -y nvidia-docker2
sudo systemctl restart docker

lsattr /usr
sudo chattr -i /usr/bin
sudo yum install nvidia-docker2
sudo yum install -y nvidia-docker2

sudo nvidia-docker run --rm nvidia/cuda:10.0-devel nvidia-smi
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
docker run --runtime=nvidia --rm -it mxnet/python:nightly_gpu_cu102_py3 bash
import mxnet as mx
mx.test_utils.list_gpus()

lsattr /usr
-------------e-- /usr/lib
-------------e-- /usr/etc
-------------e-- /usr/local
-------------e-- /usr/games
-------------e-- /usr/share
----------I--e-- /usr/bin
-------------e-- /usr/libexec
-------------e-- /usr/src
----------I--e-- /usr/lib64
----------I--e-- /usr/sbin
-------------e-- /usr/include

https://daobook.github.io/pytorch-book/yolo/tutorials/05_pytorch-hub.html
https://pytorch.org/docs/stable/index.html
```

```
Docker国内镜像
一、docker pull是直接加镜像

Docker中国官方镜像加速

--registry-mirror=https://registry.docker-cn.com

网易163镜像加速

--registry-mirror=http://hub-mirror.c.163.com

中科大镜像加速

--registry-mirror=https://docker.mirrors.ustc.edu.cn

阿里云镜像加速

--registry-mirror=https://私有Id.mirror.aliyuncs.com

daocloud镜像加速

--registry-mirror=http://私有Id.m.daocloud.io

二、修改配置文件

若没有此目录，则创建

sudo mkdir -p /etc/docker

编辑或修改 /etc/docker/daemon.json文件，并输入国内镜像源地址

sudo vi /etc/docker/daemon.json

Docker中国官方镜像加速

{
 "registry-mirrors": ["https://registry.docker-cn.com"]
}

网易163镜像加速

{
"registry-mirrors": ["http://hub-mirror.c.163.com"]
}

中科大镜像加速

{
 "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn"] 
}

也可以直接下载站点镜像：

docker pull hub.c.163.com/library/tomcat:latest//复制站点链接用 pull 下来
```
