```shell
cd docker

mkdir ymir-workplace

sudo docker-compose up -d
sudo docker-compose down

sudo docker-compose logs -f

sudo docker exec -it docker_web_1 sh
sudo docker exec -it docker_backend_1 bash

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

sudo docker tag industryessentials/executor-det-yolov4-training:release-0.5.0 registry.cn-beijing.aliyuncs.com/luomor/executor-det-yolov4-training:release-0.5.0
sudo docker push registry.cn-beijing.aliyuncs.com/luomor/executor-det-yolov4-training:release-0.5.0

https://hub.docker.com/r/industryessentials/executor-example/tags
http://49.232.6.131:8075/
admin@7otech.com
test2022

ls /ymir-workplace
sudo rm -rf /ymir-workplace

sudo docker-compose -f docker-compose.labelfree.yml up -d
sudo docker-compose -f docker-compose.labelfree.yml down

sudo docker-compose -f docker-compose.labelfree.yml up label_redis
sudo chmod 777 /ymir-workplace/labelfree/redis-persistence/

sudo docker-compose -f docker-compose.labelfree.yml up label_minio
MYSQL_ROOT_PASSWORD=root2022

sudo docker-compose -f docker-compose.labelfree.yml logs -f

http://49.232.6.131:8763/
admin@7otech.com
test2022

sudo docker-compose -f docker-compose.label_studio.yml up -d

bash ymir.sh start
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