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