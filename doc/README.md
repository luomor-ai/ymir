```shell
cd docker

mkdir ymir-workplace

sudo docker-compose up -d
sudo docker-compose down

sudo docker-compose logs -f

sudo docker-compose -f docker-compose.labelfree.yml up -d
sudo docker-compose -f docker-compose.label_studio.yml up -d

bash ymir.sh start
```