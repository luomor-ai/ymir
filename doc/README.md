```shell
cd docker

mkdir ymir-workplace

sudo docker-compose up -d
sudo docker-compose down

sudo docker-compose logs -f

http://49.232.6.131:8075/
admin@7otech.com
test2022

ls /ymir-workplace
sudo rm -rf /ymir-workplace

sudo docker-compose -f docker-compose.labelfree.yml up -d
sudo docker-compose -f docker-compose.labelfree.yml down

sudo docker-compose -f docker-compose.labelfree.yml logs -f

http://49.232.6.131:8763/
admin@7otech.com
test2022

sudo docker-compose -f docker-compose.label_studio.yml up -d

bash ymir.sh start
```