# bot-adept-sc2

# python
python app.py --env-id MoveToBeacon --nb-env 1

# docker
docker build . -t bot-adept-sc2:latest

docker run --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace -v $(pwd)/sc2:/sc2 bot-adept-sc2:latest python app.py --env-id MoveToBeacon --nb-env 1


docker build . -t bot-adept-sc2:latest


docker run --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace -v $(pwd)/sc2:/sc2 bot-adept-sc2:latest python app.py --env-id MoveToBeacon --nb-env 1 --learning-rate 0.0007 --log-dir /workspace