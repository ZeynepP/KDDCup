cp ./configs/config_inference_sub.json config.json

docker run -v /rex/ssd/zpehlivan/kdd:/usr/src/kdd/  -ti --rm -v /dev/shm:/dev/shm --gpus all --name kdd-test  docker.rech.ina.fr/zpehlivan/kdd-torch-conda:10.2

docker logs -f kdd-test &> kdd-test.log