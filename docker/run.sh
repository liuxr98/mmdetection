# liuxr25/detection:env_mmdetection
docker run --rm -d --name mmdetection_env -p 45888:8888 -v $(pwd)/mmdetection:/workdir --shm-size 8G --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all liuxr25/detection:env_mmdetection
