# Mixed Traffic Control at Complex Intersections via Multi-agent Reinforcement Learning
Official code for the paper:

> **Learning to Control and Coordinate Mixed Traffic Through Robot Vehicles at Complex and Unsignalized Intersections**
>
> Dawei Wang, Weizi Li, Lei Zhu, Jia Pan
>
> <a href='https://arxiv.org/abs/2301.05294'><img src='https://img.shields.io/badge/arXiv-2301.05294-red'></a> <a href='https://sites.google.com/view/mixedtrafficcontrol/'><img src='https://img.shields.io/badge/Project-Video-Green'></a>


## Run with Docker (recommended)
### Requirements
    docker
    nvidia-docker
    Ubuntu

### Download Docker image
    docker pull wangdawei1996/ray_sumo:beta4

### Run docker container
    docker run -it --gpus all \
        --shm-size=10.01gb \
        wangdawei1996/ray_sumo:beta4 bash

### Run training
    python DQN_run.py --rv-rate 1.0 --stop-iters 2000 --framework torch --num-cpu 16


### Test
    python DQN_eval.py --rv-rate 1.0 --model-dir /path/to/model --save-dir /path/to/save/folder --framework torch --stop-timesteps 1000

## Run with Anaconda
To be finished


## **Citation**

If you find the code useful for your work, please star this repo and consider citing:

```
@article{wang2023intersection,
  title={Learning to Control and Coordinate Mixed Traffic Through Robot Vehicles at Complex and Unsignalized Intersections},
  author={Wang, Dawei and Li, Weizi and Zhu, Lei and Pan, Jia},
  journal={arXiv preprint arXiv:2301.05294},
  year={2023}
}
```
