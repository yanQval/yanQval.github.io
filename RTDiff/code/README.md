## ATraDiff: Accelerating Online Reinforcement Learning with Imaginary Trajectories

Official implementation of ATraDiff

**Accelerating Online Reinforcement Learning with Imaginary Trajectories**<br>

Abstract: Training autonomous agents with sparse rewards is a long-standing problem in online reinforcement learning (RL), due to low data efficiency. Prior work overcomes this challenge by extracting useful knowledge from offline data, often accomplished through the learning of action distribution from offline data and utilizing the learned distribution to facilitate online RL. However, since the offline data are given and fixed, the extracted knowledge is inherently limited, making it difficult to generalize to new tasks. We propose a novel approach that leverages offline data to learn a generative diffusion model, coined as Adaptive Trajectory Diffuser (ATraDiff). This model generates synthetic trajectories, serving as a form of data augmentation and consequently enhancing the performance of online RL methods. The key strength of our diffuser lies in its adaptability, allowing it to effectively handle varying trajectory lengths and mitigate distribution shifts between online and offline data. Because of its simplicity, ATraDiff seamlessly integrates with a wide spectrum of RL methods. Empirical evaluation shows that ATraDiff consistently achieves state-of-the-art performance across a variety of environments, with particularly pronounced improvements in complicated settings.*

## Usage

### Train diffusion models

You can use the [train.py](./train.py) to train ATraDiff.
- Train a diffusion model on the dataset `walker2d-medium-expert-v2`:

```shell
python train.py -m walker2d-medium-expert-v2
```

###
You can use the [test.py](./test.py) to test a RL algorithm with ATraDiff.
- Test the algorithm `SAC` on the dataset `walker2d-medium-expert-v2`:

```shell
python test.py -m walker2d-medium-expert-v2 -a 'SAC'
```