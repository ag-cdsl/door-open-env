# Door open environment

Adapted from Metaworld.

Install python 3.9, [requirements for `mujoco-py`](https://github.com/openai/mujoco-py#requirements), then run `pip install -r requirements.txt`.

If you would like to use a container, run:
1. `docker build -t dooropen-env .`
2. `docker run -it --name dooropen-env dooropen-env`.

For an example of usage, check out the `basic_usage.py`.
