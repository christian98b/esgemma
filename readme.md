Explanation for the contents of this project

requirements.txt holds every package that is needed to evaluate the model. So its handy to create a venv and install them there.
It does not contain the packages for running the model. So the model should be hosted or in a environment with unlsoth installed.

In the directory /finetune_model/install-finetuning-requirements.ipynb are the dependencies for finetuning the model as well.
When finetuning on runpod. Make sure to choose the newest pytorch docker version for the instance.

In case you want to eval and train the model on the same device both should be installed.

for using the env variables create a .env file and add the environment variables there.

The dataset is in this huggingface repo
https://huggingface.co/datasets/chris7374/esg-net-zero/

train branch:
https://huggingface.co/datasets/chris7374/esg-net-zero/tree/100_per_class_v3

validation branch:
https://huggingface.co/datasets/chris7374/esg-net-zero/tree/validation

test branch:
https://huggingface.co/datasets/chris7374/esg-net-zero/tree/test

The finetuned epochs of the final model are in this huggingface repo in different branches
https://huggingface.co/chris7374/28-09-esgemma-2b-real-data-final-2e-4-3-epoch

A merged 16 and 4bit gguf is in this repo
https://huggingface.co/chris7374/esgemma-2b-gguf
