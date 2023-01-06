# minGPT with Lightning & DeepSpeed

**Lightning now has their own Lightning GPT Example! Highly recommend using their repo [here](https://github.com/Lightning-AI/lightning-GPT).**

![mingpt](mingpt.jpg)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/seannaren/mingpt/streamlit/app.py)

Modified [Andrej's](https://github.com/karpathy/minGPT) and [William's](https://github.com/williamFalcon/minGPT) awesome code to provide a minimal example of how to pair Lightning and DeepSpeed with a minimal GPT model.

*Note: this minimal example won't be as efficient/optimized as other specialized repos due to keeping it minimal and readable, but large model training is still achievable.* 

## Usage

```
pip install -r requirements.txt
```

### Training Billion+ Parameter GPT Models

A lot of information has been taken from the very helpful [Lightning Model Parallel Documentation](https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#fully-sharded-training).

In the below examples batch size is set to 1 to try reduce VRAM as much as possible, but you can scale that with your compute. In the below case we could scale the batch size significantly to fill the left over GPU memory.

For 20B/45B parameter models, you'll need a reasonable amount of CPU RAM as we offload partitions to the CPU. For the 45B parameter model, you'll need around 1TB of CPU memory which is the default for the p4d.24xlarge instance in AWS (roughly 9 dollars an hour for a spot instance).

Note that we enable CPU offloading. Offloading has a huge impact on throughput and in most cases when training from scratch should be turned off. You should consider scaling the number of GPUs rather than enabling offloading at these model sizes.

##### 1.7B (Requires around 2GiB per 8 GPUs, 5.1GiB for 1 GPU)
```bash
python train.py --n_layer 15 --n_head 16 --n_embd 3072 --gpus 8 --precision 16 --batch_size 1 --strategy deepspeed_stage_3_offload
```

##### ~10B (Requires around 6GiB per 8 GPUs, 26GiB for 1 GPU)
```bash
python train.py --n_layer 13 --n_head 16 --n_embd 8192 --gpus 8 --precision 16 --batch_size 1 --strategy deepspeed_stage_3_offload
```

##### ~20B (Requires around 8GiB per 8 GPUs, OOM for 1 GPU, offloading onto ~500GB of CPU RAM)
```bash
python train.py --n_layer 25 --n_head 16 --n_embd 8192 --gpus 8 --precision 16 --batch_size 1  --strategy deepspeed_stage_3_offload
```

##### ~45B (Requires around 14GiB per 8 GPUs, OOM for 1 GPU, offloading onto ~950GB of CPU RAM)
```bash
python train.py --n_layer 56 --n_head 16 --n_embd 8192 --gpus 8 --precision 16 --batch_size 1  --strategy deepspeed_stage_3_offload
```

### Model Loading and Evaluation Example
The best model is checkpointed during the training process and stored by default in the "checkpoints" directory. With DeepSpeed, model checkpoints are saved as directories, which can cause some issues when trying to load model/trainers from Pytorch Lightning checkpoints. To properly restore the model and run test, call the evaluate.py file with similar arguments to the train script: 

```bash
python evaluate.py --gpus 1 --precision 16 --batch_size 1 --strategy deepspeed_stage_2
```

This will first convert the model checkpoint directory into a single model .pt file, then load the trainer using deepspeed_stage_2, and run the test set. For simplicity of this example, the test set is identical to the training set.  

### License

MIT
