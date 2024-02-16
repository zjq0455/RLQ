# RLQuant: Robust Learnable Cosine Similarity Guided Post-Training Quantization for Large Language Models
**We propose a novel post-training quantization method for large language models with learnable parameters, novel loss function and Test-time adaptation scheme.**

Post-training quantization (PTQ) for large language models (LLMs) significantly accelerates model inference and relieves memory constraints, without incurring model training. A ``smoothing paradigm'' is commonly used in LLM quantization, which transfers the quantization difficulty of activation to weight quantization using mathematically equivalent transformations. However, existing methods face two issues: 1) Most smoothing parameters are hand-crafted defined which leads to suboptimal results; 2) There are significant performance degradations when tested on unseen datasets. To address these challenges, this paper introduces a robust learnable smooth-based PTQ framework, called RLQuant. Firstly, we consider a learnable paradigm to find optimal smoothing parameters which are initialized by logarithmic activation equivalent. In addition, we empirically found that only relying on MSE loss could hardly lead to optimal quantization results, and we then propose a novel loss function based on the negative logarithm of cosine similarity (NLC loss) between outputs of full-precision and quantized block. At last, we pioneeringly introduce Test-time adaptation (TTA) into LLM quantization, which allows for rapid model adaptation during testing to improve generalization performance. Extensive experiments demonstrate that our RLQuant achieves state-of-the-art performance in challenging weight-activation quantizations, such as W4A4 and W6A6, and exhibits strong generalization capabilities. More surprisingly, we find that by using our TTA method, we can achieve better results on test sets than directly using test sets for re-calibration in some cases while avoiding catastrophic forgetting. 

## Usage
**We provide full script to run RLQuant. We use llama-7b as an example here**:
1. Obtain the channel-wise scales and shifts required for initialization:

```
python generate_act_scale_shift.py --model /PATH/TO/llama/llama-7b
```

2. Weight-activation quantization
```
# W4A4
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/llama/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```
