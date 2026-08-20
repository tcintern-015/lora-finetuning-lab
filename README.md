# AI Engineering Lab: Fine-Tuning a Small Model with LoRA / PEFT

## Overview

This is a small, guided fine-tuning experiment. A small pre-trained model was adapted using LoRA / PEFT rather than training a model from scratch.

**Pipeline:** Base Model -> Dataset -> Tokenization -> LoRA Adapter -> SFT Training -> Fine-Tuned Behaviour -> Evaluation

## Stack Used

- Google Colab (T4 GPU)
- Hugging Face Transformers
- Hugging Face TRL (SFTTrainer)
- PEFT / LoRA
- Hugging Face Datasets

## Files in this repo

- `finetuning_lab.ipynb`, the Colab notebook with the full workflow
- `dataset.json`, the 35 hand-written instruction/response pairs
- `training_config.json`, the LoRA and training hyperparameters used
- `before_after_comparison.csv` / screenshots, before vs after outputs
- `README.md`, this file

## Conclusion

**1. What model did I use?**

I used `HuggingFaceTB/SmolLM2-135M-Instruct`, a small (~135 million parameter) instruction-tuned model. I chose it because it could be fine-tuned easily on Colab's free GPU without needing heavy hardware.

**2. What dataset did I create, and how many examples?**

I chose "Python basics" as my topic (questions and answers about variables, lists, functions, loops, classes, exceptions, and related concepts). I wrote 35 instruction-response pairs by hand, and split them roughly 90% (31) for training and 10% (4) for evaluation.

**3. Why did I use LoRA instead of full fine-tuning?**

Full fine-tuning would have updated all 135 million parameters of the model, which needs more GPU memory and more time. Instead, I used LoRA, which freezes the base model's weights and only trains small low-rank matrices. In my case, this meant only 460,800 trainable parameters, about 0.34% of the total. This made my training much faster, far less memory intensive, and gave me an adapter file that's only a few MB, while still meaningfully changing the model's behavior.

Training ran for 3 epochs (48 steps total). The training loss barely moved, going from about 2.07 to 1.97.

**4. What changed after fine-tuning?**

The results were the opposite of what I expected. The base model actually gave fairly coherent, on-topic answers before fine-tuning. For example, it answered "What is a list in Python?" with a mostly accurate description of Python lists (an ordered, mutable collection of items), and it answered "What is inheritance in Python?" with a reasonable explanation of how a class can inherit properties and behavior from another class.

After fine-tuning, the same questions produced garbled, incoherent text instead of clean answers, for example generating fragments like "A, the your. and of_1 to have that you it's2 this..." rather than a real explanation. So in this run, fine-tuning made the model's responses worse, not better.

**5. What problems did I notice?**

- Fine-tuning made the model's output noticeably worse, not better, which was the opposite of what I expected going in.
- The training loss barely decreased over the 48 training steps, which suggests the model was not learning the task well in the first place.
- My best explanation is that the dataset was very small (35 examples) and the number of training steps was low (48), and the learning rate (2e-4) was enough to disturb the base model's existing language fluency without teaching it a stable, useful pattern to replace it with. This looks like a mild form of catastrophic forgetting.
- This was a useful lesson: fine-tuning does not automatically improve a model. If the training setup is not well tuned, either more training data, more training steps, or a lower learning rate would likely be needed to get the model to genuinely improve instead of degrade.

**6. When would I use RAG instead of fine-tuning?**

I would use RAG when the information changes frequently or is too large to bake into the model's weights, such as a company's latest data or live documents, since RAG retrieves fresh information at query time without retraining. I would use fine-tuning when I want the model to consistently follow a particular style, tone, or narrow skill that should be permanently built into its behavior, rather than injecting or updating factual content. Given what I saw in this experiment, I would also add that fine-tuning needs a properly sized dataset and carefully chosen hyperparameters to actually help, otherwise RAG or prompt engineering can be the safer choice for a small project.

## Training Configuration

```json
{
  "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "lora_config": {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none"
  },
  "training_args": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "learning_rate": 0.0002,
    "max_length": 256
  },
  "results": {
    "trainable_params": 460800,
    "total_params": 134975808,
    "trainable_percent": 0.3414,
    "final_training_loss": 1.978689173857371
  }
}
```
