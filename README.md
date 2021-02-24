![](https://img.shields.io/github/license/antoiloui/belgpt2)

# Belgian GPT-2 🇧🇪

**A GPT-2 model pre-trained on a very large and heterogeneous French corpus (~60Gb).**

## Usage

You can use BelGPT-2 with [🤗 Transformers library](https://huggingface.co/antoiloui/belgpt2) as follows:

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pretrained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("antoiloui/belgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("antoiloui/belgpt2")

# Generate a sample of text
model.eval()
output = model.generate(
            bos_token_id=random.randint(1,50000),
            do_sample=True,   
            top_k=50, 
            max_length=100,
            top_p=0.95, 
            num_return_sequences=1
)

# Decode it
decoded_output = []
for sample in output:
    decoded_output.append(tokenizer.decode(sample, skip_special_tokens=True))
print(decoded_output)
```

## Documentation

Detailed documentation on the pre-trained model, its implementation, and the data can be found [here](docs/index.md).

## Citation

For attribution in academic contexts, please cite this work as:

```
@misc{louis2020belgpt2,
  author = {Louis, Antoine},
  title = {{BelGPT-2: a GPT-2 model pre-trained on French corpora.}},
  year = {2020},
  howpublished = {\url{https://github.com/antoiloui/belgpt2}},
}
```
