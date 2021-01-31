from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import os
import argparse


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", 
                        type=str, 
                        default="belgpt2",
                        help="Path of the model directory."
    )
    arguments, _ = parser.parse_known_args()
    return arguments


def load_model(model_dir=None):
    """
    Loads the saved model from disk if the directory exists.
    Otherwise it will download the model and tokenizer from hugging face.  
    Returns a tuple consisting of `(model,tokenizer)`.
    """    
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return model, tokenizer


def generate(model, tokenizer, input_text=None, num_samples=1, max_length=100, top_k=50, top_p=0.95):
    """
    """
    model.eval()
    
    if input_text:
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output = model.generate(
            input_ids = input_ids,
            do_sample = True,   
            top_k = top_k, 
            max_length = max_length,
            top_p = top_p, 
            num_return_sequences = num_samples
        )
    else:
        output = model.generate(
            bos_token_id = random.randint(1,50000),
            do_sample = True,   
            top_k = 50, 
            max_length = max_length,
            top_p = 0.95, 
            num_return_sequences = num_samples
        )

    decoded_output = []
    for sample in output:
        decoded_output.append(tokenizer.decode(sample, skip_special_tokens=True))

    return decoded_output


def main(args):
    """
    """
    # Ask user to input text.
    input_text = input("Enter text: ")
    
    # Load model and tokenizer.
    model, tokenizer = load_model(args.model_path)
    
    # Generate text.
    decoded_output = generate(model, tokenizer, input_text=input_text)
    
    print(decoded_output)
    return
    


if __name__=="__main__":
    args = parse_arguments()
    main(args)
    