import os
import glob
import argparse
from tokenizers import (ByteLevelBPETokenizer,
                        CharBPETokenizer,
                        SentencePieceBPETokenizer,
                        BertWordPieceTokenizer)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files",
                        default=None,
                        metavar="path",
                        type=str,
                        required=True,
                        help="The files to use as training; accept '**/*.txt' type of patterns if enclosed in quotes.",
    )
    parser.add_argument("--method",
                        default='byte',
                        type=str,
                        choices=['byte', 'char', 'spm', 'wpm'],
                        help="The algorithm used to learn BPE.",
    )
    parser.add_argument("--vocab_size",
                        default=30000,
                        type=int,
                        help="Size of the vocabulary to learn.",
    )
    parser.add_argument("--outdir",
                        default="./",
                        type=str,
                        help="Path to the output directory, where the files will be saved.",
    )
    arguments, _ = parser.parse_known_args()
    return arguments



def main(args):
    """
    """
    files = glob.glob(args.files)
    if not files:
        print(f"File does not exist: {args.files}")
        exit(1)
        
    if args.method == 'byte':
        # Represents a Byte-level BPE algorithm, as introduced by OpenAI with their GPT-2 model.
        print("Initializing empty Byte-Level BPE tokenizer...")
        tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
        
        print("Training tokenizer...")
        tokenizer.train(files,
                        vocab_size=args.vocab_size,
                        min_frequency=2,
                        show_progress=True,
                        special_tokens=["<s>", "<pad>", "</s>"])
    
    elif args.method == 'char':
        # Represents the original BPE algorithm, as introduced by Rico Sennrich (https://arxiv.org/abs/1508.07909).
        print("Initializing empty Char-Level BPE tokenizer...")
        tokenizer = CharBPETokenizer()
        
        print("Training tokenizer...")
        tokenizer.train(files,
                        vocab_size=args.vocab_size,
                        min_frequency=2,
                        special_tokens=["<unk>"],
                        limit_alphabet=1000,
                        initial_alphabet=[],
                        suffix="</w>",
                        show_progress=True)
        
    elif args.method == 'spm':
        # Represents the BPE algorithm, with the pretokenization used by SentencePiece.
        print("Initializing empty SentencePiece tokenizer...")
        tokenizer = SentencePieceBPETokenizer()
        
        print("Training tokenizer...")
        tokenizer.train(files,
                        vocab_size=args.vocab_size, 
                        min_frequency=2,
                        special_tokens=["<unk>"],
                        limit_alphabet=1000,
                        initial_alphabet=[],
                        show_progress=True)
        
        
    elif args.method == 'wpm':
        # Represents a WordPiece BPE algorithm, as introduced with BERT model.
        print("Initializing empty WordPiece tokenizer...")
        tokenizer = BertWordPieceTokenizer(clean_text=False, 
                                           handle_chinese_chars=False, 
                                           strip_accents=False, 
                                           lowercase=False)

        print("Training tokenizer...")
        trainer = tokenizer.train(files,
                                  vocab_size=args.vocab_size,
                                  min_frequency=2,
                                  limit_alphabet=1000,
                                  initial_alphabet=[],
                                  special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                                  show_progress=True,
                                  wordpieces_prefix="##")
        
    else:
        print(f"Algorithm does not exist: {args.method}")
        exit(1)
    
    print("Saving tokenizer...")
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    tokenizer.save(args.outdir, args.method)
    print("Done.")


if __name__=="__main__":
    args = parse_arguments()
    main(args)