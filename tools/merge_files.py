import os
import glob
import argparse


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", 
                        type=str, 
                        required=True,
                        help="Path of the data directory."
    )
    parser.add_argument("--ext", 
                        type=str,
                        required=True,
                        help="Extension of the files to merge."
    )
    parser.add_argument("--outfile", 
                        type=str, 
                        default='all.dev',
                        help="Output file."
    )
    arguments, _ = parser.parse_known_args()
    return arguments


def main(args):
    """
    """
    filepaths = glob.glob(args.dirpath + '/*.' + args.ext)
    with open(args.dirpath + '/' + args.outfile, 'w') as out:
        for file in filepaths:
            with open(file) as infile:
                for line in infile:
                    out.write(line)
    return
    


if __name__=="__main__":
    args = parse_arguments()
    main(args)
    