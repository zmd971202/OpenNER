import argparse
from OpenNER import OpenNER



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input",
                        default=None,
                        type=str,
                        required=True,
                        help="The input file.")
    parser.add_argument("--output",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file.")
    parser.add_argument("--bert_model",
                        default='bert-base-cased',
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")
    parser.add_argument("--model_dir",
                        default='OpenNER_base/',
                        type=str,
                        help="The model dir.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    args = parser.parse_args()
    
    # load OpenNER
    tagger = OpenNER(args.bert_model, args.model_dir, args.max_seq_length, args.eval_batch_size, False, args.local_rank, args.no_cuda)
    tagger.predict_file(args.input, args.output) 

if __name__ == "__main__":
    main()
