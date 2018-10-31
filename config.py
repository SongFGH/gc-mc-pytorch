import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--mode', type=str, default="train",
                                  help='train / test')
    parser.add_argument('--model-path', type=str, default="./model_")
    parser.add_argument('--data-path', type=str, default="./data")
    parser.add_argument('--data-shuffle', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--val-step', type=int, default=5)
    parser.add_argument('--test-epoch', type=int, default=50)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--neg-cnt', type=int, default=100)
    parser.add_argument('--at-k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.7)

    parser.add_argument('--model', type=str, default='GAE',
                                   help='GAE')

    parser.add_argument('--emb-dim', type=int, default=16)
    parser.add_argument('--hidden', default=[16,8])

    parser.add_argument('--user-cnt', type=int, default=6040)
    parser.add_argument('--item-cnt', type=int, default=3953)
    parser.add_argument('--class-cnt', type=int, default=5)

    parser.add_argument('--train-path', type=str, default='./data/rating_train.pkl')
    parser.add_argument('--val-path', type=str, default='./data/rating_val.pkl')
    parser.add_argument('--test-path', type=str, default='./data/rating_test.pkl')

    args = parser.parse_args()

    return args
