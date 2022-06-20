import argparse
import pytorch_validator

DEFAULT_TEST_SET_PATH = 'datasets/PrivateTestSet'
DEFAULT_MODEL = 'net_mdl_v2.pth'
DEFAULT_SUBMISSION_FILE = 'output.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NN over test set")

    parser.add_argument('-t', '--testset', default=DEFAULT_TEST_SET_PATH)
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL)
    parser.add_argument('-o', '--output', default=DEFAULT_SUBMISSION_FILE)
    parser.add_argument('-v', '--visualize', action='store_true')

    args = parser.parse_args()
    pytorch_validator.run(test_set_path=args.testset,
                          model_file_name=args.model,
                          visualize=args.visualize,
                          output_file=args.output,
                          test_indexes=None)
