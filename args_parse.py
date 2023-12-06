import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default='zinc',
        help="Name of the dataset. Available: qm9, zinc, community, ego, enzymes"
    )

    parser.add_argument(
        "--work_type", type=str,
        default='train_autoencoder', help="Options: train_autoencoder, train_prior, sample"
    )

    parser.add_argument(
        "--model_folder", type=str,
        default='run-20230820_000929-j0b7ijmd',
        help="Name of the Weight and bias folder with the saved model."
             "Enter the autoencoder model to train the prior"
             "Entre the prior model to sample"
            '''
            My prior model to sample:
            zinc: run-20230812_115952-4cyq00dq
            qm9: 'run-20230818_224344-b2f1933r'
            
            prevv: run-20230628_080858-l5qbxy9h
            run-20230628_182408-xmis2vzd
            '''
    )
    return parser.parse_args()
