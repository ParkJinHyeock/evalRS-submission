"""

    Template script for the submission. You can use this as a starting point for your code: you can
    copy this script as is into your repository, and then modify the associated Model class to include
    your logic, instead of the random baseline. Most of this script should be left unchanged for your submission
    as we should be able to run your code to confirm your scores.

    Please make sure you read and understand the competition rules and guidelines before you start.

"""

import os
from datetime import datetime
from dotenv import load_dotenv
import argparse 
from torch.multiprocessing import Process
import torch.distributed as dist
import torch
import pandas as pd
# import env variables from file
load_dotenv('upload.env', verbose=True)

# variables for the submission
EMAIL = os.getenv('EMAIL')  # the e-mail you used to sign up
assert EMAIL != '' and EMAIL is not None
BUCKET_NAME = os.getenv('BUCKET_NAME')  # you received it in your e-mail
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')  # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')  # you received it in your e-mail
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')  # you received it in your e-mail
CUDA_LAUNCH_BLOCKING="1"

# run the evaluation loop when the script is called directly
if __name__ == '__main__':
    # import the basic classes
    from evaluation.EvalRSRunner import EvalRSRunner
    from evaluation.EvalRSRunner import ChallengeDataset
    from submission.MyModel import MyModel
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', default=False, action='store_true')
    parser.add_argument('--topk', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--decay', default=0, type=float)
    parser.add_argument('--dim', default=200, type=int)
    parser.add_argument('--model_type', default='BPR', type=str)
    parser.add_argument('--lambda_pop', default=1, type=float)
    parser.add_argument('--AE_user', default=False, action='store_true')
    parser.add_argument('--use_ensemble', default=False, action='store_true')
    parser.add_argument('--gpu', default='-1', type=str)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--gamma', default=False, type=float, help="weight for fairness regularizer")
    parser.add_argument('--use_group', default=False, action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    print('\n\n==== Starting evaluation script at: {} ====\n'.format(datetime.utcnow()))
    # load the dataset
    print('\n\n==== Loading dataset at: {} ====\n'.format(datetime.utcnow()))
    # this will load the dataset with the default values for the challenge
    dataset = ChallengeDataset()
    print('\n\n==== Init runner at: {} ====\n'.format(datetime.utcnow()))
    # run the evaluation loop
    runner = EvalRSRunner(
        dataset=dataset,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        participant_id=PARTICIPANT_ID,
        bucket_name=BUCKET_NAME,
        email=EMAIL
        )
    print('==== Runner loaded, starting loop at: {} ====\n'.format(datetime.utcnow()))
    # NOTE: this evaluation will run with default values for the parameters and the upload flag
    # For local testing and iteration, you can check the tutorial in the notebooks folder and the
    # kaggle notebook: https://www.kaggle.com/code/vinidd/cikm-data-challenge
    my_model = MyModel(
        items=dataset.df_tracks,
        users=dataset.df_users,
        top_k=args.topk,
        # kwargs may contain additional arguments in case, for example, you 
        # have data augmentation functions that you wish to use in combination
        # with the dataset provided by the runner.
        my_custom_argument='my_custom_argument',
        args = args
    )
    # run evaluation with your model
    # the evaluation loop will magically perform the fold splitting, training / testing
    # and then submit the results to the leaderboard
    runner.evaluate(
        model=my_model,
        upload=args.submission
        )
    print('\n\n==== Evaluation ended at: {} ===='.format(datetime.utcnow()))
