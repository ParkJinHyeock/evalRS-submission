# EvalRS-CIKM-2022
This is a repository of team ML for the EvalRS Data Challenge.
We will update the readme and instructions as soon as possible.

## Getting Started
- Environment: AWS Deep Learning AMI GPU PyTorch 1.12.1 (Amazon Linux 2) with `p3.2xlarge` instance.
- upload upload.env

1. Activate pre-built pytorch environment
    ```
    source activate pytorch
    ```

2. Install all dependencies.
    ```
    pip install -r requirements.txt
    ```

3. Run the following command to run our submission  
    ```
    python submission.py --gpu 0 --model_type VAE --lr 1e-3 --epoch 10 --beta 0.0001 --use_group --use_ensemble --gamma 0.003
    ```
4. If you want to see best accuracy model reported on paper run

    ```
    python submission.py --gpu 0 model_type VAE --lr 1e-3 --epoch --dim 500
    ```
    

