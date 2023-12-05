# FaceFlow: Facial Action Units Estimator

## Purpose 🚀
FaceFlow is a PyTorch Lightning-based repository simplifying the creation of models for detecting facial biomechanics through [Facial Action Units](https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/).<br>
It allows easy storage of configurations as code, enabling reproducibility for each experiment.

## Overview 🏄🏽‍♂️
The regular model can use any backbone, eg [ConvNext](https://arxiv.org/abs/2201.03545).<br>
You can improve domain generalization by adding unsupervised data with [FixMatch](https://arxiv.org/abs/2001.07685) model (`lib/fixmatch.py`), as well as by using [MIRO](https://arxiv.org/abs/2203.10789) training procedure (`lib/miro.py`).<br>
The models train best on images cropped to faces, and code for cropping is in `notebooks/preprocess_disfa.ipynb`.
You can also track your experiments on [Weights And Biases](https://wandb.ai/) - they have a free tier!. The guide on how to do it is coming later, but basically to enable it, run `wandb login` and enter API Key when prompted. You also need to register your data as artifacts.<br>

## How to Run 🏃🏽
1. Prepare data in the format of AU1,AU2,...,filename. In case of DISFA use `notebooks/preprocess_disfa.ipynb`
2. Modify training parameters in the `params` folder. These files there are provided as examples that you can modify to change datasets, architectures, or hyperparameters.
3. To train a model by running `python3 src/train.py`
4. Evaluate the model by running `python3 src/test.py`
5. Export your model to ONNX with `python3 src/export.py` or do some inference with `python3 src/infer.py`

The easiest way to run the repository is by using nvidia-docker as a glorified virtualenv:<br>
`docker build --tag faceflow .` <- run only once<br>
`docker run -it --rm --gpus all -v /path/to/repo:/home -v /path/to/data:/data --shm-size=4gb faceflow`<br>
This way you can edit files locally and immediately use them inside docker.<br>

Configuration:
- datamodule: provide action units you want to train on and location of data
- model: edit backbone model and hyperparameters. You can also use FixMatchModel and MIRO in place of the regular AUModel.
- trainer: edit training schedule: epochs, monitoring, devices

Model Variants:
- Regular AUModel uses AUDataModule (`lib/data/datamodules/vanilla.py`)
- [MIRO](https://arxiv.org/abs/2203.10789) model uses AUDataModule
- [FixMatchModel](https://arxiv.org/abs/2001.07685) and [DeFixMatchModel](https://arxiv.org/abs/2203.07512) use SSLDataModule (`lib/data/datamodules/ssl.py`)


## Data 💾
The model (to be released later) is trained on the DISFA Dataset.<br>
The dataset consists of 27 videos of different people making facial expressions, one video for each person.<br>
You can request the data from its authors [here](http://mohammadmahoor.com/disfa-contact-form/) (only for research purposes).<br>
Labels that go into the model must be in a csv file with columns AU1,AU2,...,filename.<br>
All the necessary preprocessing can be done with `notebooks/preprocess_disfa.ipynb`.

## Some Comments on the Code 💻
- `datamodule` is a [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) that abstracts away datasets and corresponding dataloaders.
- `model` is a [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) wrapper that assembles the model together with its training loop.
  - `backbone` is a timm-compatible [feature extractor](https://huggingface.co/docs/timm/feature_extraction)
  - `heads_partials` are the `functools.partial` instances of `core.AUHead` that encapsulate the task logic, complete with the prediction head, the loss function and the final activation.
Each head needs the backbone output size to be fully instantiated.
  - `optimizer_partial` is a `torch.optim` or a `timm` optimizer. It needs model parameters to get instantiated.
  - `scheduler_partial` is a `timm` scheduler. It needs an optimizer on init.
- `trainer` is a [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) that handles the entire training loop, including checkpoints, logging, accelerators, precision, etc.
- `ckpt_path` refers to a Lightning checkpoint from which the training should be resumed.

## Contribution Guide ✍️
The current `params` folder is given as an example of what's possible to use, eg on Disfa dataset.<br>
If you want to make a pull request, please revert any changes you made to the `params` during the experiments, unless you want to modify the examples.<br>
<details>
<summary>Git command</summary>
<pre><code>git rm -r params
git checkout upstream/main params
git commit -m "reverting params"
</pre></code>
</details>

## Feature Backlog 🦋
- examples on how to use WandB (~Dec '23)
- training examples for using unsupervised data (~Dec '23)
- release model trained on DISFA (~Dec '23)
- train and publish models on other datasets (~Q1'24)

## FAQ ⁉️
<details>
<summary>How to make WandB use AWS credentials from EC2 IAM role</summary>
<pre><code>cmd = 'TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` && curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/iam/security-credentials/AmazonS3FullAccess'
out = subprocess.run(cmd, shell=True, capture_output=True)
creds = json.loads(out.stdout.decode("utf8"))<br>
os.environ["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
os.environ["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
os.environ["AWS_SESSION_TOKEN"] = creds["Token"]
</pre></code>
</details>

## 
Made with 💖+☕ by [TensorSense](https://tensorsense.com/)
