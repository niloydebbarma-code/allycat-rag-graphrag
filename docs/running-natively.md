# Running AllyCat Natively

This guide will show you how to run and develop with AllyCat natively.

**Note:** AllyCat supports both traditional Vector RAG and advanced GraphRAG. By default, both Flask and Chainlit interfaces are available for both modes. See [Configuration Guide](configuration.md) for details.

## Prerequisites: 

- [Python 3.11 Environment](https://www.python.org/downloads/).  You can use [uv](https://docs.astral.sh/uv/) (recommended) or [Anaconda](https://www.anaconda.com/docs/getting-started/getting-started) or [conda-forge](https://conda-forge.org/)
- Use [Ollama](https://ollama.com) if planning to run LLM locally
- Use [Replicate](https://replicate.com) if you want to run your LLM in the cloud

## Step-1: Clone this repo

```bash
git clone https://github.com/The-AI-Alliance/allycat/
cd allycat
```

## Step-2: Setup Python Environment

## 2a (recommended): using `uv`

```bash
uv sync

# create a ipykernel to run notebooks with vscode / jupyter / etc
source  .venv/bin/activate
python -m ipykernel install --user --name=allycat-1 --display-name "allycat-1"
## Choose this kernel 'allycat-1' within jupyter / vscode
```

## 2b: Using python virtual env

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2c: Setup using Anaconda Environment

Install [Anaconda](https://www.anaconda.com/) or [conda forge](https://conda-forge.org/)

And then:

```bash
conda create -n allycat-1  python=3.11
conda activate  allycat-1
pip install -r requirements.txt 
```

## Step-3 (Optional): Create an `.env` file

You can optionally specifify settings for Allycat using `.env` file.

Start with the provided sample env file

```bash
cp   env.sample.txt  .env
```

And make any necessary changes to `.env` file.

## Step-4: LLM setup

We only need option (4A) or (4B)

## Step-4A: Ollama Setup


We will use [ollama](https://ollama.com/) for running open LLMs locally.
This is the default setup.

Follow [setup instructions from Ollama site](https://ollama.com/download)

## Step-4B: Replicate Setup (Optional)

For this step, we will be using Replicate API service.  We need a Replicate API token for this step.

Follow these steps:

- Get a **free** account at [replicate](https://replicate.com/home)
- Use this [invite](https://replicate.com/invites/a8717bfe-2f3d-4a52-88ed-1356231cdf03) to add some credit  💰  to your Replicate account!
- Create an API token on Replicate dashboard

Once you have an API token, add it to the project like this:

- Add your token to `REPLICATE_API_TOKEN` in the .env file (as specified in step-3)

## Step-5: Continue to workflow

Proceed to [run Allycat](running-allycat.md)
