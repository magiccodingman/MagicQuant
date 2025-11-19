# MagicQuant
Evolution process to find the best quant tensor weights to build the most optimal GGUF options for an AI model.


# Install

The following are the instructions to set up and utilize MagicQuant. 

---

## Step 1: Setup Llama.cpp and Python Environment

There are lots of great guides about setting up `llama.cpp` and I won't rehash what has already been done well. Make sure you set up llama.cpp yourself based on the OS you're utilizing. 

When setting up cmake and ninja, if you're using GPU's (which you likely are), be sure to do things like `-DGGML_CUDA=ON` and `-DGGML_CUDA_FP16=ON` for example or whatever alternatives are for your hardware. But for example, those of us using Nvidia, it's important you enable Cuda and F16 for both speed and future conversions that'll be created.

As for your python environment, those of us on Linux, just set up your virtual python environment accordingly. For those on Windows, use [Mini Conda](https://docs.conda.io/projects/conda/en/stable/index.html) or something.

Just to reiterate, this is not a guide to set up llama.cpp or python. Just follow the million guides that already exist. Or have your friendly AI walk you through the process.

---

## Step 2: Python Environment Prerequisites

Once your python environment is ready, make sure you have `pip` available and run:

```bash
pip install datasets
```

Those should be the things you'll need for this project.

---

# How To Use/Run The Script
