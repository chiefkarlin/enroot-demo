# Instructions for Preparing Container Images

This directory will store the `enroot` filesystem images (`.sqsh` files) used in the demonstration.

## 1. Base PyTorch Container

We will use a standard PyTorch container from the NVIDIA NGC catalog.

**Steps:**

1.  **Import the container using `enroot`:**
    This command will pull the Docker image and convert it into an `enroot` filesystem bundle.

    ```bash
    enroot import 'docker://nvcr.io#nvidia/pytorch:24.09-py3'
    ```

2.  **Rename the Image File:**
    The import command will create a file with a long name. We'll rename it for easier use.

    ```bash
    mv nvcr.io#nvidia#pytorch#24.09-py3.sqsh pytorch.sqsh
    ```

    After this step, you will have `pytorch.sqsh` in this directory. This is the file you will reference in the Slurm job scripts.

## 2. Custom GPT-OSS-20B Container

For Demo 3, we will create a custom container with the `transformers` library installed.

**Steps:**

1.  **Create a writable, temporary container from the base image:**
    This gives us a temporary, modifiable version of our container.

    ```bash
    enroot create --name gpt-oss-20b pytorch.sqsh
    ```

2.  **Start the writable container and install the library:**
    This command drops you into a shell inside the container where you can make changes.

    ```bash
    enroot start --rw gpt-oss-20b
    ```

    Inside the container's shell, run:
    ```bash
    python -m pip install transformers datasets accelerate bitsandbytes scipy sentencepiece huggingface_hub trl peft
    exit
    ```

3.  **Export the modified container to a new `.sqsh` file:**
    This packages our changes into a new, portable image file.

    ```bash
    enroot export --output gpt-oss-20b.sqsh gpt-oss-20b
    ```

    You will now have `gpt-oss-20b.sqsh` in this directory.

4.  **(Optional) Clean up the temporary container:**
    Once you have the `.sqsh` file, you can remove the temporary writable container.

    ```bash
    enroot remove gpt-oss-20b
    ```
