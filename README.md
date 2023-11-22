# dyad

This is the repository for the DYAD long paper at the NEURIPS'23 WANT workshop, by Sarin and Varun [advised by Yi Yang and Gabe Maggiotti]

Here we will be sharing our layer's pytorch code and an example of how to edit a huggingface transformers existing model to use our layer instead of the typical linear layer nn.Linear(), which we refer to in our work as DENSE.

This repo contains sub projects, Sparse Transformers, Sparse BabyLM Pretraining and Struct Baby Sparse.

Struct Baby Sparse contains all the general configuration and profiling scripts for the dyad project.

Sparse Transformers is a fork of hugging face transformers library where the sparse models are added in. The sparse OPT model is under the key sparse_opt and pythis is sparse_gpt_neox. A sample config file for using sparse opt is shown under structbabySparse/sparse_config.json

Sparse BabyLM Pretraining is a fork of repo https://github.com/babylm/baseline-pretraining.To pretrain a model, go inside sparse_babylm_pretraining and execute general_script

Instructions for running are the same except, an additional environment variable SPARSE_CONFIG_FILE with a path to the directory of the config file needs to be added with the sample config file as mentioned above in it.

When installing this library ensure that sparse_transformers is installed instead of the public hugging face transformers.

Additionally for general script an additional argument --sparse needs to be added to train sparse model and --no-sparse to train regular dense model.

For zeroshot and finetune eval, using https://github.com/babylm/evaluation-pipeline and following instructions there is enough. But as always sparse_transformers need to be installed instead of hugging face transformers.

For openLm eval, 
1. Clone https://github.com/EleutherAI/lm-evaluation-harness
2. Create a separate env for this and then source activate or pyenv activate it
cd into the repo root folder
3. do pip install -e .
4. Now, go into sparse_transformers fork and do pip install -e .  This is so our own custom models work. This will override the library transformers version installed in the earlier step.
5. Now come back to inside lm-evaluation-harness; i.e. the repo root folder
6. Place the script runHarnessEval.sh (found at structbabySparse/runHarnessEval.sh) inside the repo root

Example way of running this is:
bash runHarnessEval.sh /home/structBabySparse/sparse_babylm_eval/opt-350m-sparse openLmResults/opt-350m-sparse-strictSmall cuda:0

What are these arguments? The script internally uses positional arguments bash-style

The first argument is the absolute path to the model-tokenizer directory as you refer to it. Ensure no trailing backslashes. More specifically this is the folder containing config, tokenizer and a pytorch_model.bin, the way we use for lm-eval

2. The second argument is the relative path to a  folder under which you hope to have the output jsons for each of the four tasks (+ a task 0 which runs quickly to early-test if things    everything is fine) . This folder will be created on-the-fly if it doesnt exist.

3. The third argument specifies the device. Its optional and will default to cuda:0. cpu also works but takes much longer

What will this script do?

It will first run on big bench temporal sequences which is a small task, to test if things are all fine. Its called task 0 in the script.
It will then run each of the 4 tasks which constitute openLm in succession.
>>>>>>> 451322c (Adding code for dyad)
