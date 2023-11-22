#!/bin/bash
EVALMODELPATH="${1:-/persist/structBabySparse/sparse_babylm_eval/opt-350m-sparse}"
OUTMODELPATH="${2:-openLmResults/opt-350m-sparse}"
#This argument is optional, will default to cuda 0
DEVICE="${3:-cuda:0}"

echo "Ensure the folder name args have no trailing slashes"

echo "The checkpoint-containing-model-folder being used here has the absolute folder path is $EVALMODELPATH, this is based on your first argument"
echo "The result for each task you run will be saved as a json file under $OUTMODELPATH, this is based on your second argument"
echo "The device each task will be run on in sequence is $DEVICE, this is based on the optional third argument. Backs off to cuda:0"


mkdir -p $OUTMODELPATH

# OpenLM Command 0 [Optional to Run, its to test out waters :)]

# Example Command to run Fast and check if Folders and stuff are fine. Will run in around 7 mins on CPU and even faster on GPU
#echo "Running example command 0 to check paths and stuff are fine. This runs on bigbench temporal sequences."
#python main.py --model hf-causal-experimental --model_args pretrained="${EVALMODELPATH}",dtype="float" --tasks bigbench_temporal_sequences --device $DEVICE --num_fewshot=3 --batch_size=2 --output_path="${OUTMODELPATH}/bigbench3.json"



#echo "Now we are running the actual 4 commands on each of the 4 openLm datasets in sequence, arcChallenge (25 shot), hellaswag (10 shot), truthfulqa-mc (0 shot) and finally MMLU (5 shot)"


# OpenLM Command 1

# Example of How fully materialized command looks
#python main.py --model hf-causal-experimental --model_args pretrained=/persist/structBabySparse/sparse_babylm_eval/opt-350m-sparse,dtype="float" --tasks arc_challenge --device $DEVICE --num_fewshot=25 --batch_size=2 --output_path=openLmResults/opt-350m-sparse/arcChallenge25.json
#echo "Running on arcChallenge (25 shot)"
#python main.py --model hf-causal-experimental --model_args pretrained="${EVALMODELPATH}",dtype="float" --tasks arc_challenge --device $DEVICE --num_fewshot=25 --batch_size=2 --output_path="${OUTMODELPATH}/arcChallenge25.json"

#OpenLM Command 2

# Example of How fully materialized command looks
#python main.py --model hf-causal-experimental --model_args pretrained=/persist/structBabySparse/sparse_babylm_eval/opt-350m-sparse,dtype="float" --tasks hellaswag --device $DEVICE --num_fewshot=10 --batch_size=2 --output_path=openLmResults/opt-350m-sparse/hellaswag10.json
#echo "Running on hellaswag (10 shot)"
#python main.py --model hf-causal-experimental --model_args pretrained="${EVALMODELPATH}",dtype="float" --tasks hellaswag --device $DEVICE --num_fewshot=10 --batch_size=2 --output_path="${OUTMODELPATH}/hellaswag10.json"

#Open LM Command 3

# Example of How fully materialized command looks
#python main.py --model hf-causal-experimental --model_args pretrained=/persist/structBabySparse/sparse_babylm_eval/opt-350m-sparse,dtype="float" --tasks truthfulqa-mc --device $DEVICE --num_fewshot=0 --batch_size=2 --output_path=openLmResults/opt-350m-sparse/truthfulqa-mc0.json
echo "Running on truthfulqa-mc (0 shot)"
python main.py --model hf-causal-experimental --model_args pretrained="${EVALMODELPATH}",dtype="float" --tasks truthfulqa_mc --device $DEVICE --num_fewshot=0 --batch_size=2 --output_path="${OUTMODELPATH}/truthfulqa-mc0.json"


#Open LM Command 4

#echo "This command creates an additional Task list variable because MMLU has got many tasks under it"
#export TASKSLIST="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"

# Example of How fully materialized command looks
#python main.py --model hf-causal-experimental --model_args pretrained=/persist/structBabySparse/sparse_babylm_eval/opt-350m-sparse,dtype="float" --tasks $TASKSLIST --device $DEVICE --num_fewshot=5 --batch_size=2 --output_path=openLmResults/opt-350m-sparse/MMLU5.json
#echo "Running on MMLU (5 shot)"
#python main.py --model hf-causal-experimental --model_args pretrained="${EVALMODELPATH}",dtype="float" --tasks $TASKSLIST --device $DEVICE --num_fewshot=5 --batch_size=2 --output_path="${OUTMODELPATH}/mmlu5.json"












