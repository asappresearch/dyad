import json

zeros_shots = ["anaphor_agreement","control_raising","filler_gap",
              "island_effects","qa_congruence_tricky","subject_verb_agreement",
              "argument_structure","determiner_noun_agreement","hypernym","npi_licensing",
              "quantifiers","turn_taking","binding","ellipsis","irregular_forms",
              "qa_congruence_easy","subject_aux_inversion"]

zero_shot_data = []
for test in zeros_shots:
    f = open(f'zeroshot/{test}/eval_results.json','r')
    str_data = f.read()
    data = json.loads(str_data)
    data['test'] = test
    zero_shot_data.append(data)

f = open('zero_results.json','w')
f.write(json.dumps(zero_shot_data))
f.close()

fine_tune_tests = [ "boolq","main_verb_control","multirc","syntactic_category_control",
                    "cola","main_verb_lexical_content_the","qnli","syntactic_category_lexical_content_the",
                    "control_raising_control","main_verb_relative_token_position","qqp","syntactic_category_relative_position",
                    "control_raising_lexical_content_the","mnli","relative_position_control","wsc",
                    "control_raising_relative_token_position","mnli-mm","rte",
                    "lexical_content_the_control","mrpc","sst2"]

fine_tune_data = []
for test in fine_tune_tests:
    f = open(f'finetune/{test}/eval_results.json','r')
    str_data = f.read()
    data = json.loads(str_data)
    data['finetune_test'] = test
    fine_tune_data.append(data)

f = open('finetune_results.json','w')
f.write(json.dumps(fine_tune_data))
f.close()