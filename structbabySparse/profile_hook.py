from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# from transformers.models.opt import OPTForCausalLM
# from transformers.models.opt.modeling_opt import OPTConfig
# # model_name = "facebook/opt-125m"
# # model_name = "facebook/opt-350m"
# model_name = "/home/schandy/sparse_config/dense-opt-1.3b"
# config = OPTConfig.from_pretrained(model_name)
# model = OPTForCausalLM(config=config).to('cuda')

from transformers.models.sparse_opt import SparseOPTForCausalLM
from transformers.models.sparse_opt.modeling_opt import SparseOPTConfig
# model_name = "/home/schandy/sparse_config/opt-125m"
# model_name = "/home/schandy/sparse_config/opt-350m"
model_name = "/home/schandy/sparse_config/opt-1.3b"
config = SparseOPTConfig.from_pretrained(model_name)
model = SparseOPTForCausalLM(config=config).to('cuda')

# from transformers import GPTNeoXForCausalLM
# from transformers import GPTNeoXConfig
# from transformers import SparseGPTNeoXConfig
# from transformers import SparseGPTNeoXForCausalLM
# model_name = "EleutherAI/pythia-160m"
# config = GPTNeoXConfig.from_pretrained(model_name)
# model = GPTNeoXForCausalLM(config=config).to('cuda')
# model_name = "/home/schandy/sparse_config/pythia-160m"
# config = SparseGPTNeoXConfig.from_pretrained(model_name)
# model = SparseGPTNeoXForCausalLM(config=config).to('cuda')

# Calc num params
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# pytorch_wo_embd_lm_head = pytorch_total_params - sum(p.numel() for p in model.lm_head.parameters())
# print(f'pytorch_total_params - {pytorch_total_params}, pytorch_wo_embd_lm_head - {pytorch_wo_embd_lm_head}')

import time
## Define hook functions
take_time_dict = {}
results_dict = {}

def take_time_pre(layer_name,stage,module, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    take_time_dict[layer_name] = (start,end)

def take_time(layer_name,stage,module, input, output):
    start,end = take_time_dict[layer_name]
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end)
    avg_time = results_dict[layer_name][stage]['avg_time']
    num_obs = results_dict[layer_name][stage]['num_obs']
    new_avg_time = (avg_time*num_obs + time)/(num_obs+1)
    results_dict[layer_name][stage]['avg_time'] = new_avg_time
    results_dict[layer_name][stage]['num_obs'] = num_obs+1
    # print(f'time during {stage} for {layer} is {take_time_dict[layer_name]}')
    ## for TensorBoard you should use writter

from functools import partial

for index, layer in enumerate(model.model.decoder.layers):
    layer_name = f'layer_{index}_fc1'
    results_dict[layer_name] = {}
    results_dict[layer_name]['forward'] = {'avg_time':0,'num_obs':0}
    results_dict[layer_name]['backward'] = {'avg_time':0,'num_obs':0}
    layer.fc1.register_forward_pre_hook( partial(take_time_pre, layer_name, 'forward') )
    layer.fc1.register_forward_hook( partial(take_time, layer_name, 'forward') )
    layer.fc1.register_full_backward_pre_hook( partial(take_time_pre, layer_name, 'backward') )
    layer.fc1.register_full_backward_hook( partial(take_time, layer_name, 'backward') )
    layer_name = f'layer_{index}_fc2'
    results_dict[layer_name] = {}
    results_dict[layer_name]['forward'] = {'avg_time':0,'num_obs':0}
    results_dict[layer_name]['backward'] = {'avg_time':0,'num_obs':0}
    layer.fc2.register_forward_pre_hook( partial(take_time_pre, layer_name, 'forward') )
    layer.fc2.register_forward_hook( partial(take_time, layer_name, 'forward') )
    layer.fc2.register_full_backward_pre_hook( partial(take_time_pre, layer_name, 'backward') )
    layer.fc2.register_full_backward_hook( partial(take_time, layer_name, 'backward') )

# for index, layer in enumerate(model.gpt_neox.layers):
#     layer_name = f'layer_{index}_fc1'
#     results_dict[layer_name] = {}
#     results_dict[layer_name]['forward'] = {'avg_time':0,'num_obs':0}
#     results_dict[layer_name]['backward'] = {'avg_time':0,'num_obs':0}
#     layer.mlp.dense_h_to_4h.register_forward_pre_hook( partial(take_time_pre, layer_name, 'forward') )
#     layer.mlp.dense_h_to_4h.register_forward_hook( partial(take_time, layer_name, 'forward') )
#     layer.mlp.dense_h_to_4h.register_full_backward_pre_hook( partial(take_time_pre, layer_name, 'backward') )
#     layer.mlp.dense_h_to_4h.register_full_backward_hook( partial(take_time, layer_name, 'backward') )
#     layer_name = f'layer_{index}_fc2'
#     results_dict[layer_name] = {}
#     results_dict[layer_name]['forward'] = {'avg_time':0,'num_obs':0}
#     results_dict[layer_name]['backward'] = {'avg_time':0,'num_obs':0}
#     layer.mlp.dense_4h_to_h.register_forward_pre_hook( partial(take_time_pre, layer_name, 'forward') )
#     layer.mlp.dense_4h_to_h.register_forward_hook( partial(take_time, layer_name, 'forward') )
#     layer.mlp.dense_4h_to_h.register_full_backward_pre_hook( partial(take_time_pre, layer_name, 'backward') )
#     layer.mlp.dense_4h_to_h.register_full_backward_hook( partial(take_time, layer_name, 'backward') )

for i in range(100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    labels = torch.randint(0,32000,(32,128),device='cuda')
    input_ids = torch.randint(0,32000,(32,128),device='cuda')
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    loss.item()
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end)
    print(f'Batch time is {t}')


print(results_dict)
import numpy as np
forward_avg = np.mean([results_dict[k]['forward']['avg_time'] for k in results_dict.keys()])
backward_avg = np.mean([results_dict[k]['backward']['avg_time'] for k in results_dict.keys()])
total_avg = forward_avg + backward_avg

print(f'total avg: {total_avg}, fwd_avg: {forward_avg}, bwd_avg: {backward_avg}')