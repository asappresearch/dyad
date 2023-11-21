from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

from transformers.models.opt import OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTConfig
model_name = "facebook/opt-125m"
config = OPTConfig.from_pretrained(model_name)
model = OPTForCausalLM(config=config).to('cuda')

# prompt = "Hey, are you conscious? Can you talk to me?"
# labels = tokenizer(", are you conscious? Can you talk to me? Yes", return_tensors="pt").input_ids.repeat(1024,1).to('cuda')
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.repeat(1024,1).to('cuda')

prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)

prof.start()
for i in range(100):
    labels = torch.randint(0,32000,(16,128),device='cuda')
    input_ids = torch.randint(0,32000,(16,128),device='cuda')
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    loss.item()
    prof.step()


prof.stop()

# with torch.profiler.profile(
#         activities=[
#                 torch.profiler.ProfilerActivity.CPU,
#                 torch.profiler.ProfilerActivity.CUDA,
#         ],
#         profile_memory=True, with_stack=True, with_flops=True
#         ) as p:
#     for i in range(10):
#         labels = torch.randint(0,32000,(16,128),device='cuda')
#         input_ids = torch.randint(0,32000,(16,128),device='cuda')
#         loss = model(input_ids=input_ids, labels=labels).loss
#         loss.backward()
#         loss.item()

# print(p.key_averages().table(
#     sort_by="self_cuda_time_total", row_limit=-1))