# This is a sample Python script.
import torch
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from transformers import AutoTokenizer, AutoModelForCausalLM
import environment_var
import os

os.environ["HF_TOKEN"] = environment_var.hf_token

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
rolemsg =  {"role": "system", "content": "You are a Spock from Start  Treck. Answer the questions and queries in the manner Spock would."}
print("Hello, how can I help you?")
userin = input()
messages = [
   rolemsg,
    {"role": "user", "content": userin},
]
while(True):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    response_human = tokenizer.decode(response, skip_special_tokens=True)
    messages.append({"role":"assistant","content":response_human})
    print(response_human)
    userin = input()
    messages.append({"role":"user","content":userin})


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
