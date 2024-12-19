import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "E:/clone/LLaMA-Factory/models/llama3_lora_sft"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# 假设在微调中，模型习惯 role: user/assistant 的格式。
# 若您在微调时有特定的开场词、system指令等，可在此添加。例如：
# conv = "<s>[INST] <<SYS>>系统指令<</SYS>>\n" + ...
def format_conversation(history):
    # history: [{"role":"user"/"assistant", "content":"..."}]
    # 我们将其转换为：
    # user: ...
    # assistant: ...
    conv = ""
    for turn in history:
        if turn['role'] == 'user':
            conv += f"User: {turn['content']}\n"
        else:
            conv += f"Assistant: {turn['content']}\n"
    # 等待assistant回答
    conv += "Assistant:"
    return conv

def user_submit(user_message, history):
    # 用户点击发送后将用户消息加入history
    if history is None:
        history = []
    history.append({"role": "user", "content": user_message})
    return history, ""

def bot_respond(history):
    # 当用户消息已加入history后，生成模型回复
    conversation = format_conversation(history)
    inputs = tokenizer(conversation, return_tensors='pt').to(model.device)

    with torch.no_grad():
        # 使用无采样模式，降低temperature减少随机性
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=0.1)
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # 将模型回答加入history
    history.append({"role": "assistant", "content": answer.strip()})
    return history

with gr.Blocks() as demo:
    gr.Markdown("# 模型对话界面")
    chatbot = gr.Chatbot(label="对话区域", type="messages")
    msg = gr.Textbox(placeholder="请输入你的问题...")
    clear = gr.Button("清空对话")

    # 当用户submit时，先将消息加入history，再由then调用bot_respond生成回答
    msg.submit(user_submit, [msg, chatbot], [chatbot, msg], queue=False).then(bot_respond, chatbot, chatbot)
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
