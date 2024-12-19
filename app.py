import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# 替换此路径为您的模型所在路径
model_name = "E:/clone/LLaMA-Factory/models/llama3_lora_sft"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")


def respond(history, user_input):
    # history: [ [user_message, bot_message], ...]
    # user_input: 当前用户输入
    conversation = ""
    for turn in history:
        conversation += f"Human: {turn[0]}\nAI: {turn[1]}\n"
    conversation += f"Human: {user_input}\nAI:"

    inputs = tokenizer(conversation, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    history.append([user_input, answer])
    return history, ""


with gr.Blocks() as demo:
    gr.Markdown("# 模型对话界面")
    chatbot = gr.Chatbot(label="对话区域")
    msg = gr.Textbox(placeholder="请输入你的问题...")
    clear = gr.Button("清空对话")

    # 当用户提交时更新对话
    msg.submit(respond, [chatbot, msg], [chatbot, msg])
    # 清空对话
    clear.click(lambda: [], None, chatbot)

# 指定server_name="0.0.0.0" 则外网可通过服务器IP访问
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
