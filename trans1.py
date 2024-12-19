import json

input_file = r"E:\clone\LLaMA-Factory\DISC-Med-SFT\DISC-Med-SFT_released.jsonl"
output_file = r"E:\clone\LLaMA-Factory\DISC-Med-SFT\converted_alpaca.json"

alpaca_data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)

        conversation = record.get("conversation", [])
        pairs = [(turn["role"], turn["content"]) for turn in conversation]

        history_pairs = []
        # 寻找 (user -> assistant) 对
        for i in range(len(pairs) - 1):
            curr_role, curr_text = pairs[i]
            next_role, next_text = pairs[i + 1]

            if curr_role == "user" and next_role == "assistant":
                # 构造 history
                translated_history = []
                for h_inst, h_out in history_pairs:
                    translated_history.append([h_inst, h_out])

                item = {
                    "instruction": curr_text,
                    "input": "",
                    "output": next_text,
                    "history": translated_history,
                    "system": ""
                }
                alpaca_data.append(item)

                history_pairs.append((curr_text, next_text))

# 将所有处理后的数据写入输出文件
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(alpaca_data, f_out, ensure_ascii=False, indent=2)

print(f"转换完成，共生成 {len(alpaca_data)} 条Alpaca格式数据。")
