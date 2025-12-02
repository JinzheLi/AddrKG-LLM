from SemanticConstrained_Retrieval.similar_topk import AddressRetriever
from SemanticConstrained_Retrieval.constrained_prompt import generator_prompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel, BitsAndBytesConfig

import json
import argparse
import re
import time

import csv


def load_llm(model_path):
    print("✅ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # LLaMA / Qwen，可开启 rope_scaling
    # config = AutoConfig.from_pretrained(model_path)
    # config.rope_scaling = {"name": "linear", "factor": 2.0}

    print("✅ Loading model across 4 GPUs with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "11GiB", 1: "11GiB", 2: "11GiB", 3: "11GiB"}
    )

    return tokenizer, model


def LLM_inference(tokenizer, model, query, candidates, max_new_tokens):
    # 构造 Prompt

    # for item in candidates:
    #     if isinstance(item, dict):
    #         item.pop("score", None)

    messages = generator_prompt(query, candidates)
    # 转换输入
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device)
    input_len = input_tensor.shape[1]

    # 动态控制生成长度，防止超上下文
    model_max_len = getattr(model.config, "max_position_embeddings", 10240)
    allowed_new_tokens = min(max_new_tokens, model_max_len - input_len)
    if allowed_new_tokens <= 0:
        allowed_new_tokens = 1
    print(f"✅ 输入长度: {input_len}, 允许生成: {allowed_new_tokens} tokens")

    # 推理
    with torch.inference_mode():
        outputs = model.generate(
            input_tensor,
            max_new_tokens=allowed_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    result = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return result


def clean_llm_output(text):
    # 去掉 <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)

    # 提取第一个 { ... }
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        return None
    json_str = match.group(0)

    # 将 "null" 替换为 null，避免错误解析
    json_str = re.sub(r'"\s*null\s*"', 'null', json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # 补全字段
    required_keys = ["待处理地址", "prov", "district", "township", "name"]
    for key in required_keys:
        if key not in data:
            data[key] = None

    return data


def enforce_candidate_rule(data, candidates):
    candidate_names = [item.get("name") for item in candidates if isinstance(item, dict) and "name" in item]
    if data.get("name") not in candidate_names:
        data["prov"] = None
        data["district"] = None
        data["township"] = None
        data["name"] = None
    return data


# 4. 批量处理地址列表
def process_queries(input_file, retriever, tokenizer, model, top_k, max_new_tokens, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        queries = [row["NAME"].strip() for row in csv.DictReader(f) if row["NAME"].strip()]

    jsonl_file = output_file + ".jsonl"

    open(jsonl_file, "w", encoding="utf-8").close()

    # ====== 新增计时器 ======
    total_start = time.time()
    batch_start = time.time()
    total_n = len(queries)
    # ======================

    for idx, query in enumerate(queries, start=1):
        candidates = retriever.search(query, top_k=top_k)
        print('✅ 候选地址集合', candidates)
        llm_result = LLM_inference(tokenizer, model, query, candidates, max_new_tokens)

        data = clean_llm_output(llm_result)
        if data:
            data = enforce_candidate_rule(data, candidates)
        else:
            data = {"待处理地址": query, "prov": None, "district": None, "township": None, "name": None}

        print(data)
        with open(jsonl_file, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(data, ensure_ascii=False) + "\n")

        # ====== 每 100 条统计一次耗时（秒） ======
        if (idx % 100 == 0) or (idx == total_n):
            batch_elapsed = time.time() - batch_start
            total_elapsed = time.time() - total_start
            avg_sec_per_item = total_elapsed / idx
            print(f"⏱️ 已处理 {idx}/{total_n} 条 | 最近{min(100, idx % 100 or 100)}条用时 {batch_elapsed:.2f}s | "
                  f"总用时 {total_elapsed:.2f}s | 平均 {avg_sec_per_item:.2f}s/条")
            batch_start = time.time()
        # ======================================
        if idx == 1000:
            break

    items = []
    with open(jsonl_file, "r", encoding="utf-8") as jf:
        for line in jf:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"✅ 结果已保存到 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_emb", type=str, default="save/graph_emb", help="知识图谱存储路径")
    parser.add_argument("--sbert", type=str, default="model/text2vec_base_chinese_paraphrase", help="sentence bert模型路径")
    parser.add_argument("--top_k", type=int, default=5, help="Top K 个数")

    parser.add_argument("--llm_path", type=str, default="/root/LLM_model/deepseek_R1_Distill_Qwen_32B", help="LLM模型路径")
    parser.add_argument("--input_file", type=str, default="queries.txt", help="地址输入文件，每行一个地址")
    parser.add_argument("--output_file", type=str, default="llm_results.json", help="结果输出JSON文件")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="生成的最大token数")
    args = parser.parse_args()

    # 1. 加载检索器
    retriever = AddressRetriever(args)

    # 2. 加载 LLM
    tokenizer, model = load_llm(args.llm_path)

    # 3. 批量处理地址
    process_queries(args.input_file, retriever, tokenizer, model, args.top_k, args.max_new_tokens,
                    args.output_file)

"""
python main.py --top_k 80 --graph_emb /root/ljzhe/ex_add_llm/save/graph_emb/2025-08-07_15-28-02 --input_file data/INPATIENT_SOURCE_TOPIC.csv --output_file data/llm_results.json 
"""
