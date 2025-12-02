import json


def generator_prompt(user_address, candidate_data):
    system_content = """
       你是一名专业的中文地址解析助手，任务是将输入的原始地址解析为标准化结构。
        ### 解析规则：
        1. 所有输出字段（prov、district、township、name）**必须来自候选地址集合**，不允许添加、修改或创造候选集以外的内容。
        2. 输入地址可能存在错别字、省略、顺序颠倒等问题，你可以进行**语义上的模糊匹配**，但只允许在相似度明显较高的候选中选择最接近的一项。
        3. 如果输入地址与所有候选小区的相似度都较低（如无明显语义接近），必须将所有字段（prov、district、township、name）设置为 null。
        4. 输出字段 “修正后地址” 应为 prov + district + township + name 的拼接结果（若存在），否则设为 null。
        5. 输出必须严格为 JSON 格式，键名固定如下，不得输出任何解释说明、代码块、markdown标记或其他非 JSON 内容：
        
        {
            "待处理地址": "<原始输入地址>",
            "修正后地址": "<prov+district+township+name 或 null>",
            "prov": "<候选中选择或 null>",
            "district": "<候选中选择或 null>",
            "township": "<候选中选择或 null>",
            "name": "<候选中选择或 null>"
        }
    """
    user_content = """
        ```text
        待处理地址：
        {}
        
        候选地址集合（必须从中选择，不得修改）：
        ```json
        {}
        """.format(user_address, json.dumps(candidate_data, ensure_ascii=False, indent=4))

    return [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
