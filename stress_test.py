###
# @Desc《基于首Token和平均Token响应时间的大模型负载测试方法研究》论文的相关代码,通过设置不通得并发数后返回首token时间、token数和平均token返回时间
# @Time 2025-03-11
# @Author 李鲲程@中信息通信研究院 信管中心
# @Version 1.0
# @Contact likuncheng@caict.ac.cn
###

import concurrent.futures
import json
import requests
import time
import pandas as pd
import random

from stress_test_data import test_data_list as data_list


api_key = "sk-sqofGCIMh5ZYMSpQ4d7602160f1342C6B4Ad744dCb8664Ca"

headers = {
    "content-type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "Cache-Control": "no-store, max-age=0"
}
test_equipment = "six_cards"
model_list = {
    "two_cards": "Qwen/Qwen2-72B-Instruct-GPTQ-Int4",  # 两块卡
    "six_cards": "Qwen2-72B-Instruct-GPTQ-Int4",  # 六块卡
}
base_url_list = {
    "two_cards": "http://10.8.71.111:8000/v1/chat/completions", # 两块卡
    "six_cards": "http://10.26.63.21:8011/v1/chat/completions",   # 六块卡
}
base_url = ""
model = ""
if test_equipment in base_url_list:
    base_url = base_url_list[test_equipment]
    model = model_list[test_equipment]
else:
    print("Test Method Error!")
    exit(0)

# 并发数
concurrent_cnt = 5
# 压力测试方法有两种，一种是同时发起多个线程，另一种是因为大模型是流式输出，在输出过程中都会占用资源，所以测试一分钟内相等时间间隔发起多个线程
in_minute = False

results = []

# 根据字符串表生成json格式的请求数据
def get_json_data(data_str):
    json_data = {
        "model": model,
        "messages": [
            {
                "content": "你是中国信通院AI办事助手，一个由中国信息通信研究院泰尔终端实验室、信息管理中心和数据研究中心联合打造的人工智能助手",
                "role": "system"
            },
            {
                "content": data_str,
                "role": "user"
            },
        ],
        "stream": True,
        "max_token": 8000,
        "temperature": 0.2,  # 设置一个非零值
        "top_p": 0.9,
        "seed": random.randint(10000,99999)
    }
    return json_data

# 分析post返回应答的数据
def chunk_str_detect(chunk_str):
    if chunk_str == '':
        return {"token_value": None, "error": "empty string"}
    if chunk_str.startswith("data:"):
        chunk_data_str = chunk_str[5:].strip()
        if chunk_data_str:  # 确保有数据
            # 解析数据
            chunk_data = json.loads(chunk_data_str)
            # 查看流中的每个块，通常模型返回的格式中会有 `choices` 字段
            if "role" in chunk_data['choices'][0]['delta']:
                return {"token_value": None, "error": "assistant"}
            if "choices" in chunk_data:
                # 获取流中的第一个 token
                if "content" in chunk_data['choices'][0]['delta']:
                    token_value = chunk_data['choices'][0]['delta']['content']  # 获取生成的文本（token）
                    return {"token_value": token_value, "error": None}
                else:
                    return {"token_value": None, "error": "No content"}
            else:
                return {"token_value": None, "error": "No choices field"}
        else:
            return {"token_value": None, "error": "strip chunk_str error"}
    else:
        chunk_data = json.loads(chunk_str)
        # 查看流中的每个块，通常模型返回的格式中会有 `choices` 字段
        if "choices" in chunk_data:
            # 获取流中的第一个 token
            first_token = chunk_data['choices'][0]['delta']['content']  # 获取生成的文本（token）
            return {"token_value": first_token, "error": None}
        else:
            return {"token_value": None, "error": "No choices field"}

# 调用一次大模型接口，处理返回数据
def requests_call_llm(content_str, thread_id):
    # 如果是测试一分钟内相等时间间隔的并发数
    time_split = 60 / concurrent_cnt    # 时间间隔
    if in_minute:
        time.sleep(time_split * thread_id)  #根据第几个线程计算睡眠时间
    # 生成向大模型推送的数据
    json_data = get_json_data(content_str)
    # 线程启动时间
    start = time.time()
    try:
        # 向大模型post请求
        with requests.post(base_url, data=json.dumps(json_data), headers=headers, stream=True) as response:
            if response.status_code == 200:
                first_token_time = 0    # 首token时间
                token_cnt = 0   # token个数
                answer_len = 0  #中文字数
                for chunk in response.iter_lines():
                    # 逐个token读取
                    token_cnt = token_cnt + 1
                    if first_token_time == 0:
                        first_token_time = time.time() - start
                    # 解码每一块数据，并解析 JSON
                    chunk_str = str(chunk.decode("utf-8"))
                    if chunk_str == 'data: [DONE]':
                        # 最后一个结束token
                        finish_time = time.time() - start
                        return_dict = {"thread_id":thread_id, "start_time": int(start), "first_token_time": first_token_time,
                                "finish_time": finish_time, "token_count": token_cnt, "ms/token": (finish_time-first_token_time)*1000/token_cnt}
                        # print(return_dict)
                        return return_dict
            else:
                if response.status_code == 429:
                    return {"thread_id":thread_id, "error": "超负载"}
                else:
                    return {"thread_id":thread_id, "error": response.status_code}
    except Exception as e:
        return {"thread_id":thread_id, "error": str(e)}


# 根据并发数并行发起线程，如果测试的是一分钟内平均时间间隔的并发，是在线程内向大模型发送请求前sleep
def run_concurrent_requests():
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_cnt) as executor:
        futures = [executor.submit(requests_call_llm, data_list[i % len(data_list)], i) for i in range(concurrent_cnt)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())


start_time = time.time()
run_concurrent_requests()
end_time = time.time()
total_time = end_time - start_time

pf = pd.DataFrame(results)

pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.width", None)  # 自动调整宽度以适应屏幕
pd.set_option("display.max_colwidth", None)  # 显示完整的列内容

print(pf.to_string(index=False))
