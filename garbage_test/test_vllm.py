import os
from openai import OpenAI
from openai import APIConnectionError, OpenAIError
import httpx  # 用于更详细的HTTP错误检查
import sys  # 用于退出程序


def test_and_interact_vllm():
    """
    测试 vLLM 服务连接，并在连接成功后进入交互式对话模式。
    """
    api_key = "0"  # vLLM 默认不需要 API Key，或者接受任意值
    base_url = "http://192.168.100.140:8000/v1"

    print(f"尝试连接到 vLLM 服务，Base URL: {base_url}")

    try:
        # 初始化 OpenAI 客户端
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 尝试列出模型以确认连接
        print("尝试调用 client.models.list() 确认连接...")
        models_response = client.models.list()
        print(models_response)

        # 尝试获取一个可用的模型ID，用于后续对话。
        # vLLM通常会返回一个模型ID，比如你加载的模型名称。
        if models_response.data:
            model_id = models_response.data[0].id
            print(f"--- vLLM 连接成功！---")
            print(f"已成功获取模型列表。服务正在运行。将使用模型: {model_id}")
        else:
            print("警告：vLLM 服务似乎已连接，但没有返回任何模型ID。无法进行对话。")
            return False

    except APIConnectionError as e:
        print("\n--- 错误：连接到 vLLM 失败！---")
        print(f"无法连接到 {base_url}。请检查：")
        print("1. vLLM 服务是否已启动并正在运行。")
        print("2. 端口 8000 是否没有被防火墙阻挡。")
        print("3. URL 是否正确 (http://localhost:8000/v1)。")
        print(f"具体错误信息: {e}")
        if isinstance(e.__cause__, httpx.ConnectError):
            print(f"可能的原因: 连接被拒绝或主机不可达。")
        elif isinstance(e.__cause__, httpx.HTTPStatusError):
            print(f"HTTP 状态错误: {e.__cause__.response.status_code} - {e.__cause__.response.text}")
        return False
    except OpenAIError as e:
        print("\n--- 错误：vLLM 响应异常！---")
        print(f"vLLM 服务可能已启动，但返回了非预期的错误。")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print("请检查 vLLM 服务的日志输出，看是否有启动错误或配置问题。")
        return False
    except Exception as e:
        print("\n--- 发生了未知错误！---")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        return False

    # 如果连接成功，进入交互模式
    print("\n--- 进入交互式对话模式 ---")
    print("输入 'exit' 或 'quit' 退出对话。")

    # 存储对话历史，用于多轮对话
    messages = []

    while True:
        try:
            user_input = input("\n你: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            if not user_input:
                continue

            # 将用户输入添加到对话历史
            messages.append({"role": "user", "content": user_input})

            print("模型正在思考...")
            # 调用 vLLM 的聊天完成接口
            completion = client.chat.completions.create(
                model=model_id,  # 使用之前获取到的模型ID
                messages=messages,
                max_tokens=2048,  # 可以根据需要调整最大生成token数
                temperature=0.7,  # 可以调整生成的多样性
            )

            assistant_response = completion.choices[0].message.content
            print(f"模型: {assistant_response}")

            # 将模型的回复也添加到对话历史
            messages.append({"role": "assistant", "content": assistant_response})

        except APIConnectionError as e:
            print(f"\n--- 错误：与 vLLM 的连接中断！---")
            print(f"请检查 vLLM 服务是否仍在运行。具体错误: {e}")
            print("退出对话。")
            break
        except OpenAIError as e:
            print(f"\n--- 错误：vLLM 响应异常！---")
            print(f"在对话过程中发生错误: {e}")
            print("请检查 vLLM 服务日志。")
            # 可以选择继续或退出
            # continue
            print("退出对话。")
            break
        except KeyboardInterrupt:
            print("\n检测到用户中断 (Ctrl+C)。退出对话。")
            break
        except Exception as e:
            print(f"\n--- 发生了未知错误！---")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print("退出对话。")
            break


if __name__ == "__main__":
    test_and_interact_vllm()
