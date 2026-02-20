


class FakeMemory:

    taglish_original_history = [
        {
            "role": "user",
            "content": "Hi, tanong lang, mabilis ba talaga to? Kasi yung luma naming chatbot sobrang bagal.",
            "metadata": {"timestamp": "2023-10-27 10:00:00", "user_id": "pinoy_dev_01"}
        },
        {
            "role": "assistant",
            "content": "Opo! Miako handles thousands of users with sub-second latency kasi naka-Groq LPU siya. Hindi siya n8n na mabagal.",
            "metadata": {"timestamp": "2023-10-27 10:00:05", "model": "miako-v1"}
        },
        {
            "role": "user",
            "content": "Weh? Di nga? Pano pag sabay-sabay gumamit? Edi lag yan panigurado.",
            "metadata": {"timestamp": "2023-10-27 10:00:15", "user_id": "pinoy_dev_01"}
        },
        {
            "role": "assistant",
            "content": "Hindi po. Designed siya for high concurrency using Async FastAPI. May isolation din per user kaya walang lag.",
            "metadata": {"timestamp": "2023-10-27 10:00:20", "model": "miako-v1"}
        },
        {
            "role": "user",
            "content": "Ah ganun ba. Eh pano kung nagonline lahat, di ba mahal yun? Yung budget kasi namin limited lang.",
            "metadata": {"timestamp": "2023-10-27 10:00:35", "user_id": "pinoy_dev_01"}
        }
    ]



    # FOR THE TRANSLATED MEMORY MOCK (What the LLM 'sees' as English)
    taglish_translated_history = [
        {
            "role": "user",
            "content": "Hi, just asking, is this really fast? Because our old chatbot was very slow.",
            "metadata": {"timestamp": "2023-10-27 10:00:00"}
        },
        {
            "role": "assistant",
            "content": "Yes! Miako handles thousands of users with sub-second latency because it uses Groq LPU. It is not like n8n which is slow.",
            "metadata": {"timestamp": "2023-10-27 10:00:05"}
        },
        {
            "role": "user",
            "content": "Really? No way? What if everyone uses it at the same time? That would surely lag.",
            "metadata": {"timestamp": "2023-10-27 10:00:15"}
        },
        {
            "role": "assistant",
            "content": "No sir. It is designed for high concurrency using Async FastAPI. There is also isolation per user so there is no lag.",
            "metadata": {"timestamp": "2023-10-27 10:00:20"}
        },
        {
            "role": "user",
            "content": "Ah is that so. But what if everyone goes online, isn't that expensive? Our budget is limited.",
            "metadata": {"timestamp": "2023-10-27 10:00:35"}
        }
    ]


    taglish_user_input = "Ah is that so. But what if everyone goes online, isn't that expensive? Our budget is limited."


fake_memory = FakeMemory()