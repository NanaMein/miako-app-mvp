


class FakeMemory:

    taglish_original_history = [
        {
            "role": "user",
            "content": "Hi, tanong lang, mabilis ba talaga to? Kasi yung luma naming chatbot sobrang bagal.",
            "metadata": {"created_at": "2023-10-27 10:00:00"}
        },
        {
            "role": "assistant",
            "content": "Opo! Miako handles thousands of users with sub-second latency kasi naka-Groq LPU siya. Hindi siya n8n na mabagal.",
            "metadata": {"created_at": "2023-10-27 10:00:05"}
        },
        {
            "role": "user",
            "content": "Weh? Di nga? Pano pag sabay-sabay gumamit? Edi lag yan panigurado.",
            "metadata": {"created_at": "2023-10-27 10:00:15"}
        },
        {
            "role": "assistant",
            "content": "Hindi po. Designed siya for high concurrency using Async FastAPI. May isolation din per user kaya walang lag.",
            "metadata": {"created_at": "2023-10-27 10:00:20"}
        },
        {
            "role": "user",
            "content": "Ah ganun ba. Eh pano kung nagonline lahat, di ba mahal yun? Yung budget kasi namin limited lang.",
            "metadata": {"created_at": "2023-10-27 10:00:35"}
        }
    ]



    # FOR THE TRANSLATED MEMORY MOCK (What the LLM 'sees' as English)
    taglish_translated_history = [
        {
            "role": "user",
            "content": "Hi, just asking, is this really fast? Because our old chatbot was very slow.",
            "metadata": {"created_at": "2023-10-27 10:00:00"}
        },
        {
            "role": "assistant",
            "content": "Yes! Miako handles thousands of users with sub-second latency because it uses Groq LPU. It is not like n8n which is slow.",
            "metadata": {"created_at": "2023-10-27 10:00:05"}
        },
        {
            "role": "user",
            "content": "Really? No way? What if everyone uses it at the same time? That would surely lag.",
            "metadata": {"created_at": "2023-10-27 10:00:15"}
        },
        {
            "role": "assistant",
            "content": "No sir. It is designed for high concurrency using Async FastAPI. There is also isolation per user so there is no lag.",
            "metadata": {"created_at": "2023-10-27 10:00:20"}
        },
        {
            "role": "user",
            "content": "Ah is that so. But what if everyone goes online, isn't that expensive? Our budget is limited.",
            "metadata": {"created_at": "2023-10-27 10:00:35"}
        }
    ]

    new_taglish_history = [
        {
            "role": "user",
            "content": "Hi, just asking, is this really fast? Because our old chatbot was very slow.",
            "metadata": {
                "created_at": "2023-10-27 10:00:00",
                 "original_text":"Hi, tanong lang, mabilis ba talaga to? Kasi yung luma naming chatbot sobrang bagal."
             }
        },
        {
            "role": "assistant",
            "content": "Yes! Miako handles thousands of users with sub-second latency because it uses Groq LPU. It is not like n8n which is slow.",
            "metadata": {
                "created_at": "2023-10-27 10:00:05",
                "original_text":"Yes! Miako handles thousands of users with sub-second latency because it uses Groq LPU. It is not like n8n which is slow.",
            }
        },
        {
            "role": "user",
            "content": "Really? No way? What if everyone uses it at the same time? That would surely lag.",
            "metadata": {
                "created_at": "2023-10-27 10:00:15",
                "original_text":"Weh? Di nga? Pano pag sabay-sabay gumamit? Edi lag yan panigurado."
            }
        },
        {
            "role": "assistant",
            "content": "No sir. It is designed for high concurrency using Async FastAPI. There is also isolation per user so there is no lag.",
            "metadata": {
                "created_at": "2023-10-27 10:00:20",
                "original_text":"Hindi po. Designed siya for high concurrency using Async FastAPI. May isolation din per user kaya walang lag."
            }
        },
        # {
        #     "role": "user",
        #     "content": "Ah is that so. But what if everyone goes online, isn't that expensive? Our budget is limited.",
        #     "metadata": {
        #         "created_at": "2023-10-27 10:00:35",
        #         "original_text":"Ah ganun ba. Eh pano kung nagonline lahat, di ba mahal yun? Yung budget kasi namin limited lang."
        #     }
        # }
        # #Optional
    ]


    taglish_user_input = "Ah is that so. But what if everyone goes online, isn't that expensive? Our budget is limited."


fake_memory = FakeMemory()