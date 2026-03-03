from llm_workflow.workflows.base import ChatbotExecutor
from llm_workflow.workflows.flows import AdaptiveChatbot
from typing import List, Dict, Any
import time
import asyncio





SAMPLE_TAGALOG = [
    "Kamusta? May problema po ako sa aking laptop. Hindi siya magsisimula.",
    "Magandang araw po! Pasensya na po, ano pong exact na nangyayari sa inyong laptop? May error message po ba na lumalabas?",
    "Hindi po, wala pong error message. Parang dead lang talaga siya. Sinubukan ko nang i-plug sa charger pero walang reaction.",
    "Naiintindihan ko po ang inyong problema. Pwede po bang tanungin kung gaano na po katagal 'to? At nangyari po ba ito bigla o may nanguna pong incident?",
    "Nangyari po ito kahapon pagkatapos kong mag-update ng Windows. Nag-shutdown siya ng normal pero ngayon hindi na siya bumubukas.",
    "Ah, salamat po sa impormasyon! Malamang po na may issue sa Windows update. Pwede po ba kayong subukan ang hard reset? Pindutin lang po ang power button ng 15 seconds, pagkatapos ay pakitanggal ang charger at battery kung maaari.",
    "Sinubukan ko na po 'yan pero hindi pa rin gumagana. Ano pong next step?",
    "Pasensya na po, Master. Pwede po ba tayong subukan ang external monitor? Baka po kasi ang issue ay sa display card o screen lamang. May VGA o HDMI port po ba ang inyong laptop?",
    "Wala po akong external monitor. May iba pong paraan? Baka po ba ito hardware issue?"
]
SAMPLE_LAO = [
    "ສະບາຍດີ? ຂ້ອຍມີບັນຫາກັບ Laptop ຂອງຂ້ອຍ. ມັນເປີດບໍ່ຂຶ້ນເລີຍ.",
    "ສະບາຍດີ! ຂໍໂທດເດີ້, ມັນເກີດຫຍັງຂຶ້ນກັບ Laptop ຂອງເຈົ້າແທ້? ມີ Error message ຫຍັງຂຶ້ນມາບໍ່?",
    "ບໍ່ມີ, ບໍ່ມີ Error ຫຍັງເລີຍ. ຄືຈັ່ງມັນຕາຍໄປເລີຍ. ລອງສຽບສາຍ Charge ແລ້ວ ແຕ່ກໍບໍ່ມີການຕອບສະໜອງ.",
    "ເຂົ້າໃຈແລ້ວ. ຂໍຖາມແດ່ ມັນເປັນແບບນີ້ດົນປານໃດແລ້ວ? ແລ້ວມັນເປັນເອງເລີຍ ຫຼື ວ່າມີເຫດການຫຍັງເກີດຂຶ້ນກ່ອນໜ້ານີ້ບໍ່?",
    "ມັນເປັນຕັ້ງແຕ່ມື້ວານນີ້ ຫຼັງຈາກຂ້ອຍ Update Windows. ມັນ Shutdown ປົກກະຕິ ແຕ່ຕອນນີ້ເປີດບໍ່ຂຶ້ນແລ້ວ.",
    "ໂອ້, ຂອບໃຈສຳລັບຂໍ້ມູນ! ອາດຈະເປັນຍ້ອນການ Update Windows. ລອງເຮັດ Hard reset ເບິ່ງກ່ອນໄດ້ບໍ່? ໃຫ້ກົດປຸ່ມ Power ຄ້າງໄວ້ 15 ວິນາທີ, ຫຼັງຈາກນັ້ນໃຫ້ຖອດສາຍ Charge ແລະ ແບັດເຕີຣີອອກ (ຖ້າຖອດໄດ້).",
    "ລອງແລ້ວ ແຕ່ກໍຍັງໃຊ້ບໍ່ໄດ້. ຂັ້ນຕອນຕໍ່ໄປຕ້ອງເຮັດແນວໃດ?",
    "ຂໍໂທດນຳເດີ້ເຈົ້າ. ລອງຕໍ່ກັບຈໍ External monitor ເບິ່ງໄດ້ບໍ່? ບາງທີບັນຫາອາດຈະຢູ່ທີ່ Display card ຫຼື ຫນ້າຈໍ. Laptop ຂອງເຈົ້າມີ Port VGA ຫຼື HDMI ບໍ່?",
    "ຂ້ອຍບໍ່ມີຈໍ Monitor ນອກ. ມີວິທີອື່ນບໍ່? ຫຼື ວ່າມັນເປັນຍ້ອນ Hardware?"
]
SAMPLE_BURMESE = [
    "နေကောင်းလား? ကျွန်တော့် Laptop မှာ ပြဿနာတက်နေလို့။ စက်ဖွင့်လို့မရတော့ဘူး။",
    "မင်္ဂလာပါရှင်! အားနာပါတယ်။ Laptop က အတိအကျ ဘာဖြစ်နေတာလဲဟင်? Error message တစ်ခုခုများ တက်လာသေးလား?",
    "မတက်ပါဘူး။ ဘာ Error message မှမပြဘူး။ စက်က လုံးဝအသေဖြစ်နေတာ။ Charger ထိုးကြည့်ပေမယ့်လည်း ဘာမှမထူးခြားဘူး။",
    "နားလည်ပါပြီ။ ဒါဖြစ်နေတာ ဘယ်လောက်ကြာပြီလဲဟင်? ပြီးတော့ ဒါက ရုတ်တရက်ဖြစ်သွားတာလား၊ ဒါမှမဟုတ် တစ်ခုခုဖြစ်ပြီးမှ ဖြစ်သွားတာလား?",
    "မနေ့က Windows update လုပ်ပြီးမှ ဖြစ်သွားတာပါ။ ပုံမှန်အတိုင်း Shutdown ကျသွားပေမယ့် အခုကျတော့ ဖွင့်လို့မရတော့ဘူး။",
    "အော် အချက်အလက်အတွက် ကျေးဇူးပါ။ Windows update ကြောင့် ဖြစ်နိုင်ပါတယ်။ Hard reset အရင်စမ်းကြည့်လို့ရမလား? Power button ကို ၁၅ စက္ကန့်လောက် ဖိထားပေးပါ၊ ပြီးရင် Charger နဲ့ Battery ကို (ဖြုတ်လို့ရရင်) ဖြုတ်ထားပေးပါ။",
    "စမ်းကြည့်ပြီးပြီ၊ ဒါပေမယ့် မရသေးဘူး။ နောက်ထပ် ဘာလုပ်ရမလဲ?",
    "ဟုတ်ကဲ့ပါ။ External monitor နဲ့များ စမ်းကြည့်လို့ရမလား? တစ်ခါတလေ Display card ဒါမှမဟုတ် Screen ကြောင့်လည်း ဖြစ်နိုင်လို့ပါ။ Laptop မှာ VGA ဒါမှမဟုတ် HDMI port ပါလားခင်ဗျာ?",
    "ကျွန်တော့်မှာ External monitor မရှိဘူး။ တခြားနည်းရော ရှိသေးလား? ဒါ Hardware issue များ ဖြစ်နေတာလား?"
]

class SampleStates:
    def __init__(self, sample: list | None = None, state: int = 0):
        self.state = state
        self.lock = asyncio.Lock()
        self.sample = sample

    def get_choice(self):
        return self._sample_choice()

    async def get_choice_async(self):
        async with self.lock:
            if self.state >=len(SAMPLE_TAGALOG):
                self.state = 0
            _choice = SAMPLE_TAGALOG[self.state]
            self.state += 1
            return _choice

    async def get_sample(self):
        async with self.lock:
            if self.sample is not None:
                _sample = self.sample
            else:
                _sample = SAMPLE_TAGALOG

            if self.state >=len(_sample):
                self.state = 0
            sample_choice = _sample[self.state]
            self.state += 1
            return sample_choice



    def _sample_choice(self):
        _choice = SAMPLE_TAGALOG[self.state]
        self.state += 1
        return _choice



async def execute_with_timer(user_id: str, message: str):
    t0 = time.perf_counter()

    chatbot = AdaptiveChatbot(user_id=user_id, input_message=message)
    flow = ChatbotExecutor(chatbot)

    t1 = time.perf_counter()

    try:
        result = await flow.execute()
        success = True
    except Exception as e:
        result = str(e)
        success = False

    t2 = time.perf_counter()

    return {
        "user_id": user_id,
        "success": success,
        "total_time": t2 - t0,
        "flow_time": t2 - t1,
        "init_time": t1 - t0,
        "result_length": len(str(result))
    }


async def run_concurrent_all_language():
    tasks = []
    state_num = 5
    range_num = 3

    tagalog = SampleStates(sample=SAMPLE_TAGALOG, state=state_num)
    for _ in range(range_num):
        msg = await tagalog.get_sample()
        tasks.append(
            execute_with_timer(
                user_id="tagalog_",
                message=msg
            )
        )

    lao = SampleStates(sample=SAMPLE_LAO, state=state_num)
    for _ in range(range_num):
        msg = await lao.get_sample()
        tasks.append(
            execute_with_timer(
                user_id="lao_",
                message=msg
            )
        )

    burmese = SampleStates(sample=SAMPLE_BURMESE, state=state_num)
    for _ in range(range_num):
        msg = await burmese.get_sample()
        tasks.append(
            execute_with_timer(
                user_id="burmese_",
                message=msg
            )
        )

    task_results = await asyncio.gather(*tasks, return_exceptions=False)

    for r in task_results:
        print(f"[{r['user_id']}]")
        print(f"Success: {r['success']}")
        print(f"Total Time: {r['total_time']:.4f}s")
        print(f"Flow Time: {r['flow_time']:.4f}s")
        print(f"Init Time: {r['init_time']:.4f}s")
        print(f"Result Length: {r['result_length']}")
        print("-" * 50)

    return task_results


def summarize_async(_results):
    # Only collect times from successful requests
    times = [r["total_time"] for r in _results if r["success"]]

    print("\n=== SUMMARY ===")
    print(f"Total requests: {len(_results)}")
    print(f"Successful: {len(times)}")
    print(f"Failed: {len(_results) - len(times)}")

    # Check if there are any successful times before calculating
    if times:
        print(f"Avg time: {sum(times) / len(times):.4f}s")
        print(f"Min time: {min(times):.4f}s")
        print(f"Max time: {max(times):.4f}s")
    else:
        # Handle the case where there are no successful requests
        print("Avg time: N/A (No successful requests)")
        print("Min time: N/A")
        print("Max time: N/A")

    # Maybe... maybe check why everything failed? 🥺
    if len(times) == 0 and len(_results) > 0:
        print("\n⚠️ Warning: All tasks failed. Please check the logs...")


# def summarize_async(_results):
#     times = [r["total_time"] for r in _results if r["success"]]
#
#     print("\n=== SUMMARY ===")
#     print(f"Total requests: {len(_results)}")
#     print(f"Successful: {len(times)}")
#     print(f"Failed: {len(_results) - len(times)}")
#
#     if not times:
#         print("Avg time: N/A")
#         print("Min time: N/A")
#         print("Max time: N/A")
#         return
#
#     print(f"Avg time: {sum(times)/len(times):.4f}s")
#     print(f"Min time: {min(times):.4f}s")
#     print(f"Max time: {max(times):.4f}s")



# if __name__ == "__main__":
#     results = asyncio.run(run_concurrent_all_language())
#     summarize_async(results)




async def send_message_round(
        round_num: int,
        users: List[Dict[str, Any]],
        results_log: List[Dict[str, Any]]
) -> None:
    """Send one message per user in a round, concurrently"""
    tasks = []

    print(f"\n🔄 === ROUND {round_num} ===")

    for user_info in users:
        msg = await user_info["sample_state"].get_sample()
        print(f"  📤 User [{user_info['user_id']}]: Sending message...")

        task = execute_with_timer(
            user_id=user_info["user_id"],
            message=msg
        )
        tasks.append(task)

    # ✅ All users send concurrently in this round
    round_results = await asyncio.gather(*tasks, return_exceptions=False)

    for r in round_results:
        results_log.append(r)
        status = "✅" if r["success"] else "❌"
        print(f"  {status} User [{r['user_id']}]: Success={r['success']}, Time={r['total_time']:.4f}s")


async def run_concurrent_with_intervals():
    """
    Simulates 3 users, each sending 3 messages with 5-second intervals between rounds.
    This tests for race conditions while keeping async loop open.
    """
    results_log: List[Dict[str, Any]] = []
    num_rounds = 3
    interval_seconds = 5

    # ✅ Create 3 UNIQUE users (not shared user_ids!)
    users = [
        {
            "user_id": "tagalog_user_001",
            "sample_state": SampleStates(sample=SAMPLE_TAGALOG, state=0),
            "language": "Tagalog"
        },
        {
            "user_id": "lao_user_001",
            "sample_state": SampleStates(sample=SAMPLE_LAO, state=0),
            "language": "Lao"
        },
        {
            "user_id": "burmese_user_001",
            "sample_state": SampleStates(sample=SAMPLE_BURMESE, state=0),
            "language": "Burmese"
        }
    ]

    print("🚀 Starting concurrent test with intervals...")
    print(f"   Users: {[u['user_id'] for u in users]}")
    print(f"   Rounds: {num_rounds}")
    print(f"   Interval: {interval_seconds}s between rounds")

    for round_num in range(1, num_rounds + 1):
        # Send messages for all users in this round
        await send_message_round(round_num, users, results_log)

        # ✅ Non-blocking sleep - keeps async loop open!
        if round_num < num_rounds:
            print(f"\n⏳ Waiting {interval_seconds} seconds before next round...")
            await asyncio.sleep(interval_seconds)  # 🌸 Non-blocking!

    return results_log


async def verify_memory_integrity(results_: List[Dict[str, Any]]) -> None:
    """
    Verify that each user's messages were stored correctly.
    This helps detect race conditions.
    """
    print("\n🔍 === MEMORY INTEGRITY CHECK ===")

    # Group results by user
    user_results: Dict[str, List[Dict[str, Any]]] = {}
    for r in results_:
        uid = r["user_id"]
        if uid not in user_results:
            user_results[uid] = []
        user_results[uid].append(r)

    all_passed = True

    for user_id, user_data in user_results.items():
        total = len(user_data)
        success = sum(1 for r in user_data if r["success"])

        # ✅ Each user should have exactly 3 messages (one per round)
        expected = 3
        status = "✅ PASS" if total == expected and success == expected else "❌ FAIL"

        if total != expected or success != expected:
            all_passed = False

        print(f"  [{user_id}]")
        print(f"    Total messages: {total}/{expected} {status}")
        print(f"    Successful: {success}/{expected}")

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All integrity checks PASSED! No race conditions detected!")
    else:
        print("⚠️  Some checks FAILED! Possible race conditions exist!")
    print("=" * 50)


def summarize_interval(_results: List[Dict[str, Any]]) -> None:
    times = [r["total_time"] for r in _results if r["success"]]

    print("\n=== SUMMARY ===")
    print(f"Total requests: {len(_results)}")
    print(f"Successful: {len(times)}")
    print(f"Failed: {len(_results) - len(times)}")

    if not times:
        print("Avg time: N/A")
        print("Min time: N/A")
        print("Max time: N/A")
        return

    print(f"Avg time: {sum(times) / len(times):.4f}s")
    print(f"Min time: {min(times):.4f}s")
    print(f"Max time: {max(times):.4f}s")


if __name__ == "__main__":
    results = asyncio.run(run_concurrent_with_intervals())
    summarize_interval(results)
    asyncio.run(verify_memory_integrity(results))