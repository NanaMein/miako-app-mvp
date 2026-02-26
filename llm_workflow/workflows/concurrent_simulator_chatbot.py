from llm_workflow.workflows.base import ChatbotExecutor
from llm_workflow.workflows.flows import AdaptiveChatbot
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
    range_num = 10

    tagalog = SampleStates(sample=SAMPLE_TAGALOG, state=state_num)
    for _ in range(range_num):
        msg = await tagalog.get_sample()
        tasks.append(
            execute_with_timer(
                user_id=f"tagalog_{tagalog.state}",
                message=msg
            )
        )

    lao = SampleStates(sample=SAMPLE_LAO, state=state_num)
    for _ in range(range_num):
        msg = await lao.get_sample()
        tasks.append(
            execute_with_timer(
                user_id=f"lao_{lao.state}",
                message=msg
            )
        )

    burmese = SampleStates(sample=SAMPLE_BURMESE, state=state_num)
    for _ in range(range_num):
        msg = await burmese.get_sample()
        tasks.append(
            execute_with_timer(
                user_id=f"burmese_{burmese.state}",
                message=msg
            )
        )

    task_results = await asyncio.gather(*tasks, return_exceptions=False)

    for r in task_results:
        print(f"[{r['user_id']}]")
        print(f"Time: {r['execution_time']:.4f}s")
        print(f"Success: {r['success']}")
        print(f"Result: {r['result']}")
        print("-" * 50)

    return task_results

def summarize(_results):
    times = [r["execution_time"] for r in _results if r["success"]]

    print("\n=== SUMMARY ===")
    print(f"Total requests: {len(_results)}")
    print(f"Successful: {len(times)}")
    print(f"Failed: {len(_results) - len(times)}")
    print(f"Avg time: {sum(times)/len(times):.4f}s")
    print(f"Min time: {min(times):.4f}s")
    print(f"Max time: {max(times):.4f}s")

if __name__ == "__main__":
    results = asyncio.run(run_concurrent_all_language())
    summarize(results)



