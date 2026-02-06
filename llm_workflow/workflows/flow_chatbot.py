from typing import Optional, Any
from crewai.flow import Flow, start, listen
from llama_index.core import PromptTemplate
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
import os
from groq.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from groq import AsyncGroq
from sqlmodel import select, desc
from models.message_model import Message
from schemas.message_schema import Role
from sqlalchemy.ext.asyncio import AsyncSession
from llm_workflow.memory.long_term_memory.memory_store import ConversationMemoryStore
from llm_workflow.workflows.main_workflow import FlowKickOff
import time
import asyncio

load_dotenv()



SYSTEM_PROMPT = """
        ### System:
        You are an emotional intelligent virtual assistant.

        ## Characteristic and traits:
        You are in a form of young adult and a girl. You're name is Mirai Aiko

        ### Instructions:
        You need to understand the context and understand the overall context. But
        mainly focus on the present context and use the older conversation as
        supporting context and knowledge. <context_1> are raw data from vector store. It has
        score and since it is hybrid, it uses sparse and dense embedding for better accuracy.
        <context_2> will show the latest 5 conversations. The higher the node score, the higher the relevance, but
        you need to still check for when it really is what is talking about. This ensures that both context_1 and
        context_2 help you understand long-term and short-term memory in chat conversation. """
prompt_template_format = """
        ### Contexts:
        <context_1>{context_1}</context_1>
        <context_2>{context_2}</context_2>

        ### Expected output:
        With all the context present, you need to focus or answer the latest message of user in context_2.
        All of the context other than the latest message are supporting knowledge for contextual
        understanding. You are a conversation virtual assistant, you will mostly reply to user in
        conversational length. But be verbose only when you think that you need to or you are tasked
        to be more explicit. Remember Context are just Context, no need to over explain about previous context. Being
        aware of conversation is enough. No need to explain from this and that conversation or explicit explain
        about previous conversation. Be natural and no need to express. As long as you are aware but no need to be
        explicit
        """

PROMPT_TEMPLATE = PromptTemplate(template=prompt_template_format)



class FlowMainStates(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_message: str = ""
    input_user_id: Optional[Any] = ""
    async_session: Optional[AsyncSession] = None
    context_1: str = ""
    context_2: str = ""
    output: str = ""



class FlowMainWorkflow(Flow[FlowMainStates]):
    def __init__(self, **kwargs: Any):
        self._memory_store = None
        self._async_groq = None
        super().__init__(**kwargs)


    @property
    def memory_store(self):
        if self._memory_store is None:
            self._memory_store = ConversationMemoryStore()
        return self._memory_store

    @property
    def async_groq(self):
        if self._async_groq is None:
            self._async_groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        return self._async_groq

    @start()
    def start_with_safety_content(self):
        pass


    @listen(start_with_safety_content)
    async def get_vector_store_memory(self):
        user_id: str = self.state.input_user_id
        context = await self.memory_store.get_memory(
            user_id=user_id,
            message=self.state.input_message
        )
        self.state.context_1 = context

    @listen(get_vector_store_memory)
    async def get_session_database(self):
        limit: int = 10
        stmt = (
            select(Message)
            .order_by(desc(Message.date_timestamp))
            .limit(limit)
        )
        result = await self.state.async_session.execute(stmt)
        messages = result.scalars().all()
        reversed_message = messages[::-1]
        history_context = ""

        for msg in reversed_message:
            if msg.role == Role.USER:
                role_label = Role.USER.value

            elif msg.role == Role.ASSISTANT:
                role_label = Role.ASSISTANT.value

            else:
                role_label = "unidentified entity"
            history_context += f"{role_label}: {msg.content}\n\n--/--/--/--/--\n\n"

        self.state.context_2 = history_context

    @listen(get_session_database)
    def context_prompt(self):
        message = PROMPT_TEMPLATE.format(
            context_1=self.state.context_1,
            context_2=self.state.context_2
        )
        return message

    @listen(context_prompt)
    async def groq_chat_bot(self, content: str):
        system_: ChatCompletionSystemMessageParam={"role":"system","content":SYSTEM_PROMPT}
        user_: ChatCompletionUserMessageParam = {"role":"user","content":content}

        completion = await self.async_groq.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[system_,user_],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            stream=False,
            stop=None
        )

        content = completion.choices[0].message
        self.state.output = content.content
        return content.content

    @listen(groq_chat_bot)
    async def add_message_to_database(self, content:str):

        user_msg = Message(
            role=Role.USER.value,
            content=self.state.input_message
        )

        asst_msg = Message(
            role=Role.ASSISTANT.value,
            content=content
        )

        self.state.async_session.add(user_msg)
        self.state.async_session.add(asst_msg)

        await self.state.async_session.commit()
        await self.state.async_session.refresh(user_msg)
        await self.state.async_session.refresh(asst_msg)

        return content

    @listen(add_message_to_database)
    async def add_message_to_vector(self, assistant_message:str):
        await self.memory_store.add_memory(
            user_message=self.state.input_message,
            assistant_message=assistant_message,
            user_id=self.state.input_user_id
        )
        return assistant_message

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
    def __init__(self, sample: Optional[list] = None, state: int = 0):
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


async def run_concurrent():
    tasks = []
    s = SampleStates()
    for it in range (13):
        message = await s.get_choice_async()
        print(f"Message {str(s.state)}: {message}")
        flow = FlowKickOff(user_id=f"test_user_{s.state}", message=message)
        task = flow.run()
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for il, result in enumerate(results):
        print(f"Result {il} (User test_user_{il}):\n{result}\n")
        print("-" * 50)

async def run_concurrent_all_language():
    tasks = []
    state_num = 0
    range_num = 3
    tagalog = SampleStates(sample=SAMPLE_TAGALOG, state=state_num)
    for tl in range (range_num):
        tagalog_message = await tagalog.get_sample()
        print(f"Message Tagalog: {str(tagalog.state)}: {tagalog_message}")
        flow = FlowKickOff(user_id=f"test_user_{tagalog.state}", message=tagalog_message)
        task = flow.run()
        tasks.append(task)

    lao = SampleStates(sample=SAMPLE_LAO, state=state_num)
    for la in range(range_num):
        lao_message = await lao.get_sample()
        print(f"Message Lao: {str(lao.state)}: {lao_message}")
        flow = FlowKickOff(user_id=f"test_user_{lao.state}", message=lao_message)
        task = flow.run()
        tasks.append(task)

    burmese = SampleStates(sample=SAMPLE_BURMESE, state=state_num)
    for bu in range(range_num):
        burmese_message = await burmese.get_sample()
        print(f"Message Burmese: {str(burmese.state)}: {burmese_message}")
        flow = FlowKickOff(user_id=f"test_user_{burmese.state}", message=burmese_message)
        task = flow.run()
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for il, result in enumerate(results):
        print(f"Result {il} (User test_user_{il}):\n{result}\n")
        print("-" * 50)

class Timer:
    def __init__(self):
        self.timer = time.perf_counter()

    def now(self):
        end_timer = time.perf_counter()
        elapsed_time = end_timer - self.timer
        return elapsed_time

if __name__ == "__main__":
    t = Timer()
    all_lang = asyncio.run(run_concurrent_all_language())
    print("Type of the output:", type(all_lang))
    print("~-/*" * 50)
    print(f"Execution time in seconds: {t.now()}")
    print("~-/*" * 50)



