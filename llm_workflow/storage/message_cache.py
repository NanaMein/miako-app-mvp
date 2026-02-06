from typing import Union, Dict, Optional, List
import asyncio
import uuid
import time



_MASTER_LOCK = asyncio.Lock()
_USER_MEMORY: Dict[str, "UserMemory"] = {}
_CLEANUP_TASK: Optional[asyncio.Task] = None
MAX_TTL_SECONDS = 3600
CLEANUP_INTERVAL = 600
DEFAULT_SYSTEM = "You are a helpful assistant"



class UserMemory:
    __slots__ = ["lock","messages","last_accessed"]
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.messages: list[dict] = []
        self.last_accessed: float = time.monotonic()

async def _background_cleanup() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        time_now = time.monotonic()
        keys_to_remove = []

        async with _MASTER_LOCK:
            for user_id, mem in _USER_MEMORY.items():
                time_before = mem.last_accessed

                time_computed = time_now - time_before

                if time_computed > MAX_TTL_SECONDS:
                    keys_to_remove.append(user_id)

            for key in keys_to_remove:
                del _USER_MEMORY[key]


class MessageStorage:
    def __init__(self, user_id: Union[str, uuid.UUID, None] = None):
        self._user_id = user_id

        global _CLEANUP_TASK
        if _CLEANUP_TASK is None:
            try:
                loop = asyncio.get_running_loop()
                _CLEANUP_TASK = loop.create_task(_background_cleanup())

            except RuntimeError:
                pass

    @staticmethod
    def human_content(content: str) -> dict[str, str]:
        return {"role":"user","content":content}

    @staticmethod
    def ai_content(content: str) -> dict[str, str]:
        return {"role": "assistant", "content": content}

    @staticmethod
    def system_instructions(content: str) -> dict[str, str]:
        return {"role": "system", "content": content}

    @property
    def user_id(self) -> Union[str, None]:
        _id = self._user_id

        if _id is None:
            return None

        if isinstance(_id, uuid.UUID):
            _id = _id.hex
        return _id

    async def _get_user_memory(self) -> Union[UserMemory, None]:
        user_id = self.user_id
        if user_id is None:
            raise Exception("No user_id provided")

        _user = _USER_MEMORY.get(user_id)
        if _user:
            _user.last_accessed = time.monotonic()
            return _user

        async with _MASTER_LOCK:
            _user = _USER_MEMORY.get(user_id)
            if _user is None:
                _user = UserMemory()
                _USER_MEMORY[user_id] = _user
            _user.last_accessed = time.monotonic()
            return _user

    async def add_human_message(self, content: str):
        _user = await self._get_user_memory()
        async with _user.lock:
            _user.messages.append(self.human_content(content))
            _user.last_accessed = time.monotonic()
        return self

    async def add_ai_message(self, content: str):
        _user = await self._get_user_memory()
        async with _user.lock:
            _user.messages.append(self.ai_content(content))
            _user.last_accessed = time.monotonic()
        return self

    async def get_messages(self) -> List[dict]:
        _user = await self._get_user_memory()
        async with _user.lock:
            return list(_user.messages)

    async def get_messages_with_system(self, system_instructions: str = DEFAULT_SYSTEM) -> List[dict]:
        _user = await self._get_user_memory()
        async with _user.lock:
            return [self.system_instructions(system_instructions), *_user.messages]