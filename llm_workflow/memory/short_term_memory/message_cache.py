from typing import Union, Dict, Optional, List, Any, Protocol, Self
from collections import deque
from datetime import datetime
import asyncio
import uuid
import time



_MASTER_LOCK = asyncio.Lock()
_USER_MEMORY: Dict[str, "UserMemory"] = {}
_CLEANUP_TASK: Optional[asyncio.Task] = None
MAX_TTL_SECONDS = 3600
CLEANUP_INTERVAL = 600
DEFAULT_SYSTEM = "You are a helpful assistant"

class StorageBase(Protocol):

    @property
    def user_id(self) -> str | None:
        ...

    @staticmethod
    def _create_message(role: str, content: str, **kwargs: Any) -> dict[str, Any]:
        ...

    async def add_human_message(self, content: str, **metadata: Any) -> Self:
        ...

    async def add_ai_message(self, content: str, **metadata: Any) -> Self:
        ...

    async def get_messages(self, include_metadata: bool = False) -> List[dict[str,Any]]:
        ...

    async def get_messages_with_system(self,system_instruction: str = DEFAULT_SYSTEM) -> List[dict[str,Any]]:
        ...


class UserMemory:
    __slots__ = ["lock","messages","last_accessed"]
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.messages: deque = deque(maxlen=50)
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


    @staticmethod
    def _create_message(role: str, content: str, **kwargs) -> dict[str, Any]:
        msg = {"role": role, "content": content, "created_at": datetime.now().isoformat()}
        if kwargs:
            for k in("role", "content", "created_at"):
                kwargs.pop(k, None)
            msg.update(kwargs)
        return msg

    async def add_human_message(self, content: str, **metadata):
        _user = await self._get_user_memory()
        async with _user.lock:
            msg = self._create_message(role="user", content=content, **metadata)
            _user.messages.append(msg)
            _user.last_accessed = time.monotonic()
        return self

    async def add_ai_message(self, content: str, **metadata):
        _user = await self._get_user_memory()
        async with _user.lock:
            msg = self._create_message(role="assistant", content=content, **metadata)
            _user.messages.append(msg)
            _user.last_accessed = time.monotonic()
        return self

    async def get_messages(self, include_metadata: bool = False) -> List[dict]:
        _user = await self._get_user_memory()
        async with _user.lock:
            current_history = list(_user.messages)

            if include_metadata:
                return current_history

            clean_list = []

            for msg in current_history:
                clean_list.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            return clean_list



    async def get_messages_with_system(self, system_instructions: str = DEFAULT_SYSTEM) -> List[dict]:
        _user = await self._get_user_memory()
        async with _user.lock:

            clean_list = [{"role":"system", "content":system_instructions}]

            for msg in _user.messages:
                clean_msg = {
                    "role":msg["role"],
                    "content":msg["content"]
                }
                clean_list.append(clean_msg)

            return clean_list