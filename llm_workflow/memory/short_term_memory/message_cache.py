from typing import Union, Dict, Optional, List, Any, Protocol, Self
from collections import deque
from datetime import datetime
import asyncio
import uuid
import time

_MASTER_LOCK = asyncio.Lock()
_USER_MEMORY: Dict[str, "UserMemory"] = {}
_CLEANUP_TASK: Optional[asyncio.Task[None]] = None
MAX_TTL_SECONDS = 3600
CLEANUP_INTERVAL = 600
DEFAULT_SYSTEM = "You are a helpful assistant"

class StorageBase(Protocol):

    @property
    def user_id(self) -> str | None:
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
        self.messages: deque[dict[str, Any]] = deque(maxlen=50)
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

    async def _get_user_memory(self) -> UserMemory:
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
    def _create_message(role: str, content: str, **kwargs: Any) -> dict[str, Any]:
        msg = {"role": role, "content": content, "created_at": datetime.now().isoformat()}
        if kwargs:
            for k in("role", "content", "created_at"):
                kwargs.pop(k, None)
            msg.update(kwargs)
        return msg

    async def add_human_message(self, content: str, **metadata: Any)-> Self:
        _user = await self._get_user_memory()
        async with _user.lock:
            msg = self._create_message(role="user", content=content, **metadata)
            _user.messages.append(msg)
            _user.last_accessed = time.monotonic()
        return self

    async def add_ai_message(self, content: str, **metadata: Any)->Self:
        _user = await self._get_user_memory()
        async with _user.lock:
            msg = self._create_message(role="assistant", content=content, **metadata)
            _user.messages.append(msg)
            _user.last_accessed = time.monotonic()
        return self

    async def get_messages(self, include_metadata: bool = False) -> List[dict[str,Any]]:
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



    async def get_messages_with_system(self, system_instructions: str = DEFAULT_SYSTEM) -> List[dict[str,Any]]:
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

async def _background_cleanup_v1() -> None:
    try:
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL)
                time_now = time.monotonic()
                keys_to_remove = []

                async with _MASTER_LOCK:
                    for user_id, mem in list(_USER_MEMORY.items()):

                        time_before = mem.last_accessed

                        if (time_now - time_before) > MAX_TTL_SECONDS:
                            keys_to_remove.append(user_id)

                    for key in keys_to_remove:
                        _USER_MEMORY.pop(key, None)
            except asyncio.CancelledError:
                raise

    finally:
        global _CLEANUP_TASK
        _CLEANUP_TASK = None



class MessageStorageV1:
    def __init__(self, user_id: str | uuid.UUID | None = None):
        self._user_id = user_id

        global _CLEANUP_TASK
        if _CLEANUP_TASK is None:
            try:
                loop = asyncio.get_running_loop()
                _CLEANUP_TASK = loop.create_task(_background_cleanup_v1())

            except RuntimeError:
                pass

    @property
    def user_id(self) -> str | None:
        _id = self._user_id

        if _id is None:
            raise ValueError("No user_id provided")

        if isinstance(_id, uuid.UUID):
            _id = _id.hex
        return _id



    async def _get_user_contents(self) -> UserMemory:
        user_id = self.user_id

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
    def _create_message_template(role: str, content: str, **kwargs: Any) -> dict[str, Any]:

        user_metadata = kwargs.pop("metadata", {})

        if not isinstance(user_metadata, dict):
            raise TypeError("Metadata must be dict")

        reserved_keys = {"role", "content", "metadata"}
        clean_kwargs = {}
        for key, value in kwargs.items():
            if key not in reserved_keys:
                clean_kwargs[key] = value

        final_metadata = {**user_metadata, **clean_kwargs}
        return {
            "role": role,
            "content": content,
            "metadata": final_metadata
        }



    async def add_human_message(self, content: str, **metadata: Any) -> Self:
        _user = await self._get_user_contents()
        async with _user.lock:
            msg = self._create_message_template(role="user", content=content, **metadata)
            _user.messages.append(msg)
            _user.last_accessed = time.monotonic()
        return self


    async def add_ai_message(self, content: str, **metadata: Any) -> Self:
        _user = await self._get_user_contents()
        async with _user.lock:
            msg = self._create_message_template(role="assistant", content=content, **metadata)
            _user.messages.append(msg)
            _user.last_accessed = time.monotonic()
        return self

    async def update_last_message(self, metadata: dict[str, Any]):
        _user = await self._get_user_contents()

        async with _user.lock:
            if not _user.messages:
                return False

            last_message = _user.messages[-1]
            if "metadata" not in last_message or not isinstance(last_message["metadata"], dict):
                last_message["metadata"] = {}

            last_message["metadata"].update(metadata)
            _user.last_accessed = time.monotonic()
            return True


    async def get_messages(self, include_metadata: bool = False) -> List[dict[str,Any]]:
        _user = await self._get_user_contents()
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


    async def get_messages_with_system(self,system_instruction: str = DEFAULT_SYSTEM) -> List[dict[str,Any]]:
        _user = await self._get_user_contents()
        async with _user.lock:

            clean_list_with_system = [{"role":"system", "content":system_instruction}]
            for msg in _user.messages:
                clean_msg = {
                    "role":msg["role"],
                    "content":msg["content"]
                }
                clean_list_with_system.append(clean_msg)

            return clean_list_with_system

    async def get_metadata_only(self, include_only: list[str] | None = None, flatten: bool = False):
        _user = await self._get_user_contents()
        async with _user.lock:
            results: list[dict[str, Any]] = []

            for msg in _user.messages:
                meta = msg.get("metadata", {})

                if not isinstance(meta, dict):
                    meta = {}

                if include_only is None:
                    if flatten:
                        results.append(meta)
                    else:
                        results.append({"metadata": meta})
                    continue

                filtered_meta = {
                    key: value
                    for key, value in meta.items()
                    if key in include_only
                }
                if flatten:
                    results.append(filtered_meta)
                else:
                    results.append({"metadata":filtered_meta})
            return results


            clean_list = []
            for msg in _user.messages:
                only_meta = { "metadata": msg["metadata"] }
                clean_list.append(only_meta)

            for key, value in only_meta.items():
                if key in include_only:
                    clean_list[key] = value

            return clean_list


