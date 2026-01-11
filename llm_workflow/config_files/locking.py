from cachetools import LRUCache, TTLCache
from typing import Optional, Union, Any
import asyncio


MANAGER_LIST_CACHE = LRUCache(maxsize=1000)
MANAGER_LOCK = asyncio.Lock()

class LockManager:
    def __init__(self, user_id: Union[str, Any] = None):
        self._user_id = user_id
        self.get_user_lock = self.get_lock()

    async def get_lock(self) -> asyncio.Lock:
        user_id = self._user_id

        if user_id in MANAGER_LIST_CACHE:
            return MANAGER_LIST_CACHE[user_id]

        async with MANAGER_LOCK:

            if user_id in MANAGER_LIST_CACHE:
                return MANAGER_LIST_CACHE[user_id]

            new_user_lock = asyncio.Lock()
            MANAGER_LIST_CACHE[user_id] = new_user_lock
            return new_user_lock

    async def get_lock_with_id(self, user_id: Optional[str]):
        if user_id in MANAGER_LIST_CACHE:
            return MANAGER_LIST_CACHE[user_id]

        async with MANAGER_LOCK:
            if user_id in MANAGER_LIST_CACHE:
                return MANAGER_LIST_CACHE[user_id]

            new_user_lock = asyncio.Lock()
            MANAGER_LIST_CACHE[user_id] = new_user_lock
            return new_user_lock