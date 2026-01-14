from cachetools import LRUCache, TTLCache
from typing import Optional, Union, Any
import asyncio


DEFAULT_LIST_CACHE = LRUCache(maxsize=1000)
DEFAULT_LOCK = asyncio.Lock()

class LockManager:
    def __init__(
            self, user_id: Union[str, Any],
            lru_cache: Optional[LRUCache] = DEFAULT_LIST_CACHE,
            asyncio_lock: Optional[asyncio.Lock] = DEFAULT_LOCK,

    ):

        self._user_id = user_id
        self.get_user_lock = self.get_lock()
        self.manager_list_cache = lru_cache
        self.manager_lock = asyncio_lock

    async def get_lock(self) -> asyncio.Lock:
        user_id =str(self._user_id)

        if user_id in self.manager_list_cache:
            return self.manager_list_cache[user_id]

        async with self.manager_lock:

            if user_id in self.manager_list_cache:
                return self.manager_list_cache[user_id]

            new_user_lock = asyncio.Lock()
            self.manager_list_cache[user_id] = new_user_lock
            return new_user_lock

    async def get_lock_with_id(self, user_id: Optional[str]):
        if user_id in self.manager_list_cache:
            return self.manager_list_cache[user_id]

        async with self.manager_lock:
            if user_id in self.manager_list_cache:
                return self.manager_list_cache[user_id]

            new_user_lock = asyncio.Lock()
            self.manager_list_cache[user_id] = new_user_lock
            return new_user_lock