from cachetools import LRUCache, TTLCache
from typing import Optional, Union, Any
import asyncio


DEFAULT_LRU_CACHE = LRUCache(maxsize=1000)
DEFAULT_TTL_CACHE = TTLCache(maxsize=1000, ttl=3600)
DEFAULT_LOCK = asyncio.Lock()

class LockManager:
    def __init__(
            self, user_id: Union[str, Any],
            cache: Optional[Union[LRUCache, TTLCache]] = None,
            asyncio_lock: Optional[asyncio.Lock] = None,

    ):
        self._user_id = user_id
        self.manager_list_cache = cache
        self.manager_lock = asyncio_lock

    def default_cache(self):
        if self.manager_list_cache:
            return self.manager_list_cache
        return DEFAULT_LRU_CACHE

    def default_lock(self):
        if self.manager_lock:
            return self.manager_lock
        return DEFAULT_LOCK


    async def get_lock(self) -> asyncio.Lock:
        user_id = str(self._user_id)
        internal_cache = self.default_cache()
        internal_lock = self.default_lock()

        if user_id in internal_cache:
            return internal_cache[user_id]

        async with internal_lock:

            if user_id in internal_cache:
                return internal_cache[user_id]

            new_user_lock = asyncio.Lock()
            internal_cache[user_id] = new_user_lock
            return new_user_lock