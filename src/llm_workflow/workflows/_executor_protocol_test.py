from src.llm_workflow.workflows.flows import AdaptiveChatbot
from src.llm_workflow.workflows.base import ChatbotExecutor





async def _chat_once(_executor: ChatbotExecutor) -> bool:
    # 1. Get user input
    _user_input = input("\n👤 User: ").strip()

    # 2. Exit condition
    if _user_input.lower() in {"exit", "quit", ":q"}:
        print("🪄 Magic shutting down... Goodbye!")
        return False

    try:
        # 3. Update the bot's state with the new message
        # Since AdaptiveChatbot is a class, we just update the attribute
        _executor.chat.input_message = _user_input

        # 4. Execute the flow
        print("🧠 Thinking...")
        result = await _executor.execute()

        # Note: Your executor.execute() currently returns whatever bot.run() returns.
        # If your flow returns a tuple (response, list), unpack it here:
        if isinstance(result, tuple):
            resp, _list = result
        else:
            resp, _list = result, []

        print(f"🤖 Bot: {resp}")
        if _list:
            print(f"📋 Context/Metadata: {_list}")

    except Exception as e:
        print(f"❌ Error: {e}")

    return True

async def _main_async():
    print("--- 🪄 Traceback Magic Interactive Test ---")
    print("Type 'exit' to quit.")

    # Initialize the Bot and the Executor
    _bot = AdaptiveChatbot(user_id="user_123", input_message="")
    _executor = ChatbotExecutor(chat=_bot)

    while True:
        keep_going = await _chat_once(_executor)
        if not keep_going:
            break

#     import asyncio
#     try:
#         asyncio.run(_main_async())
#     except KeyboardInterrupt:
#         pass