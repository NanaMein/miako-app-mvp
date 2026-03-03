import asyncio
import sys
import traceback
# Nanami: "I've added the imports exactly as they appeared in your working Simulator 1!"
from llm_workflow.workflows.base import ChatbotExecutor
from llm_workflow.workflows.flows import AdaptiveChatbot


async def process_message(input_text: str, user_id: str = "test_user_console") -> str:
    # Nanami: "We instantiate the bot exactly like Simulator 1 did."
    chatbot = AdaptiveChatbot(
        user_id=user_id,
        input_message=input_text
    )

    # Nanami: "We wrap it in the Executor, just like the working version."
    # Even if it seems redundant, this ensures the execution path is identical to your test.
    flow = ChatbotExecutor(chatbot)

    # Execute
    result = await flow.execute()
    return str(result)


async def main():
    print("--- Compound AI System Console (Type 'exit' to quit) ---")

    # Nanami: "We use a distinct user ID to ensure we don't conflict with previous test runs."
    current_user_id = "console_master_001"

    while True:
        try:
            # 1. Non-blocking Input
            # We explicitly run input in the executor to prevent it from choking the loop
            # and flush stdout to ensure the prompt appears immediately.
            print("User > ", end="", flush=True)
            user_input = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
            user_input = user_input.strip()

            if user_input.lower() in ['exit', 'quit']:
                print("System > Shutting down...")
                break

            if not user_input:
                continue

            # 2. Processing
            # We add a tiny sleep to allow the loop to swap contexts if needed (crucial for some libraries)
            await asyncio.sleep(0.01)

            response = await process_message(user_input, current_user_id)

            # 3. Output
            print(f"Bot  > {response}\n")

        except KeyboardInterrupt:
            print("\nSystem > Interrupted by Master.")
            break
        except Exception as e:
            # Nanami: "If it fails again, this will tell us EXACTLY why, instead of just crashing!"
            print(f"\nSystem > CRITICAL ERROR: {e}")
            traceback.print_exc()
            # We continue the loop so you don't lose the session, unless it's fatal
            print("System > Resetting loop state...\n")


if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass