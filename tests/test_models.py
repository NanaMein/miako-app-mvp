# tests/test_models.py
import pytest
from sqlmodel import select
from models.user_model import User
from models.conversation_model import Conversation
from models.message_model import Message


@pytest.mark.asyncio
async def test_user_conversation_flow(session):
    # 1. Create a User
    # ----------------
    new_user = User(user_name="Nyanko", hashed_password="supersecurepassword")
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    assert new_user.id is not None
    assert new_user.user_name == "Nyanko"

    # 2. Create a Conversation linked to User
    # ---------------------------------------
    new_conv = Conversation(user_id=new_user.id)
    session.add(new_conv)
    await session.commit()
    await session.refresh(new_conv)

    assert new_conv.id is not None
    assert new_conv.user_id == new_user.id

    # 3. Create Messages linked to Conversation
    # -----------------------------------------
    msg1 = Message(content="How do I exit vim?", conversation_id=new_conv.id)
    msg2 = Message(content="You don't. You live there now.", conversation_id=new_conv.id)
    session.add_all([msg1, msg2])
    await session.commit()

    # 4. Verify Relationships (The "Magic" Part)
    # ------------------------------------------
    # We fetch the user again to see if the relationship loaded
    # Note: In async, relationships are lazy. We must explicitly load them
    # or refresh the object with the relationship attribute.

    # Option A: Refresh the object to load the relationship
    await session.refresh(new_user, ["conversations"])

    assert len(new_user.conversations) == 1
    assert new_user.conversations[0].id == new_conv.id

    # Option B: Verify the conversation side
    await session.refresh(new_conv, ["messages", "user"])
    assert len(new_conv.messages) == 2
    assert new_conv.user.user_name == "Nyanko"

    await session.delete(new_user)
    await session.commit()

    result  = await session.execute(select(Conversation).where(Conversation.id == new_conv.id))
    db_conv = result.scalars().first()

    assert db_conv is None, "Conversation was NOT deleted when User was deleted!"

    print("\n\nTest Passed: Relationships are solid.")