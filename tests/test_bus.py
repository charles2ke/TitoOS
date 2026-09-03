import pytest

from titoos import BROADCAST, Message, MessageBus


def test_send_delivers_to_recipient_mailbox():
    bus = MessageBus()
    bus.register("a")
    bus.register("b")

    bus.post("a", "b", "ping")

    assert bus.pending("b") == 1
    (message,) = bus.receive("b")
    assert message == Message(sender="a", to="b", payload="ping")
    assert bus.receive("b") == []


def test_send_to_unknown_recipient_raises():
    bus = MessageBus()
    bus.register("a")

    with pytest.raises(KeyError):
        bus.post("a", "nobody", "ping")


def test_broadcast_skips_sender():
    bus = MessageBus()
    for name in ("a", "b", "c"):
        bus.register(name)

    message = bus.broadcast("a", "hello")

    assert message.is_broadcast and message.to == BROADCAST
    assert bus.receive("a") == []
    assert [m.payload for m in bus.receive("b")] == ["hello"]
    assert [m.payload for m in bus.receive("c")] == ["hello"]


def test_metadata_is_carried_along():
    bus = MessageBus()
    bus.register("b")

    bus.post("a", "b", "payload", priority=3)

    (message,) = bus.receive("b")
    assert message.metadata == {"priority": 3}


def test_message_metadata_is_immutable():
    metadata = {"priority": 3}
    message = Message(sender="a", to="b", payload="payload", metadata=metadata)

    metadata["priority"] = 4

    assert message.metadata == {"priority": 3}
    with pytest.raises(TypeError):
        message.metadata["priority"] = 5


def test_unregister_removes_mailbox():
    bus = MessageBus()
    bus.register("a")
    bus.unregister("a")

    assert "a" not in bus.mailboxes()
    assert not bus.has_traffic()
