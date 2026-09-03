import pytest
import asyncio
from datetime import datetime
from src.notifier import WatchNotificationManager

@pytest.mark.asyncio
async def test_alert_deduplication_cooldown():
    notifier = WatchNotificationManager(
        enabled=True,
        voice_alert=False,
        ntfy_topic=None,
        telegram_bot_token=None,
        cooldown_seconds=30
    )

    # First alert should be permitted
    assert notifier.should_alert("Garden", "rat") is True
    res1 = await notifier.notify_animal_detected(
        confidence=0.92,
        description="Rat observed near compost bin",
        device_name="Garden",
        object_type="rat"
    )
    assert res1.get("sent") is True

    # Immediate second alert for same camera and target must be suppressed by cooldown
    assert notifier.should_alert("Garden", "rat") is False
    res2 = await notifier.notify_animal_detected(
        confidence=0.95,
        description="Rat still present near compost bin",
        device_name="Garden",
        object_type="rat"
    )
    assert res2.get("sent") is False
    assert res2.get("suppressed") is True

    # Alert for different camera or different object is still permitted
    assert notifier.should_alert("cam1", "rat") is True
    assert notifier.should_alert("Garden", "pheasant") is True

    # Clearing cooldown permits alert again
    notifier.clear_cooldown("Garden", "rat")
    assert notifier.should_alert("Garden", "rat") is True
