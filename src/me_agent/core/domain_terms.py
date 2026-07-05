"""Shared ECU-domain term groups used by routing, retrieval, and fallback logic."""

THERMAL_QUERY_EXPANSION = (
    ("thermal", "tolerance", "tolerant", "temperature", "temp", "environment", "harsh"),
    "operating temperature temp range high temperature thermal",
)
FIRMWARE_QUERY_EXPANSION = (
    ("firmware", "remote", "update", "updates", "ota"),
    "ota over the air over-the-air firmware update updates remote",
)
AI_QUERY_EXPANSION = (
    ("ai", "inference", "edge", "accelerator", "neural", "npu"),
    "npu neural processing unit accelerator edge ai inference tops",
)

GENERATION_QUERY_EXPANSION_GROUPS = (
    THERMAL_QUERY_EXPANSION,
    FIRMWARE_QUERY_EXPANSION,
    AI_QUERY_EXPANSION,
)

RETRIEVAL_QUERY_EXPANSION_GROUPS = (
    THERMAL_QUERY_EXPANSION,
    FIRMWARE_QUERY_EXPANSION,
    AI_QUERY_EXPANSION,
    (
        ("memory", "ram"),
        "memory ram lpddr sram",
    ),
    (
        ("storage", "capacity", "emmc", "flash"),
        "storage capacity emmc flash gb mb",
    ),
    (
        ("current", "load", "draw", "voltage", "power"),
        "power consumption load current draw voltage amps ma",
    ),
    (
        ("can", "bus", "speed", "interface"),
        "can bus canfd fd interface channel mbps speed",
    ),
)
