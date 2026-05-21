"""
gateway/lora_parser.py
──────────────────────
Parses raw LoRa packets received by the depot gateway into Python dicts
ready for TimescaleDB ingest.

The 52-byte packet format mirrors node_firmware.ino exactly.
Also handles the JSON fallback format (development mode firmware).

Usage:
    from gateway.lora_parser import LoRaParser
    parser = LoRaParser()
    reading = parser.parse(raw_bytes)
    if reading:
        push_to_ingest_api(reading)
"""

import struct
import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger


# ── Packet format ─────────────────────────────────────────────────────────
# Mirrors the NodePacket struct in node_firmware.ino
#
#  Offset  Bytes  Type    Field
#  0       4      uint32  node_id
#  4       4      uint32  timestamp (Unix epoch)
#  8       2      int16   tilt_x100  (tilt_deg × 100)
#  10      2      int16   tilt_rate_x1000
#  12      6      uint8×6 moisture[0..5]  (% saturation)
#  18      2      uint16  pore_pressure_x10  (kPa × 10)
#  20      2      uint16  rainfall_1hr_x10   (mm × 10)
#  22      2      uint16  rainfall_6hr_x10
#  24      1      uint8   battery_pct
#  25      1      uint8   rssi  (stored as positive; subtract 150 for dBm)
#  26      1      uint8   risk_class  (0=GREEN 1=YELLOW 2=RED, local estimate)
#  27      1      uint8   hop_count
#  28      8      bytes   CRC + message_id
#  36      16     bytes   reserved
#  Total: 52 bytes

PACKET_FORMAT = "<IIhh6BHHHHBB8s16s"
PACKET_SIZE   = struct.calcsize(PACKET_FORMAT)   # should be 52

LABEL_MAP = {0: "GREEN", 1: "YELLOW", 2: "RED"}


@dataclass
class SensorReading:
    node_id:       str
    timestamp:     str          # ISO 8601 UTC
    tilt_deg:      float
    tilt_rate:     float        # deg/min
    moisture_d1:   float
    moisture_d2:   float
    moisture_d3:   float
    moisture_d4:   float
    moisture_d5:   float
    moisture_d6:   float
    pore_pressure: float        # kPa
    rainfall_1hr:  float        # mm/hr
    rainfall_6hr:  float        # mm
    battery_pct:   int
    rssi_dbm:      int
    local_risk:    str          # node's own estimate (not Navigator)
    hop_count:     int
    gateway_ts:    str          # when gateway received the packet

    def to_dict(self) -> dict:
        return asdict(self)


class LoRaParser:
    """
    Parses both binary (production) and JSON (development) LoRa packets.
    Validates ranges and logs anomalies.
    """

    # ── Validation ranges ─────────────────────────────────────────────────
    VALID_RANGES = {
        "tilt_deg":      (0.0,   5.0),
        "pore_pressure": (0.0, 100.0),
        "rainfall_1hr":  (0.0, 300.0),
        "battery_pct":   (0,    100),
        "moisture":      (0.0, 100.0),
    }

    def parse(self, raw: bytes | str) -> Optional[SensorReading]:
        """
        Parse a raw LoRa packet.

        Parameters
        ----------
        raw : bytes  → binary packet (production firmware)
              str    → JSON string (development/debug firmware)

        Returns None if the packet is malformed or fails validation.
        """
        try:
            if isinstance(raw, (bytes, bytearray)):
                return self._parse_binary(raw)
            else:
                return self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Packet parse error: {e} | raw={raw!r}")
            return None

    # ── Binary parser ─────────────────────────────────────────────────────

    def _parse_binary(self, raw: bytes) -> Optional[SensorReading]:
        if len(raw) < PACKET_SIZE:
            logger.warning(f"Short packet: {len(raw)} bytes (expected {PACKET_SIZE})")
            return None

        (
            node_id_int,
            timestamp_epoch,
            tilt_x100,
            tilt_rate_x1000,
            m1, m2, m3, m4, m5, m6,
            pore_x10,
            rain_1hr_x10,
            rain_6hr_x10,
            _unused,
            battery_pct,
            rssi_raw,
            risk_class,
            hop_count,
            _crc,
            _reserved,
        ) = struct.unpack(PACKET_FORMAT, raw[:PACKET_SIZE])

        reading = SensorReading(
            node_id       = f"NM-{node_id_int:02d}",
            timestamp     = datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc).isoformat(),
            tilt_deg      = tilt_x100 / 100.0,
            tilt_rate     = tilt_rate_x1000 / 1000.0,
            moisture_d1   = float(m1),
            moisture_d2   = float(m2),
            moisture_d3   = float(m3),
            moisture_d4   = float(m4),
            moisture_d5   = float(m5),
            moisture_d6   = float(m6),
            pore_pressure = pore_x10 / 10.0,
            rainfall_1hr  = rain_1hr_x10 / 10.0,
            rainfall_6hr  = rain_6hr_x10 / 10.0,
            battery_pct   = int(battery_pct),
            rssi_dbm      = int(rssi_raw) - 150,      # decode: stored as uint8 offset
            local_risk    = LABEL_MAP.get(risk_class, "UNKNOWN"),
            hop_count     = int(hop_count),
            gateway_ts    = datetime.now(timezone.utc).isoformat(),
        )

        return self._validate(reading)

    # ── JSON parser (development firmware) ───────────────────────────────

    def _parse_json(self, raw: str) -> Optional[SensorReading]:
        data = json.loads(raw)

        moisture = data.get("moisture", [0] * 6)
        while len(moisture) < 6:
            moisture.append(0.0)

        # Use gateway receive time if node timestamp is missing/stale
        ts_epoch = data.get("timestamp", int(time.time()))
        ts_str   = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()

        reading = SensorReading(
            node_id       = f"NM-{data['node_id']:02d}",
            timestamp     = ts_str,
            tilt_deg      = float(data.get("tilt_deg", 0.0)),
            tilt_rate     = float(data.get("tilt_rate", 0.0)),
            moisture_d1   = float(moisture[0]),
            moisture_d2   = float(moisture[1]),
            moisture_d3   = float(moisture[2]),
            moisture_d4   = float(moisture[3]),
            moisture_d5   = float(moisture[4]),
            moisture_d6   = float(moisture[5]),
            pore_pressure = float(data.get("pore_pressure", 0.0)),
            rainfall_1hr  = float(data.get("rainfall_1hr", 0.0)),
            rainfall_6hr  = float(data.get("rainfall_6hr", 0.0)),
            battery_pct   = int(data.get("battery_pct", 0)),
            rssi_dbm      = int(data.get("rssi", -100)),
            local_risk    = LABEL_MAP.get(data.get("risk_class", 0), "UNKNOWN"),
            hop_count     = int(data.get("hop_count", 0)),
            gateway_ts    = datetime.now(timezone.utc).isoformat(),
        )

        return self._validate(reading)

    # ── Validation ────────────────────────────────────────────────────────

    def _validate(self, r: SensorReading) -> Optional[SensorReading]:
        lo, hi = self.VALID_RANGES["tilt_deg"]
        if not (lo <= r.tilt_deg <= hi):
            logger.warning(f"{r.node_id}: tilt_deg={r.tilt_deg} out of range [{lo},{hi}]")
            return None

        lo, hi = self.VALID_RANGES["pore_pressure"]
        if not (lo <= r.pore_pressure <= hi):
            logger.warning(f"{r.node_id}: pore_pressure={r.pore_pressure} out of range")
            return None

        if not (0 <= r.battery_pct <= 100):
            logger.warning(f"{r.node_id}: battery_pct={r.battery_pct} invalid")
            r.battery_pct = max(0, min(100, r.battery_pct))

        # Low battery warning
        if r.battery_pct < 15:
            logger.warning(f"{r.node_id}: LOW BATTERY — {r.battery_pct}%")

        return r


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = LoRaParser()

    # Test JSON packet (development firmware)
    json_packet = json.dumps({
        "node_id":       7,
        "timestamp":     1720964100,
        "tilt_deg":      0.41,
        "tilt_rate":     0.05,
        "moisture":      [82.0, 85.0, 84.0, 81.0, 79.0, 77.0],
        "pore_pressure": 3.2,
        "rainfall_1hr":  12.1,
        "rainfall_6hr":  48.3,
        "battery_pct":   87,
        "rssi":          -98,
        "risk_class":    1,
        "hop_count":     2,
    })

    result = parser.parse(json_packet)
    if result:
        print("✅ JSON packet parsed:")
        for k, v in result.to_dict().items():
            print(f"   {k:<20} {v}")
    else:
        print("❌ JSON parse failed")

    # Test binary packet
    raw_binary = struct.pack(
        PACKET_FORMAT,
        7,            # node_id
        1720964100,   # timestamp
        41,           # tilt_x100  (0.41°)
        50,           # tilt_rate_x1000  (0.050°/min)
        82, 85, 84, 81, 79, 77,  # moisture[0..5]
        32,           # pore_x10  (3.2 kPa)
        121,          # rain_1hr_x10  (12.1 mm/hr)
        483,          # rain_6hr_x10  (48.3 mm)
        0,            # unused
        87,           # battery_pct
        52,           # rssi_raw  (52 - 150 = -98 dBm)
        1,            # risk_class  (YELLOW)
        2,            # hop_count
        b'\x00' * 8, # crc
        b'\x00' * 16, # reserved
    )

    result2 = parser.parse(raw_binary)
    if result2:
        print(f"\n✅ Binary packet parsed: node={result2.node_id}, "
              f"tilt={result2.tilt_deg}, p_pore={result2.pore_pressure}")
    else:
        print("❌ Binary parse failed")
