import can
import cantools
import logging
import time
import asyncio
import copy
import crc8
from typing import Tuple, Optional, List, Dict


class MCMController:
    """Motion Control Module interface for vehicle control"""

    def __init__(self, channel: str):
        self.logger = logging.getLogger(__name__)
        self.db = cantools.database.Database()

        try:
            self.db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
            self.db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
            self.db.add_dbc_file('./sygnal_dbc/mcm/Override.dbc')
            self.logger.info("Loaded DBC files successfully")
        except Exception as e:
            self.logger.error(f"Failed to load DBC files: {e}")
            raise

        try:
            self.bus = can.Bus(bustype='socketcan',
                               channel=channel,
                               bitrate=500000)
            self.logger.info(f"CAN bus initialized on channel {channel}")
        except Exception as e:
            self.logger.error(f"Failed to initialize CAN bus: {e}")
            raise

        self.control_count = 0
        self.bus_address = 1
        self.control_states = {
            'steer': False,
            'throttle': False,
            'brake': False
        }

    def calc_crc8(self, data: bytearray) -> int:
        """Calculate CRC8 for CAN message"""
        data_copy = copy.copy(data)
        del data_copy[7]
        hash = crc8.crc8()
        hash.update(data_copy)
        return hash.digest()[0]

    async def control_request(self, module: str, request: bool) -> None:
        """Send control request message for any module"""
        try:
            control_enable_msg = self.db.get_message_by_name('ControlEnable')
            enable = 1 if request else 0
            interface = {'brake': 0, 'throttle': 1, 'steer': 2}[module]

            data = bytearray(control_enable_msg.encode({
                'BusAddress': self.bus_address,
                'InterfaceID': interface,
                'Enable': enable,
                'CRC': 0
            }))
            data[7] = self.calc_crc8(data)

            msg = can.Message(
                arbitration_id=control_enable_msg.frame_id,
                is_extended_id=False,
                data=data
            )
            self.bus.send(msg)
            self.logger.info(f"Sent control request: {msg}")

            await asyncio.sleep(0.05)
            self.control_states[module] = request
            self.logger.debug(f"Control request sent: {module} {'enabled' if request else 'disabled'}")

        except Exception as e:
            self.logger.error(f"Control request error: {e}")
            raise

    async def update_setpoint(self, module: str, value: float) -> None:
        """Send control command message for any module"""
        try:
            if self.control_states[module]:
                control_cmd_msg = self.db.get_message_by_name('ControlCommand')
                interface = {'brake': 0, 'throttle': 1, 'steer': 2}[module]

                # Ensure value is within bounds [-1, 1]
                value = max(min(value, 1.0), -1.0)

                data = bytearray(control_cmd_msg.encode({
                    'BusAddress': self.bus_address,
                    'InterfaceID': interface,
                    'Count8': self.control_count,
                    'Value': value,
                    'CRC': 0
                }))
                data[7] = self.calc_crc8(data)

                msg = can.Message(
                    arbitration_id=control_cmd_msg.frame_id,
                    is_extended_id=False,
                    data=data
                )
                self.bus.send(msg)
                self.logger.info(f"Sent {module} command: {msg}")

                self.control_count = (self.control_count + 1) % 256
                await asyncio.sleep(0.05)
                self.logger.debug(f"{module.capitalize()} setpoint updated: {value}")
            else:
                self.logger.warning(f"Cannot update {module} setpoint: control not enabled")

        except Exception as e:
            self.logger.error(f"Setpoint update error: {e}")
            raise

    async def send_heartbeat_clear_seed(self, subsystem_id: int) -> None:
        """Send HeartbeatClearSeed message for specified subsystem"""
        try:
            heartbeat_clear_msg = self.db.get_message_by_name('HeartbeatClearSeed')
            seed = int(time.time() * 1000) % 0xFFFFFFFF  # Use timestamp as seed

            data = bytearray(heartbeat_clear_msg.encode({
                'BusAddress': self.bus_address,
                'SubsystemID': subsystem_id,
                'ResetSeed': seed,
                'CRC': 0
            }))
            data[7] = self.calc_crc8(data)

            msg = can.Message(
                arbitration_id=heartbeat_clear_msg.frame_id,
                is_extended_id=False,
                data=data
            )
            self.bus.send(msg)
            self.logger.info(f"Sent HeartbeatClearSeed for subsystem {subsystem_id}")
            await asyncio.sleep(0.05)  # Small delay between messages

        except Exception as e:
            self.logger.error(f"Failed to send HeartbeatClearSeed: {e}")
            raise

    async def send_heartbeat_clear_key(self, subsystem_id: int) -> None:
        """Send HeartbeatClearKey message for specified subsystem"""
        try:
            heartbeat_clear_msg = self.db.get_message_by_name('HeartbeatClearKey')
            key = int(time.time() * 1000) % 0xFFFFFFFF  # Use timestamp as key

            data = bytearray(heartbeat_clear_msg.encode({
                'BusAddress': self.bus_address,
                'SubsystemID': subsystem_id,
                'ResetKey': key,
                'CRC': 0
            }))
            data[7] = self.calc_crc8(data)

            msg = can.Message(
                arbitration_id=heartbeat_clear_msg.frame_id,
                is_extended_id=False,
                data=data
            )
            self.bus.send(msg)
            self.logger.info(f"Sent HeartbeatClearKey for subsystem {subsystem_id}")
            await asyncio.sleep(0.05)  # Small delay between messages

        except Exception as e:
            self.logger.error(f"Failed to send HeartbeatClearKey: {e}")
            raise

    async def clear_subsystem_state(self, subsystem_id: int) -> None:
        """Clear state for a specific subsystem using seed-key sequence"""
        try:
            # Send seed first
            await self.send_heartbeat_clear_seed(subsystem_id)
            await asyncio.sleep(0.1)  # Wait for seed to be processed

            # Then send key
            await self.send_heartbeat_clear_key(subsystem_id)
            await asyncio.sleep(0.1)  # Wait for key to be processed

            self.logger.info(f"Completed state clearing for subsystem {subsystem_id}")

        except Exception as e:
            self.logger.error(f"Failed to clear state for subsystem {subsystem_id}: {e}")
            raise