# Automated Column Chromatography Data Collection System

## Device Communication Architecture
**Connection Protocol**: 
- RS232 serial communication for real-time data synchronization
- Parallel connection of 4 serial ports for multi-device control

**Connected Devices**:
1. Mobile phase pumps (×2)
2. UV-Vis detector
3. Autosampler

## Key Features
- **Real-time Monitoring**:
  - Continuous data streaming from detectors
  - Instant pump pressure feedback
- **Dynamic Port Handling**:
  ```python
  # Example port detection logic
  def detect_serial_ports():
      return [p.device for p in serial.tools.list_ports.comports()]
  ```
  - Automatic port number adaptation
  - Hot-swappable device support

## Data Flow
`Control Terminal` ↔ `RS232 Serial Hub` ↔ **{Pump1, Pump2, Detector, Autosampler}**
```
