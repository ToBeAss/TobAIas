try:
    import spidev
except ImportError:
    print("spidev library not installed. Please install it using 'pip install spidev'.")
    spidev = None

sensor_readings = []

def read_data():
    # Check if spidev was successfully imported
    if spidev is None:
        print("spidev is not available. Exiting read_data function.")
        return None
    
    # Initialize SPI
    spi = spidev.SpiDev()
    spi.open(0, 0) # Open SPI bus 0, device (CS) 0
    spi.max_speed_hz = 1350000 # Set SPI clock speed

    # Sensor calibration values
    dry_value = 750
    wet_value = 850

    # Read ADC data
    adc = spi.xfer2([1, (8 + 0) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]

    # Map data
    data_mapped = max(0, min(100, int((data - dry_value) * 100 / (wet_value - dry_value))))
    return data_mapped

def save_data(reading: int, timestamp):
    data = {
        "timestamp" : timestamp,
        "reading" : reading
    }

    sensor_readings.append(data)

print(read_data())
