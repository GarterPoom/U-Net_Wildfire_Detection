"""Entry script for training the U-Net wildfire segmentation model."""

import time  # Import time to measure the duration of the execution (the stopwatch)
import logging  # Import logging to handle writing information to files and console
import os  # Import os to perform operating system tasks like directory creation
from datetime import datetime  # Import datetime to create dynamic, timestamped filenames
from pathlib import Path  # Import Path for modern, robust filesystem path handling
from unet_wildfire_training import TrainingConfig  # Import the configuration class from your package
from unet_wildfire_training.training import train_model  # Import the main training function

def main() -> None:
    # 1. Define the directory where all your logs will be stored
    log_dir = Path("logs") 
    # 2. Create the 'logs' directory; 'parents=True' creates missing folders in the path, 'exist_ok=True' prevents error if it exists
    log_dir.mkdir(parents=True, exist_ok=True) 
    
    # 3. Generate a timestamp string (YearMonthDay_HourMinuteSecond) to make the filename unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    # 4. Create the full path for the log file using the dynamic timestamp
    log_file = log_dir / f"wildfire_training_log_{timestamp}.log" 
    
    # 5. Configure the logging system
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum level to capture: INFO, WARNING, ERROR, and CRITICAL
        format='%(asctime)s [%(levelname)s] %(message)s',  # Define how each log line looks (Timestamp, Level, Message)
        handlers=[
            logging.FileHandler(log_file),  # This tells Python to write logs into our dynamic file
            logging.StreamHandler()         # This tells Python to also print logs to the terminal/command prompt
        ]  # Use both handlers so you see it live AND save it to the file
    )

    # 6. Initialize the stopwatch by capturing the high-precision start time
    start_time = time.perf_counter() 
    
    # 7. Log a message indicating the start of the process
    logging.info("======================================================") # Log a visual separator
    logging.info(f"STARTING TRAINING SESSION: {timestamp}") # Log the start time/date
    logging.info("======================================================") # Log a visual separator

    try: # Wrap the execution in a try block to ensure we can handle errors gracefully
        # 8. Create the configuration object with default values
        config = TrainingConfig() 
        # 9. Call the training function with the configuration
        train_model(config) 
        # 10. Log a message if the training finishes without crashing
        logging.info("======================================================")
        logging.info("SUCCESS: Training process completed without errors.")
        logging.info("======================================================")
    except Exception as e: # Catch any unexpected error that occurs during training
        # 11. Log the error details so they are saved in the log file for debugging
        logging.error(f"CRITICAL ERROR during training: {e}", exc_info=True) 
    finally: # The 'finally' block runs no matter what (even if the code crashed)
        # 12. Capture the end time to stop the stopwatch
        end_time = time.perf_counter() 
        # 13. Subtract the start time from the end time to get total elapsed time
        duration = end_time - start_time 
        # 14. Log the final duration to the console and the file
        logging.info(f"STOPWATCH: Total execution time: {duration:.2f} seconds")

if __name__ == "__main__": # Standard Python idiom to ensure main() only runs if script is executed directly
    main() # Call the main function
