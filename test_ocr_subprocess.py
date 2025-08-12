import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_ocr_test(image_path: str):

    logging.info(f"Attempting OCR on image: {image_path}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir 

    ocr_env_python = os.path.join(project_root, '.venv_ocr_craft', 'bin', 'python')

    trocr_script = os.path.join(project_root, 'ocr', 'trocr_craft.py')

    if not os.path.exists(ocr_env_python):
        logging.error(f"Error: OCR Python executable not found at '{ocr_env_python}'. "
                      f"Please ensure the virtual environment is correctly set up and the path is accurate.")
        sys.exit(1)
    if not os.path.exists(trocr_script):
        logging.error(f"Error: TrOCR script not found at '{trocr_script}'. "
                      f"Please ensure the script exists and the path is accurate.")
        sys.exit(1)
    if not os.path.exists(image_path):
        logging.error(f"Error: Input image not found at '{image_path}'. "
                      f"Please ensure the image exists and the path provided is correct.")
        sys.exit(1)

    try:
        logging.info(f"Executing command: {ocr_env_python} {trocr_script} {image_path}")
        
        result = subprocess.run(
            [ocr_env_python, trocr_script, image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        text_content = result.stdout.strip()
        logging.info("OCR subprocess completed successfully.")
        
        logging.info("\n--- Extracted Text (from stdout of trocr_script) ---")
        print(text_content)
        
        logging.info("\n--- Subprocess Stderr (for logs/errors from trocr_script) ---")
        print(result.stderr.strip()) 

    except FileNotFoundError as e:
        logging.error(f"Failed to find executable or script: {e}. "
                      f"Ensure '{ocr_env_python}' and '{trocr_script}' are correct paths.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"OCR script returned error code {e.returncode}. "
                      f"Command: {e.cmd}\n"
                      f"Stderr: {e.stderr.strip()}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during OCR subprocess execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        run_ocr_test(test_image_path)
    else:
        logging.error("Usage: python test_ocr_subprocess.py <path_to_image>")
        sys.exit(1)
