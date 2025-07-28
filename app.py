# app.py
#type:ignore
from flask import Flask, request, jsonify
import os
import logging
import concurrent.futures

# --- Imports for async workers ---
# We only need to import the async worker function itself
from transcription_async import process_transcription_async

app = Flask(__name__)
# Initialize the ThreadPoolExecutor globally
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Configure basic logging for the Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- The /transcribe route implementing the 200 OK + Webhook pattern ---
@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    API endpoint to handle transcription requests asynchronously.
    Expects a JSON payload with audio or transcription parameters.
    The webhook URL for results is configured via environment variable.
    Immediately returns a processing receipt response (HTTP 200 OK).
    The actual transcription happens in a background thread, and the result
    is sent to the pre-configured webhook URL upon completion.
    """
    try:
        data = request.get_json()

        # Get the webhook URL from an environment variable
        webhook_url = os.getenv("TRANSCRIPTION_WEBHOOK_URL")
        logging.info(f"Transcription request received. Submitting to background for webhook: {webhook_url}")

        # Submit the background transcription task to the thread pool
        executor.submit(process_transcription_async, webhook_url, data)

        # Immediately return a 200 OK response
        return jsonify({
            "jobId": data.get("jobId"),
            "status": "QUEUED",
        }), 200 # HTTP 200 OK

    except Exception as e:
        logging.error(f"Error in transcribe endpoint: {e}", exc_info=True)
        return jsonify({
            "jobId": data.get("jobId"),
            "status": "FAILED",
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
