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
# Initialize the ProcessPoolExecutor globally
executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)

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
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        # Get the webhook URL from an environment variable
        webhook_url = os.getenv("TRANSCRIPTION_WEBHOOK_URL")

        if not webhook_url:
            logging.error("TRANSCRIPTION_WEBHOOK_URL environment variable is not set.")
            return jsonify({
                "status": "error",
                "message": "Server not configured for transcription results. TRANSCRIPTION_WEBHOOK_URL is missing."
            }), 500 # Use 500 as it's a server-side configuration error

        logging.info(f"Transcription request received. Submitting to background for webhook: {webhook_url}")

        # Submit the background transcription task to the thread pool
        # app.py only needs to know about process_transcription_async
        executor.submit(process_transcription_async, webhook_url, data)

        # Immediately return a 200 OK response
        return jsonify({
            "status": "accepted",
            "message": "Transcription request received and processing has begun. Results will be sent to the configured webhook.",
            "webhook_configured_at": webhook_url
        }), 200 # HTTP 200 OK

    except Exception as e:
        logging.error(f"Error in transcribe endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")