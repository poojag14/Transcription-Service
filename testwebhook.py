from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Configure logging for the webhook receiver
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/webhook', methods=['POST'])
def receive_webhook():
    """
    Endpoint to receive webhook POST requests.
    It logs the received payload.
    """
    try:
        payload = request.get_json()
        if payload:
            logging.info(f"WEBHOOK RECEIVED!")
            logging.info(f"Payload: {payload}")
            return jsonify({"status": "success", "message": "Webhook received"}), 200
        else:
            logging.warning("WEBHOOK RECEIVED: No JSON payload.")
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400
    except Exception as e:
        logging.error(f"Error processing webhook: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Run on a different port than your main app.py (e.g., 5001)
    app.run(debug=True, port=5001, host="0.0.0.0")