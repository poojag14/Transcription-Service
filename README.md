# Transcription-Service

FILE DETAILS
- transcription.py - contains the complete logic of transcription using whisperx (handle_transcription_request function)
- transcription_async.py - calls the handle_transcription_request and sends the result to specific webhook url
- app.py - takes the input json payload and calls process_transcription_async to perform transcription asynchronously
- testwebhook.py - temporary flask application to view the output of our transcription


IMPORTANT
- .env file should consist of llm ky, hugging face token and TRANSCRIPTION_WEBHOOK_URL

STEPS TO RUN
- Run app.py
- Run testwebhook.py
- Lastly, run the curl command sending the input json with audio file to the app.py endpoint (/transcribe)

- You can view the results on the terminal where you ran testwebhook.py.
