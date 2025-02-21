from fastapi import FastAPI, Request, HTTPException, Query, Depends
import os
import uvicorn
import google.generativeai as generative_ai
from typing import Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_bolt import App
import json

app = FastAPI(title="Gemini Question Answering API")

# Environment variables
slack_token = os.environ.get("SLACK_BOT_TOKEN")
slack_signing_secret = os.environ.get("SLACK_SIGNING_SECRET")  # For event verification
generative_ai.configure(api_key=os.environ.get("CHAT_BOT_TOKEN"))
channel = os.environ.get("CHANNEL")

# Initialize Slack app and handler
slack_app = App(token=slack_token, signing_secret=slack_signing_secret)
handler = SlackRequestHandler(app=slack_app)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/slack/events")
async def slack_events(req: Request):
    """Handles Slack events, including message events."""
    return await handler.handle(req)

@app.post("/send_slack_message")
async def send_slack_message(channel_id: str = Query(..., title="Channel ID"), message_text: str = Query(..., title="Message Text")):
    """Sends a message to a specified Slack channel."""
    try:
        await slack_app.client.chat_postMessage(
            channel=channel_id,
            text=message_text
        )
        return {"status": "message_sent", "channel": channel_id}
    except SlackApiError as e:
        raise HTTPException(status_code=500, detail=f"Error sending message to Slack: {e}")

@app.post("/gemini_to_slack")
async def gemini_to_slack(channel_id: str = Query(..., title="Channel ID"), question: str = Query(..., title="Question for Gemini")):
    """Questions Gemini and sends the answer to a specified Slack channel."""
    try:
        gemini_answer = await get_gemini_answer(question)
        if gemini_answer:
            try:
                await slack_app.client.chat_postMessage(
                    channel=channel_id,
                    text=gemini_answer
                )
                return {"status": "gemini_answer_sent", "channel": channel_id, "question": question}
            except SlackApiError as e:
                raise HTTPException(status_code=500, detail=f"Error sending message to Slack: {e}")
        else:
            return {"status": "gemini_no_answer", "channel": channel_id, "question": question, "detail": "Gemini could not generate an answer."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Gemini interaction: {e}")




@slack_app.event("message")
async def handle_message_events(ack, client: WebClient, event: dict, logger): # Added ack
    """Handles regular message events (non-mentions)."""
    await ack() # Acknowledge event receipt
    if event.get("subtype") is None or event.get("subtype") != "bot_message":
        text = event.get("text")
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")

        if event.get("channel_type") == "im": # Still handle direct messages in message event
            question = text.strip()  # No need to remove bot mention in DM
            if question:
                answer = await get_gemini_answer(question)
                if answer:
                    try:
                        await client.chat_postMessage(
                            channel=channel_id,
                            text=answer,
                            thread_ts=thread_ts
                        )
                    except SlackApiError as e:
                        print(f"Error sending message to Slack: {e}")
                else:
                    await client.chat_postMessage(
                        channel=channel_id,
                        text="I couldn't generate a response.",
                        thread_ts=thread_ts
                    )


@slack_app.event("app_mention")
async def handle_app_mention_events(ack, client: WebClient, event: dict, logger): # Added ack
    """Handles app_mention events (when the bot is mentioned)."""
    await ack() # Acknowledge event receipt
    logger.info(f"App mention event received: {event}") # Log the event for debugging
    try:
        text = event.get("text")
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")

        question = text.replace(f"<@{slack_app.api_client.auth_test()['user_id']}>", "").strip()

        if question:
            answer = await get_gemini_answer(question)
            if answer:
                try:
                    await client.chat_postMessage(
                        channel=channel_id,
                        text=answer,
                        thread_ts=thread_ts
                    )
                except SlackApiError as e:
                    print(f"Error sending message to Slack: {e}")
            else:
                await client.chat_postMessage(
                    channel=channel_id,
                    text="I couldn't generate a response.",
                    thread_ts=thread_ts
                )

    except Exception as e:
        print(f"Error handling app_mention event: {e}")
        await client.chat_postMessage(
            channel=channel_id,
            text="An error occurred.",
            thread_ts=thread_ts
        )


async def get_gemini_answer(question: str, model: Optional[str] = None, temperature: Optional[float] = 0.0, max_output_tokens: Optional[int] = 512, top_p: Optional[float] = None, top_k: Optional[int] = None):
    """Gets an answer from Gemini."""
    try:
        if model is None:
            model = "gemini-2.0-flash-001"
        gemini_model = generative_ai.GenerativeModel(model)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k

        response = gemini_model.generate_content(contents=[question], generation_config=generation_config)

        if response.candidates:
            answer = response.candidates[0].content.parts[0].text
            return answer
        else:
            return None

    except Exception as e:
        print(f"Error getting Gemini answer: {e}")
        return None


if __name__ == '__main__':
    uvicorn.run(app, port=int(os.environ.get("PORT", 80)), host='0.0.0.0')