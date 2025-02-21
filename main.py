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


slack_token = os.environ.get("SLACK_BOT_TOKEN")
slack_signing_secret = os.environ.get("SLACK_SIGNING_SECRET")  # For event verification
generative_ai.configure(api_key=os.environ.get("CHAT_BOT_TOKEN"))
channel = os.environ.get("CHANNEL")


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


@slack_app.event("message")
async def handle_message_events(client: WebClient, event: dict, logger):
    """Handles regular message events (non-mentions)."""
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
async def handle_app_mention_events(client: WebClient, event: dict, logger):
    """Handles app_mention events (when the bot is mentioned)."""
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