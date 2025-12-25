from collections import defaultdict
from datetime import datetime, timedelta

import telethon
import telethon.tl.types
from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
import asyncio
from ezmm import Image

from config.globals import api_keys
from config.globals import telegram_channels, temp_dir
from defame.evidence_retrieval.integrations import SocialMediaPost
from defame.evidence_retrieval.integrations.social_media.common import SocialMediaPostMetadata


class Telegram:
    """
    The Telegram API Integration. See https://docs.telethon.dev/ for more information.
    """
    name = "Telegram"
    session_name = f"{temp_dir}/Telegram_Session"
    phone_number = ""
    channels = []

    def __init__(self):
        self.api_id = api_keys["telegram_api_id"]
        self.api_hash = api_keys["telegram_api_hash"]
        self.phone_number = api_keys["telegram_phone"] # to mimic real human behavior and access channels AND chats, a phone number is required
        self.client = None
        self.channels = telegram_channels

    def keyword_search(self, keywords:str, start_time:datetime, timeframe:int=60, post_limit: int = 50):
        """
        Retrieves posts from Telegram using keyword search. Requires the Channels to be set in config
        :param keywords: Keywords to search for.
        :param post_limit: Number of posts to retrieve.
        :param start_time: Start time of search in UTC.
        :param timeframe: Search timeframe in seconds.
        """
        async def wrapper():
            if self.client is None or self.client.loop.is_closed():
                self.client = TelegramClient(
                    session=self.session_name,
                    api_id=self.api_id,
                    api_hash=self.api_hash,
                    timeout=60,
                )
            await self.client.start(phone=self.phone_number)
            try:
                return await self._async_keyword_search(keywords, start_time, timeframe, post_limit)
            finally:
                await self.client.disconnect()

        return asyncio.run(wrapper())


    async def _async_keyword_search(self, keywords:str, start_time:datetime, timeframe:int, post_limit: int):
        """
        Retrieves posts from Telegram using keyword search. Requires the Channels to be set in config
        :param keywords: Keywords to search for.
        :param post_limit: Number of posts to retrieve.
        :param start_time: Start time of search in UTC.
        :param timeframe: Search timeframe in seconds.
        """
        matching_posts = []

        channels_found = []
        for c in self.channels:
            try:
                await self.client(JoinChannelRequest(c))
                channels_found.append(c)
            except Exception as e:
                pass
        if len(channels_found) == 0:
            raise Exception(f"No channels found for for Telegram given {self.channels}. Found: {channels_found}")

        end_time = start_time + timedelta(minutes=timeframe)

        # Messages can be made up of individual sub-messages (especially for multiple images)
        # Therefor they need to be grouped first
        grouped_messages = defaultdict(list)
        for c in channels_found:
            async for message in self.client.iter_messages(
                c, search=keywords, limit=post_limit
            ):
                if message.grouped_id is None:
                    grouped_messages[message.id].append(
                        message
                    )  # it is a message without a group, it gets its own group
                else:
                    # if it is a message with a group, corresponding posts are gathered
                    msgs = await self.client.get_messages(
                        c, min_id=message.id - 10, max_id=message.id + 10
                    )
                    grouped_messages[message.grouped_id] = [
                        m for m in msgs if m.grouped_id == message.grouped_id
                    ]
        groups = sorted(
            grouped_messages.values(), key=lambda msg: msg[0].date, reverse=True
        )[: min(post_limit, len(grouped_messages.values()))]
        groups = list(filter(lambda msg: start_time <= msg[0].date <= end_time, groups))
        groups = groups[: min(post_limit, len(groups))]

        # Create the actual SocialMediaPost objects
        for msg_group in groups:
            images = []
            author = None
            date = None
            text = []
            msg_id = -1
            for message in msg_group:
                if message.text:
                    text.append(message.text)
                    msg_id = message.id
                if message.web_preview and (
                    preview_text := message.web_preview.description
                ):
                    text.append(preview_text)
                if message.date and date is None:
                    date = message.date
                if message.sender.username and author is None:
                    author = message.sender.username if message.sender else None

                if (
                    message.media
                    and (
                        type(message.media) is telethon.tl.types.MessageMediaPhoto
                        or type(message.media) is telethon.tl.types.MessageMediaWebPage
                    )
                ):  # if an image is present, it is downloaded as bytes and added to the mmSequence
                    img_bytes = await self.client.download_media(
                        message.media, file=bytes
                    )  # download image as bytestream
                    images.append(
                        Image(binary_data=img_bytes)
                    )  # create mmImage to use in mmSequence

            post = SocialMediaPost(message=" ".join(text), metadata=SocialMediaPostMetadata(platform="Telegram", author_username=author, post_url=f"https://t.me/{author}/{msg_id}", created_at=date, media=images))
            matching_posts.append(post)

        return matching_posts

