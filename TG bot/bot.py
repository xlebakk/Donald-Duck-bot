
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from llm_client import get_llm_client

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


history: dict[int, list[dict]] = defaultdict(list)


def trim_history(chat_id: int) -> None:
    """
    Безопасно обрезает историю до MAX_HISTORY сообщений,
    гарантированно сохраняя системный промпт на первой позиции.
    """
    h = history[chat_id]
    if not h:
        return

    has_system = h[0].get("role") == "system"
    max_allowed = config.MAX_HISTORY + (1 if has_system else 0)

    if len(h) > max_allowed:
        if has_system:
            # Оставляем системный промпт + последние N сообщений
            history[chat_id] = [h[0]] + h[-config.MAX_HISTORY :]
        else:
            history[chat_id] = h[-config.MAX_HISTORY :]



async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    name = update.effective_user.first_name or "друг"
    text = (
        f"👋 Привет, {name}!\n\n"
        f"Я AI-чат-бот, работаю через *{config.LLM_BACKEND.upper()}* "
        f"with модель `{_current_model()}`.\n\n"
        "Просто напиши мне что-нибудь — и я отвечу.\n\n"
        " Команды:\n"
        "/clear — очистить историю диалога\n"
        "/model — текущая модель\n"
        "/help  — справка"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    chat_id = update.effective_chat.id
    history[chat_id].clear()
    await update.message.reply_text(" История очищена. Начинаем с чистого листа!")


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    text = (
        f" *Бэкенд:* `{config.LLM_BACKEND}`\n"
        f" *Модель:* `{_current_model()}`\n"
        f" *Стриминг:* {'да' if config.STREAM else 'нет'}\n"
        f" *Макс. история:* {config.MAX_HISTORY} сообщений"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    text = (
        " *Справка*\n\n"
        "Просто пиши текст — бот ответит с помощью локальной LLM.\n\n"
        "*Команды:*\n"
        "/start — приветствие\n"
        "/clear — сбросить историю диалога\n"
        "/model — информация о модели\n"
        "/help  — эта справка\n\n"
        "*Как сменить модель?*\n"
        "Измени `LLM_BACKEND` и настройки моделей в файле `.env`."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    if not history[chat_id] and getattr(config, "SYSTEM_PROMPT", None):
        history[chat_id].append({"role": "system", "content": config.SYSTEM_PROMPT})

    history[chat_id].append({"role": "user", "content": user_text})
    trim_history(chat_id)

    await update.message.chat.send_action(ChatAction.TYPING)
    llm = get_llm_client()

    try:
        if config.STREAM:
            await _stream_reply(update, chat_id, llm)
        else:
            await _simple_reply(update, chat_id, llm)
    except Exception as exc:
        logger.exception("Ошибка при обращении к LLM: %s", exc)
        
        if history[chat_id] and history[chat_id][-1]["role"] == "user":
            history[chat_id].pop()
            
        await update.message.reply_text(
            f" Ошибка подключения к {config.LLM_BACKEND.upper()}.\n"
            f"Убедись, что локальный сервер запущен.\n\n`{exc}`",
            parse_mode=ParseMode.MARKDOWN,
        )


async def _simple_reply(update: Update, chat_id: int, llm) -> None:
    """Обычный запрос с ожиданием полного ответа."""
    answer = await llm.chat(history[chat_id])
    if not answer:
        answer = "Пустой ответ от модели."
        
    history[chat_id].append({"role": "assistant", "content": answer})
    trim_history(chat_id)
    await update.message.reply_text(answer)


async def _stream_reply(update: Update, chat_id: int, llm) -> None:
    """Стриминг ответа с оптимизированной частотой обновления экрана."""
    placeholder = await update.message.reply_text("⏳ Думаю...")

    buffer: list[str] = []
    last_edit = time.monotonic()
    EDIT_INTERVAL = 0.6  # Оптимальный интервал для баланса скорости и лимитов TG

    async for chunk in llm.chat_stream(history[chat_id]):
        buffer.append(chunk)
        now = time.monotonic()
        
        if now - last_edit >= EDIT_INTERVAL:
            current_text = "".join(buffer).strip()
            if current_text:
                try:
                    await placeholder.edit_text(current_text)
                except BadRequest:
                    pass
                except Exception as e:
                    logger.debug("Ошибка редактирования стрима: %s", e)
            last_edit = now

    full_answer = "".join(buffer).strip()
    if not full_answer:
        full_answer = "Пустой ответ от модели."

    try:
        await placeholder.edit_text(full_answer)
    except BadRequest:
        pass

    history[chat_id].append({"role": "assistant", "content": full_answer})
    trim_history(chat_id)


def _current_model() -> str:
    if config.LLM_BACKEND.lower() == "ollama":
        return config.OLLAMA_MODEL
    return config.LMSTUDIO_MODEL


async def _run_application() -> None:
    """Асинхронная функция для инициализации и запуска бота."""
    app = Application.builder().token(config.TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот готов к обработке сообщений. Для остановки нажмите Ctrl+C.")
    
    await app.initialize()
    await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    await app.start()
    
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Остановка бота...")
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


def main() -> None:
    if not config.TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN не задан! Добавь его в файл .env")

    logger.info(
        "Запуск бота | бэкенд=%s | модель=%s | стриминг=%s",
        config.LLM_BACKEND,
        _current_model(),
        config.STREAM,
    )
    
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(_run_application())


if __name__ == "__main__":
    main()