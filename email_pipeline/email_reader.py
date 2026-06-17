import imaplib
import email
from email.header import decode_header
from .config import EMAIL_ADDRESS, EMAIL_APP_PASSWORD


def read_latest_email():

    mail = imaplib.IMAP4_SSL("imap.gmail.com")

    mail.login(
        EMAIL_ADDRESS,
        EMAIL_APP_PASSWORD
    )

    mail.select("inbox")

    _, messages = mail.search(None, "UNSEEN")

    ids = messages[0].split()

    if not ids:
        return None

    latest_id = ids[-1]

    _, msg_data = mail.fetch(latest_id, "(RFC822)")

    raw_email = msg_data[0][1]

    message = email.message_from_bytes(raw_email)

    body = ""

    if message.is_multipart():
        for part in message.walk():

            content_type = part.get_content_type()

            if content_type == "text/plain":
                body = part.get_payload(
                    decode=True
                ).decode(
                    errors="ignore"
                )
                break
    else:
        body = message.get_payload(
            decode=True
        ).decode(
            errors="ignore"
        )

    mail.logout()

    return body