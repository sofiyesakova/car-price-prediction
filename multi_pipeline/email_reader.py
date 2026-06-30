import imaplib
import email
from datetime import datetime
from .config import EMAIL_ADDRESS, EMAIL_APP_PASSWORD


def read_latest_email():

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)

    mail.select("inbox")

    result, data = mail.search(None, "UNSEEN")

    mail_ids = data[0].split()

    if not mail_ids:
        return None

    latest_id = mail_ids[-1]

    result, msg_data = mail.fetch(latest_id, "(RFC822)")

    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
                break
    else:
        body = msg.get_payload(decode=True).decode()

    # ==================================================
    # EXTRACT DATE
    # ==================================================

    date_tuple = email.utils.parsedate_tz(msg["Date"])
    received_datetime = datetime.fromtimestamp(
        email.utils.mktime_tz(date_tuple)
    )

    # ==================================================
    # RETURN STRUCTURED OBJECT
    # ==================================================

    return {
        "body": body,
        "received_datetime": received_datetime
    }