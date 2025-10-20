import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from email.message import EmailMessage
import smtplib, ssl

user = os.getenv("SMTP_USER")
pwd  = os.getenv("SMTP_PASS")
to   = os.getenv("MANAGER_EMAIL", user)

msg = EmailMessage()
msg["Subject"] = "SMTP smoke test"
msg["From"] = user
msg["To"] = to
msg.set_content("If you see this, SMTP works.")

print("Trying SSL:465…")
try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=30) as s:
        s.set_debuglevel(1 if os.getenv("SMTP_DEBUG")=="1" else 0)
        s.login(user, pwd)
        s.send_message(msg)
    print("✅ Sent via SSL:465")
except Exception as e:
    print("SSL failed:", e)
    print("Trying STARTTLS:587…")
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
        s.set_debuglevel(1 if os.getenv("SMTP_DEBUG")=="1" else 0)
        s.ehlo(); s.starttls(); s.ehlo()
        s.login(user, pwd)
        s.send_message(msg)
    print("✅ Sent via STARTTLS:587")
