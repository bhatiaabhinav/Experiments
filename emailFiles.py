#use python 2.7
import smtplib
import time
import sys
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
# from email.MIMEMultipart import MIMEMultipart
# from email.MIMEText import MIMEText
# from email.MIMEImage import MIMEImage

def send_mail(send_from, send_to, subject, text, files=None,
              server="127.0.0.1"):
    assert isinstance(send_to, list)

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(part)


    smtp = smtplib.SMTP('smtp.gmail.com:587')
    smtp.ehlo()
    smtp.starttls()
    smtp.login('bhatiaabhinav93@gmail.com', 'g21@H.pphmf')
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()
    print 'sent'


filenames = sys.argv[1:]

while True:
    try:
        print 'sending ', filenames
        send_mail('bhatiaabhinav93@gmail.com', ['bhatiaabhinav93@gmail.com'], 'TF Logs', 'PFA', filenames)
    except Exception:
        print 'could not send email'
        pass
    time.sleep(30*60)