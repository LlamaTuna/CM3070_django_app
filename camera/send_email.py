import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import cv2
from datetime import datetime
from .models import EmailSettings

class SendEmail:
    def __init__(self, request):
        self.request = request
        self.alert_buffer = []
        self.frame_buffer = []

    def log_event(self, event):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {event}"
        self.alert_buffer.append(log_entry)
        print("SendEmail logged event:", log_entry)

    def send_email_snapshot(self):
        print("Attempting to send email snapshot...")
        if not self.alert_buffer:
            print("Alert buffer is empty, no email will be sent.")
            return
        try:
            if self.request:
                email_settings = EmailSettings.objects.get(user=self.request.user)
            else:
                print("Request object is not available.")
                return

            print(f"Email Settings: {email_settings.__dict__}")  # Debug statement

            smtp_server = email_settings.smtp_server
            smtp_port = email_settings.smtp_port
            smtp_user = email_settings.smtp_user
            smtp_password = email_settings.smtp_password
            from_email = smtp_user
            to_email = email_settings.email
            subject = "Motion Detection Alert Snapshot"
            body = "\n".join(self.alert_buffer)
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            selected_frames = self.select_representative_frames(self.frame_buffer, 2)

            for i, frame in enumerate(selected_frames):
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_data = img_encoded.tobytes()
                image = MIMEImage(image_data, name=f"event_{i + 1}.jpg")
                msg.attach(image)

            print("Connecting to SMTP server...")
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            print("Logging into SMTP server...")
            server.login(smtp_user, smtp_password)
            text = msg.as_string()
            print("Sending email...")
            server.sendmail(from_email, to_email, text)
            server.quit()

            self.alert_buffer = []
            self.frame_buffer = []
            print("Email sent successfully")
        except Exception as e:
            print(f"Failed to send snapshot email: {str(e)}")

    def select_representative_frames(self, frames, num_frames):
        if len(frames) <= num_frames:
            return frames
        interval = len(frames) // num_frames
        selected_frames = [frames[i * interval] for i in range(num_frames)]
        return selected_frames
