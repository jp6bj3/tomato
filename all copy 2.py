from flask import Flask, request, jsonify, send_file
from pydub import AudioSegment
import speech_recognition as sr
import openai
import os
import re
from flask_cors import CORS
import tempfile
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://jp6bj3.github.io"}})

# 配置環境變數
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("未設置 OPENAI_API_KEY 環境變數")

# 郵件配置
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = os.getenv("EMAIL_USERNAME")
SMTP_PASSWORD = os.getenv("EMAIL_PASSWORD")

if not SMTP_USERNAME or not SMTP_PASSWORD:
    raise ValueError("未設置郵件相關環境變數")

# 文件路徑設置
UPLOAD_FOLDER = tempfile.mkdtemp()
OUTPUT_FOLDER = tempfile.mkdtemp()
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

def send_email(recipient_email, subject, body, attachment_path=None):
    """
    發送郵件函數
    :param recipient_email: 收件人郵箱
    :param subject: 郵件主題
    :param body: 郵件內容
    :param attachment_path: 附件路徑（可選）
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # 添加郵件正文
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # 如果有附件，添加附件
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                attachment = MIMEApplication(f.read())
                attachment.add_header(
                    'Content-Disposition', 
                    'attachment', 
                    filename=os.path.basename(attachment_path)
                )
                msg.attach(attachment)

        # 連接到 SMTP 服務器並發送郵件
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"郵件已發送至 {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"發送郵件失敗: {str(e)}")
        return False

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def clean_temp_files(*file_paths):
    """清理臨時文件"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"清理臨時文件失敗 {file_path}: {e}")

#SRT摘要生成 
@app.route('/upload/srt-summary', methods=['POST'])
def srt_summary():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未提供檔案'}), 400

        email = request.form.get('email')
        if not email:
            return jsonify({'error': '未提供電子郵件地址'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未選擇檔案'}), 400

        if not file.filename.endswith('.srt'):
            return jsonify({'error': '檔案格式錯誤，請上傳 SRT 檔案'}), 400

        # 記錄請求
        logger.info(f"收到來自 {email} 的 SRT 摘要請求")

        # 使用安全的文件名
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            full_text = read_and_clean_srt(file_path)
            structured_summaries = process_text_chunks(full_text)
            final_summary = generate_final_summary(structured_summaries)

            output_file_path = os.path.join(OUTPUT_FOLDER, 'srt_summary.txt')
            with open(output_file_path, "w", encoding="utf-8") as summary_file:
                summary_file.write(final_summary)

            # 發送郵件
            email_subject = f"SRT 檔案摘要結果 - {filename}"
            email_body = f"您的 SRT 檔案 {filename} 摘要已生成完成，請查看附件。"
            
            if send_email(email, email_subject, email_body, output_file_path):
                logger.info(f"摘要結果已發送至 {email}")
            else:
                logger.error(f"發送摘要結果至 {email} 失敗")

            return send_file(
                output_file_path,
                as_attachment=True,
                mimetype='text/plain',
                download_name='srt_summary.txt'
            )
        finally:
            clean_temp_files(file_path)

    except Exception as e:
        logger.error(f"SRT 摘要生成失敗: {str(e)}")
        return jsonify({'error': f"處理失敗: {str(e)}"}), 500

def read_and_clean_srt(file_path):
    """讀取並清理 SRT 文件內容"""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    text_segments = re.findall(
        r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.*?)\n\n",
        content, re.DOTALL
    )
    full_text = " ".join(text_segments).replace("\n", " ")
    full_text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\s]", "", full_text)
    return re.sub(r"\s+", " ", full_text).strip()

def process_text_chunks(full_text, max_chunk_length=1500):
    """處理文本塊並生成摘要"""
    chunks = [full_text[i:i + max_chunk_length] 
             for i in range(0, len(full_text), max_chunk_length)]
    
    structured_summaries = []
    for chunk in chunks:
        try:
            title, summary = generate_summary_and_title(chunk)
            structured_summaries.append((title, summary))
        except Exception as e:
            logger.error(f"處理文本塊失敗: {e}")
            raise

    return structured_summaries

def generate_summary_and_title(text, max_tokens=300):
    """使用 GPT-3.5 生成摘要和標題"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一個專業的摘要生成助手。"},
            {"role": "user", "content": 
             f"請為以下文本生成摘要和主題名稱：\n\n{text}\n\n"
             "請以'主題：'開頭提供主題名稱，並以'摘要：'開頭提供摘要。"}
        ],
        max_tokens=max_tokens,
        temperature=0.3
    )
    content = response['choices'][0]['message']['content'].strip()
    
    title_match = re.search(r"主題：(.*?)\n", content)
    summary_match = re.search(r"摘要：(.*)", content, re.DOTALL)
    
    title = title_match.group(1).strip() if title_match else "未命名主題"
    summary = summary_match.group(1).strip() if summary_match else "無法生成摘要"
    
    return title, summary

#音訊轉文字
@app.route('/upload/audio-transcription', methods=['POST'])
def audio_transcription():
    try:
        if 'audioFile' not in request.files:
            return jsonify({'error': '未提供音頻文件'}), 400

        email = request.form.get('email')
        if not email:
            return jsonify({'error': '未提供電子郵件地址'}), 400

        audio_file = request.files['audioFile']
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': '不支援的音頻格式'}), 400

        # 記錄請求
        logger.info(f"收到來自 {email} 的音頻轉文字請求")

        segment_length = int(request.form.get('segmentLength', 30)) * 1000

        # 保存上傳的音頻文件
        audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
        audio_file.save(audio_path)

        try:
            recognizer = sr.Recognizer()
            audio = AudioSegment.from_file(audio_path)

            output_file_path = os.path.join(OUTPUT_FOLDER, 'transcription_results.txt')
            temp_segment_path = os.path.join(UPLOAD_FOLDER, 'temp_segment.wav')

            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for i in range(0, len(audio), segment_length):
                    segment = audio[i:i + segment_length]
                    segment.export(temp_segment_path, format="wav")

                    try:
                        with sr.AudioFile(temp_segment_path) as source:
                            audio_data = recognizer.record(source)
                            text = recognizer.recognize_google(audio_data, language="zh-TW")
                            output_file.write(f"片段 {i // segment_length + 1}: {text}\n")
                    except sr.UnknownValueError:
                        output_file.write(f"片段 {i // segment_length + 1}: [無法識別的音頻]\n")
                    except sr.RequestError as e:
                        logger.error(f"Google Speech Recognition 服務錯誤: {e}")
                        output_file.write(f"片段 {i // segment_length + 1}: [服務錯誤]\n")

            # 發送郵件
            email_subject = f"音頻轉文字結果 - {audio_file.filename}"
            email_body = f"您的音頻檔案 {audio_file.filename} 轉文字結果已生成完成，請查看附件。"
            
            if send_email(email, email_subject, email_body, output_file_path):
                logger.info(f"轉文字結果已發送至 {email}")
            else:
                logger.error(f"發送轉文字結果至 {email} 失敗")

            return send_file(
                output_file_path,
                as_attachment=True,
                mimetype='text/plain',
                download_name='transcription_results.txt'
            )

        finally:
            clean_temp_files(audio_path, temp_segment_path)

    except Exception as e:
        logger.error(f"音頻轉文字失敗: {str(e)}")
        return jsonify({'error': f"處理失敗: {str(e)}"}), 500

#音訊轉SRT
@app.route('/upload/audio-to-srt', methods=['POST'])
def audio_to_srt():
    try:
        if 'audioFile' not in request.files:
            return jsonify({'error': '未提供音頻文件'}), 400

        email = request.form.get('email')
        if not email:
            return jsonify({'error': '未提供電子郵件地址'}), 400

        audio_file = request.files['audioFile']
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': '不支援的音頻格式'}), 400

        logger.info(f"收到來自 {email} 的音頻轉 SRT 請求")

        # 保存音頻文件
        audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
        audio_file.save(audio_path)

        try:
            with open(audio_path, "rb") as f:
                response = openai.Audio.transcribe("whisper-1", f, response_format="verbose_json")

            output_file_path = os.path.join(OUTPUT_FOLDER, "transcription_results.srt")
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for i, segment in enumerate(response['segments'], start=1):
                    start_time = ms_to_srt_time(int(segment['start'] * 1000))
                    end_time = ms_to_srt_time(int(segment['end'] * 1000))
                    text = segment['text'].strip()
                    
                    output_file.write(f"{i}\n")
                    output_file.write(f"{start_time} --> {end_time}\n")
                    output_file.write(f"{text}\n\n")

            # 發送郵件
            email_subject = f"音頻轉 SRT 結果 - {audio_file.filename}"
            email_body = f"您的音頻檔案 {audio_file.filename} 轉 SRT 結果已生成完成，請查看附件。"
            
            if send_email(email, email_subject, email_body, output_file_path):
                logger.info(f"SRT 結果已發送至 {email}")
            else:
                logger.error(f"發送 SRT 結果至 {email} 失敗")

            return send_file(
                output_file_path,
                as_attachment=True,
                mimetype='text/plain',
                download_name='transcription_results.srt'
            )

        finally:
            clean_temp_files(audio_path)

    except Exception as e:
        logger.error(f"音頻轉 SRT 失敗: {str(e)}")
        return jsonify({'error': f"處理失敗: {str(e)}"}), 500


def ms_to_srt_time(ms):
    """將毫秒轉換為 SRT 時間格式"""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_final_summary(structured_summaries):
    """生成最終的結構化摘要"""
    final_summary = "### 結構化摘要\n\n"
    for idx, (title, summary) in enumerate(structured_summaries, 1):
        final_summary += f"{idx}. **{title}**\n   {summary}\n\n"
    return final_summary

@app.errorhandler(Exception)
def handle_error(error):
    """全域錯誤處理器"""
    logger.error(f"未捕獲的錯誤: {str(error)}")
    return jsonify({'error': '伺服器內部錯誤'}), 500

if __name__ == '__main__':
    # 確保目錄存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 設置日誌文件
    file_handler = logging.FileHandler(
        filename=f'app_{datetime.now().strftime("%Y%m%d")}.log',
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    app.logger.addHandler(file_handler)
    
    app.run(host='0.0.0.0', port=5000)