from flask import Flask, request, jsonify, send_file
from pydub import AudioSegment
import speech_recognition as sr
import openai
import os
import re
import json
import psycopg2
from psycopg2.extras import Json
from flask_cors import CORS
import tempfile
from werkzeug.utils import secure_filename
import logging
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# 在 all.py 中添加根路由處理器
@app.route('/', methods=['GET', 'HEAD'])
def root():
    """處理根路徑請求"""
    return jsonify({
        'status': 'running',
        'message': 'Audio Processing API is running',
        'endpoints': {
            'srt_summary': '/upload/srt-summary',
            'audio_transcription': '/upload/audio-transcription',
            'audio_to_srt': '/upload/audio-to-srt',
            'user_records': '/user-records'
        }
    })

# 添加健康檢查端點
@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

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

# 數據庫配置
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("未設置 DATABASE_URL 環境變數")

# 文件路徑設置
UPLOAD_FOLDER = tempfile.mkdtemp()
OUTPUT_FOLDER = tempfile.mkdtemp()
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

class DatabaseManager:
    def __init__(self):
        self.db_url = DATABASE_URL
        self.init_db()

    def get_connection(self):
        """獲取數據庫連接"""
        return psycopg2.connect(self.db_url)

    def init_db(self):
        """初始化數據庫表"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 創建使用者記錄表
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_records (
                            id SERIAL PRIMARY KEY,
                            email TEXT UNIQUE NOT NULL,
                            first_use TIMESTAMP NOT NULL,
                            last_use TIMESTAMP NOT NULL,
                            srt_summary_count INTEGER DEFAULT 0,
                            transcription_count INTEGER DEFAULT 0,
                            srt_transcription_count INTEGER DEFAULT 0,
                            total_usage INTEGER DEFAULT 0
                        )
                    """)
                    
                    # 創建使用記錄表
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS usage_logs (
                            id SERIAL PRIMARY KEY,
                            email TEXT NOT NULL,
                            service_type TEXT NOT NULL,
                            usage_time TIMESTAMP NOT NULL,
                            status TEXT NOT NULL,
                            file_name TEXT,
                            FOREIGN KEY (email) REFERENCES user_records(email)
                        )
                    """)
                conn.commit()
            logger.info("數據庫表初始化成功")
        except Exception as e:
            logger.error(f"數據庫初始化失敗: {str(e)}")
            raise

    def log_usage(self, email, service_type, status="success", file_name=None):
        """記錄使用情況"""
        try:
            current_time = datetime.now()
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 檢查用戶是否存在
                    cur.execute("""
                        INSERT INTO user_records (email, first_use, last_use, total_usage)
                        VALUES (%s, %s, %s, 1)
                        ON CONFLICT (email) DO UPDATE SET
                            last_use = %s,
                            total_usage = user_records.total_usage + 1,
                            {}_count = user_records.{}_count + 1
                        WHERE user_records.email = %s
                    """.format(
                        service_type.replace('-', '_'),
                        service_type.replace('-', '_')
                    ), (email, current_time, current_time, current_time, email))

                    # 記錄使用日誌
                    cur.execute("""
                        INSERT INTO usage_logs 
                        (email, service_type, usage_time, status, file_name)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (email, service_type, current_time, status, file_name))

                conn.commit()
            logger.info(f"記錄使用情況成功: {email} - {service_type}")
        except Exception as e:
            logger.error(f"記錄使用情況失敗: {str(e)}")
            raise

    def get_statistics(self):
        """獲取使用統計"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 獲取基本統計資料
                    cur.execute("""
                        SELECT 
                            COUNT(DISTINCT email) as total_users,
                            SUM(total_usage) as total_usage,
                            COUNT(DISTINCT CASE 
                                WHEN last_use > NOW() - INTERVAL '30 days' 
                                THEN email 
                            END) as active_users
                        FROM user_records
                    """)
                    stats = cur.fetchone()

                    # 獲取用戶詳細記錄
                    cur.execute("""
                        SELECT 
                            email,
                            first_use,
                            last_use,
                            srt_summary_count,
                            transcription_count,
                            srt_transcription_count,
                            total_usage
                        FROM user_records
                        ORDER BY last_use DESC
                    """)
                    users = []
                    for record in cur.fetchall():
                        users.append({
                            "email": record[0],
                            "firstUse": record[1].isoformat(),
                            "lastUse": record[2].isoformat(),
                            "srtSummaryCount": record[3],
                            "transcriptionCount": record[4],
                            "srtTranscriptionCount": record[5],
                            "totalUsage": record[6]
                        })

                    return {
                        "totalUsers": stats[0],
                        "totalUsage": stats[1] or 0,
                        "activeUsers": stats[2] or 0,
                        "users": users
                    }
        except Exception as e:
            logger.error(f"獲取統計資料失敗: {str(e)}")
            raise

# 初始化數據庫管理器
db_manager = DatabaseManager()

def send_email(recipient_email, subject, body, attachment_path=None):
    """發送郵件函數"""
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                attachment = MIMEApplication(f.read())
                attachment.add_header(
                    'Content-Disposition', 
                    'attachment', 
                    filename=os.path.basename(attachment_path)
                )
                msg.attach(attachment)

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

@app.route('/user-records', methods=['GET'])
def get_user_records():
    """獲取用戶記錄的API端點"""
    try:
        return jsonify(db_manager.get_statistics())
    except Exception as e:
        logger.error(f"獲取用戶記錄失敗: {str(e)}")
        return jsonify({'error': '獲取記錄失敗'}), 500

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
            # 記錄使用者活動
            db_manager.log_usage(email, "srt-summary", file_name=filename)

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
        except Exception as e:
            # 記錄失敗狀態
            db_manager.log_usage(email, "srt-summary", "failed", filename)
            raise
        finally:
            clean_temp_files(file_path)

    except Exception as e:
        logger.error(f"SRT 摘要生成失敗: {str(e)}")
        return jsonify({'error': f"處理失敗: {str(e)}"}), 500

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

        # 記錄使用者活動
        filename = secure_filename(audio_file.filename)
        db_manager.log_usage(email, "transcription", file_name=filename)

        segment_length = int(request.form.get('segmentLength', 30)) * 1000

        # 保存上傳的音頻文件
        audio_path = os.path.join(UPLOAD_FOLDER, filename)
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
            email_subject = f"音頻轉文字結果 - {filename}"
            email_body = f"您的音頻檔案 {filename} 轉文字結果已生成完成，請查看附件。"
            
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

        except Exception as e:
            # 記錄失敗狀態
            db_manager.log_usage(email, "transcription", "failed", filename)
            raise
        finally:
            clean_temp_files(audio_path, temp_segment_path)

    except Exception as e:
        logger.error(f"音頻轉文字失敗: {str(e)}")
        return jsonify({'error': f"處理失敗: {str(e)}"}), 500

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

        # 記錄請求和使用者活動
        filename = secure_filename(audio_file.filename)
        logger.info(f"收到來自 {email} 的音頻轉 SRT 請求")
        db_manager.log_usage(email, "srt-transcription", file_name=filename)

        # 保存音頻文件
        audio_path = os.path.join(UPLOAD_FOLDER, filename)
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
            email_subject = f"音頻轉 SRT 結果 - {filename}"
            email_body = f"您的音頻檔案 {filename} 轉 SRT 結果已生成完成，請查看附件。"
            
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

        except Exception as e:
            # 記錄失敗狀態
            db_manager.log_usage(email, "srt-transcription", "failed", filename)
            raise
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