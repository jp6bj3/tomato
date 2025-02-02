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

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://jp6bj3.github.io"}})

# 配置 OpenAI API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("未設置 OPENAI_API_KEY 環境變數")

# 文件路徑設置
UPLOAD_FOLDER = tempfile.mkdtemp()  # 使用臨時目錄
OUTPUT_FOLDER = tempfile.mkdtemp()  # 使用臨時目錄
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

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

@app.route('/upload/srt-summary', methods=['POST'])
def srt_summary():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未提供檔案'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未選擇檔案'}), 400

        if not file.filename.endswith('.srt'):
            return jsonify({'error': '檔案格式錯誤，請上傳 SRT 檔案'}), 400

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

@app.route('/upload/audio-transcription', methods=['POST'])
def audio_transcription():
    try:
        if 'audioFile' not in request.files:
            return jsonify({'error': '未提供音頻文件'}), 400

        audio_file = request.files['audioFile']
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': '不支援的音頻格式'}), 400

        segment_length = int(request.form.get('segmentLength', 30)) * 1000

        # 保存上傳的音頻文件
        audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
        audio_file.save(audio_path)

        try:
            # 初始化語音識別器
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

@app.route('/upload/audio-to-srt', methods=['POST'])
def audio_to_srt():
    try:
        if 'audioFile' not in request.files:
            return jsonify({'error': '未提供音頻文件'}), 400

        audio_file = request.files['audioFile']
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': '不支援的音頻格式'}), 400

        # 保存上傳的音頻文件
        audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
        audio_file.save(audio_path)

        try:
            segment_length = 120 * 1000  # 2分鐘
            audio = AudioSegment.from_file(audio_path)

            output_file_path = os.path.join(OUTPUT_FOLDER, "transcription_results.srt")
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for i in range(0, len(audio), segment_length):
                    segment = audio[i:i + segment_length]
                    segment_count = i // segment_length + 1
                    
                    # 保存臨時音頻段
                    segment_path = os.path.join(UPLOAD_FOLDER, f"segment_{segment_count}.wav")
                    segment.export(segment_path, format="wav")

                    try:
                        # 使用 Whisper 進行轉錄
                        with open(segment_path, "rb") as f:
                            response = openai.Audio.transcribe("whisper-1", f)
                            text = response['text']

                        # 計算時間戳
                        start_time = i
                        end_time = i + len(segment)
                        
                        # 寫入 SRT 格式
                        output_file.write(f"{segment_count}\n")
                        output_file.write(f"{ms_to_srt_time(start_time)} --> {ms_to_srt_time(end_time)}\n")
                        output_file.write(f"{text}\n\n")

                    finally:
                        clean_temp_files(segment_path)

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