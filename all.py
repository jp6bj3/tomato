from flask import Flask, request, jsonify, send_file
from pydub import AudioSegment
import speech_recognition as sr
import openai
import os
import re
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://jp6bj3.github.io"}})

# 配置 OpenAI API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")

# 文件路徑設置
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/upload/srt-summary', methods=['POST'])
def srt_summary():
    if 'file' not in request.files:
        return jsonify({'error': '未提供檔案'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未選擇檔案'}), 400

    if not file.filename.endswith('.srt'):
        return jsonify({'error': '檔案格式錯誤，請上傳 SRT 檔案'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 讀取 SRT 並生成摘要
    def read_and_clean_srt(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        text_segments = re.findall(r"\\d+\\n\\d{2}:\\d{2}:\\d{2},\\d{3} --> \\d{2}:\\d{2}:\\d{2},\\d{3}\\n(.*?)\\n\\n", content, re.DOTALL)
        full_text = " ".join(text_segments).replace("\\n", " ")
        full_text = re.sub(r"[^a-zA-Z0-9\\u4e00-\\u9fff\\s]", "", full_text)
        return re.sub(r"\\s+", " ", full_text).strip()

    full_text = read_and_clean_srt(file_path)
    title, summary = generate_summary_and_title(full_text)

    output_file_path = os.path.join(OUTPUT_FOLDER, 'srt_summary.txt')
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"主題：{title}\\n摘要：{summary}")

    return send_file(output_file_path, as_attachment=True)

@app.route('/upload/audio-transcription', methods=['POST'])
def audio_transcription():
    if 'audioFile' not in request.files:
        return jsonify({'error': '未提供音頻文件'}), 400

    audio_file = request.files['audioFile']
    segment_length = int(request.form.get('segmentLength', 30)) * 1000

    temp_audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(temp_audio_path)

    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(temp_audio_path)

    output_file_path = os.path.join(OUTPUT_FOLDER, 'transcription_results.txt')
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for i in range(0, len(audio), segment_length):
            segment = audio[i:i + segment_length]
            segment.export("temp_segment.wav", format="wav")

            with sr.AudioFile("temp_segment.wav") as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data, language="zh-TW")
                    output_file.write(f"片段 {i // segment_length + 1}: {text}\\n")
                except sr.UnknownValueError:
                    output_file.write(f"片段 {i // segment_length + 1}: [無法識別的音頻]\\n")

    return send_file(output_file_path, as_attachment=True)

@app.route('/upload/audio-to-srt', methods=['POST'])
def audio_to_srt():
    if 'audioFile' not in request.files:
        return jsonify({'error': '未提供音頻文件'}), 400

    audio_file = request.files['audioFile']
    input_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(input_path)

    segment_length = 120 * 1000
    audio = AudioSegment.from_file(input_path)

    output_file_path = os.path.join(OUTPUT_FOLDER, "transcription_results.srt")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        segment_count = 1
        for i in range(0, len(audio), segment_length):
            segment = audio[i:i + segment_length]
            segment_path = os.path.join(UPLOAD_FOLDER, f"segment_{segment_count}.wav")
            segment.export(segment_path, format="wav")

            with open(segment_path, "rb") as f:
                response = openai.Audio.transcribe("whisper-1", f)
                text = response['text']

            output_file.write(f"{segment_count}\\n")
            output_file.write(f"{ms_to_srt_time(i)} --> {ms_to_srt_time(i + len(segment))}\\n")
            output_file.write(f"{text}\\n\\n")
            segment_count += 1

    return send_file(output_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)