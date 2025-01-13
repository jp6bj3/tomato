# -*- coding: utf-8 -*-

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
    try:
        # 檢查是否有上傳的檔案
        if 'file' not in request.files:
            return jsonify({'error': '未提供檔案'}), 400

        file = request.files['file']

        # 檢查檔案名稱是否有效
        if file.filename == '':
            return jsonify({'error': '未選擇檔案'}), 400

        # 檢查檔案是否為 SRT 格式
        if not file.filename.endswith('.srt'):
            return jsonify({'error': '檔案格式錯誤，請上傳 SRT 檔案'}), 400

        # 保存上傳的檔案
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # 讀取並清理 SRT 文件
        def read_and_clean_srt(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                # 提取字幕文字
                text_segments = re.findall(
                    r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.*?)\n\n",
                    content, re.DOTALL
                )
                # 合併文字並清理格式
                full_text = " ".join(text_segments).replace("\n", " ")
                full_text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\s]", "", full_text)
                full_text = re.sub(r"\s+", " ", full_text).strip()
                return full_text
            except Exception as e:
                raise ValueError(f"SRT 文件解析失敗: {e}")


        # 使用 GPT-3.5 Turbo 生成摘要和主題
        def generate_summary_and_title(text, max_tokens=300):
            """使用 GPT-3.5 Turbo 生成摘要和主題標題。"""
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一個專業的摘要生成助手。"},
                    {"role": "user", "content": f"請為以下文本生成摘要和主題名稱：\n\n{text}\n\n請以'主題：'開頭提供主題名稱，並以'摘要：'開頭提供摘要。"}
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
        
        full_text = read_and_clean_srt(file_path)


        # 分段處理
        max_chunk_length = 1500  # 每段最大長度
        chunks = [full_text[i:i + max_chunk_length] for i in range(0, len(full_text), max_chunk_length)]

        structured_summaries = []
        for idx, chunk in enumerate(chunks):
            try:
                title, summary = generate_summary_and_title(chunk)
                structured_summaries.append((title, summary))
            except Exception as e:
                return jsonify({'error': f'處理第 {idx + 1} 段時發生錯誤: {str(e)}'}), 500

        # 動態生成結構化摘要
        final_summary = "### 結構化摘要\n\n"
        for idx, (title, summary) in enumerate(structured_summaries):
            final_summary += f"{idx + 1}. **{title}**\n   {summary}\n\n"

        # 儲存摘要到檔案
        output_file_path = os.path.join(OUTPUT_FOLDER, 'srt_summary.txt')
        with open(output_file_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(final_summary)

        # 返回結果檔案
        return send_file(
            output_file_path,
            as_attachment=True,
            mimetype='text/plain',
            download_name='srt_summary.txt'
        )

    except ValueError as ve:
        # 捕獲 SRT 解析相關錯誤
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        # 捕獲其他未預期錯誤
        return jsonify({'error': f"伺服器內部錯誤: {e}"}), 500

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
             # 計算片段的開始和結束時間
            start_time = i
            end_time = i + len(segment)

            # 轉換為 SRT 格式時間字符串
            def ms_to_srt_time(ms):
                seconds, milliseconds = divmod(ms, 1000)
                minutes, seconds = divmod(seconds, 60)
                hours, minutes = divmod(minutes, 60)
                return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

            srt_start_time = ms_to_srt_time(start_time)
            srt_end_time = ms_to_srt_time(end_time)

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