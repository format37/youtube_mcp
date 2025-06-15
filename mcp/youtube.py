import yt_dlp
import os
import shutil
import glob
from fastapi import FastAPI, HTTPException, Request
from mcp.server.fastmcp import FastMCP
import traceback
import logging
from pydub import AudioSegment
import uuid
from openai import OpenAI
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY == '':
    raise Exception("OPENAI_API_KEY environment variable not found")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastMCP server
mcp = FastMCP("youtube")

TRANSCRIPTIONS_CSV = os.path.join("data", "transcriptions.csv")

def get_cached_transcription(url):
    """
    Check if the transcription for the given URL exists in the CSV mapping.
    If found, return the UUID and the transcript text (from uuid.txt).
    Otherwise, return (None, None).
    """
    if not os.path.exists(TRANSCRIPTIONS_CSV):
        return None, None
    try:
        with open(TRANSCRIPTIONS_CSV, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 2 and row[0] == url:
                    uuid_val = row[1]
                    txt_path = os.path.join("data", f"{uuid_val}.txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            return uuid_val, f.read()
    except Exception as e:
        logger.error(f"Error reading cache CSV: {e}")
    return None, None

def save_transcription_cache(url, uuid_val, transcript):
    """
    Save the link-uuid mapping to the CSV and the transcript to uuid.txt.
    """
    txt_path = os.path.join("data", f"{uuid_val}.txt")
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        # Append to CSV
        with open(TRANSCRIPTIONS_CSV, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([url, uuid_val])
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def download_audio(url):
    output_dir = "data"
    
    # Generate a UUID for the filename to avoid filesystem issues with special characters
    unique_filename = str(uuid.uuid4())

    # Check that cookies.txt exists
    if not os.path.exists('cookies.txt'):
        raise Exception("ATTENTION: cookies.txt not found")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': os.path.join(output_dir, f'{unique_filename}.%(ext)s'),
        'cookiefile': 'cookies.txt',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract info and download
        info = ydl.extract_info(url, download=True)
        
        # Modify the info dict to reflect the correct extension after post-processing
        info_with_mp3_ext = dict(info)
        info_with_mp3_ext['ext'] = 'mp3'
        info_with_mp3_ext['title'] = unique_filename  # Use UUID as title for filename generation
        
        # Get the actual filename that will be used
        actual_filename = ydl.prepare_filename(info_with_mp3_ext)
        
        # Log the original video title for reference
        logger.info(f"Downloaded video: '{info.get('title', 'Unknown')}' as '{actual_filename}'")
        
        return actual_filename

def split_audio_ffmpeg(audio_path, chunk_length=10*60):
    """
    Splits the audio file into chunks using ffmpeg.
    Returns a list of paths to the chunks.
    """
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            return []
            
        # Get the duration of the audio in seconds
        cmd_duration = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{audio_path}\""
        logger.info(f"Running duration command: {cmd_duration}")
        duration_output = os.popen(cmd_duration).read().strip()
        
        if not duration_output:
            logger.error(f"Failed to get duration for {audio_path}")
            
            # Try alternative method
            logger.info("Trying alternative method to get duration")
            try:
                audio = AudioSegment.from_file(audio_path)
                duration = len(audio) / 1000.0  # Convert milliseconds to seconds
                logger.info(f"Got duration using AudioSegment: {duration} seconds")
            except Exception as e:
                logger.error(f"Alternative duration method failed: {e}")
                return []
        else:
            duration = float(duration_output)
            logger.info(f"Audio duration: {duration} seconds")
        
        # Calculate number of chunks
        chunks_count = int(duration // chunk_length) + (1 if duration % chunk_length > 0 else 0)
        logger.info(f"Splitting into {chunks_count} chunks")
        
        # For very short files, just return the original
        if chunks_count <= 1 and duration < chunk_length:
            logger.info(f"Audio is short enough, skipping split")
            return [audio_path]
        
        chunk_paths = []
        
        for i in range(chunks_count):
            start_time = i * chunk_length
            remaining_duration = duration - start_time
            # Adjust chunk length for last chunk if needed
            current_chunk_length = min(chunk_length, remaining_duration)
            
            # Unique filename for the chunk
            chunk_filename = f"/tmp/{uuid.uuid4()}.mp3"
            
            # Use ffmpeg to extract a chunk of the audio - with quotes for paths with spaces
            cmd_extract = f'ffmpeg -ss {start_time} -t {current_chunk_length} -i "{audio_path}" -acodec copy "{chunk_filename}" -y'
            logger.info(f"Running command: {cmd_extract}")
            
            result = os.system(cmd_extract)
            if result != 0:
                logger.error(f"Error extracting chunk {i+1} with command: {cmd_extract}")
                
                # Try alternative method with pydub
                try:
                    logger.info(f"Trying alternative method to extract chunk {i+1}")
                    audio = AudioSegment.from_file(audio_path)
                    start_ms = int(start_time * 1000)
                    end_ms = int(min(duration, start_time + current_chunk_length) * 1000)
                    chunk = audio[start_ms:end_ms]
                    chunk.export(chunk_filename, format="mp3")
                    logger.info(f"Successfully extracted chunk using pydub")
                except Exception as e:
                    logger.error(f"Alternative extraction method failed: {e}")
                    continue
                
            # Verify the chunk file exists and has a non-zero size
            if os.path.exists(chunk_filename) and os.path.getsize(chunk_filename) > 0:
                chunk_paths.append(chunk_filename)
                logger.info(f"Successfully created chunk {i+1}: {chunk_filename}")
            else:
                logger.error(f"Chunk file {chunk_filename} does not exist or is empty")
        
        logger.info(f"Successfully created {len(chunk_paths)} chunks")
        return chunk_paths
        
    except Exception as e:
        logger.error(f"Error in split_audio_ffmpeg: {str(e)}")
        return []
    
def transcribe_chunk(audio_path):
    # user_lang = 'en'
    
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            logger.error(f"Chunk file not found: {audio_path}")
            return "\n[Transcription error: File not found]\n"
            
        # Log chunk file size
        file_size = os.path.getsize(audio_path)
        logger.info(f"Transcribing chunk of size: {file_size} bytes")
        
        # Check if file is empty
        if file_size == 0:
            logger.error(f"Chunk file is empty: {audio_path}")
            return "\n[Transcription error: Empty file]\n"
            
        with open(audio_path, "rb") as audio_file:
            # Add timeout parameters to avoid hanging
            # logger.info(f"Sending chunk to OpenAI API with language: {user_lang}")
            logger.info(f"Sending chunk to OpenAI API")
            response = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                temperature=0,
                response_format="text",
                # language=user_lang,
                timeout=600  # 10 minute timeout
            )
            
        logger.info(f"Received transcription of length: {len(response)}")
        return response
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error transcribing chunk: {error_message}")
        
        # More specific error messages based on the exception
        if "timed out" in error_message.lower():
            return "\n[Transcription error: API request timed out. The chunk may be too large.]\n"
        elif "too large" in error_message.lower():
            return "\n[Transcription error: File too large for the API.]\n"
        else:
            # Return empty string to avoid breaking the whole process
            return f"\n[Transcription error: {error_message}]\n"

def split_text_by_symbols(text: str, chunk_size: int = 100000) -> list:
    """
    Split text into chunks of specified size (default 100k symbols).
    
    Args:
        text (str): The text to split
        chunk_size (int): Maximum number of characters per chunk (default 100000)
    
    Returns:
        list: List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    text_length = len(text)
    
    for i in range(0, text_length, chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info(f"Split text of {text_length} characters into {len(chunks)} chunks")
    return chunks

def transcribe_audio(audio_path):
    try:
        # Split the audio into chunks
        chunk_paths = split_audio_ffmpeg(audio_path)
        
        if not chunk_paths:
            error_msg = "Failed to split audio file into chunks"
            logger.error(f"{error_msg}")
            return error_msg

        full_text = ""
        
        for idx, chunk_path in enumerate(chunk_paths):
            try:
                logger.info(f"Processing chunk {idx+1} of {len(chunk_paths)}")
                
                # Transcribe chunk
                text = transcribe_chunk(chunk_path)
                full_text += text
                
            except Exception as chunk_error:
                logger.error(f"Error processing chunk {idx+1}: {str(chunk_error)}")
                full_text += f"\n[Error in chunk {idx+1}]\n"
            finally:
                # Ensure chunk file is removed even if there's an error
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        return full_text
        
    except Exception as e:
        error_msg = f"Error in transcription process: {str(e)}"
        logger.error(f"{error_msg}")
        return error_msg

@mcp.tool()
async def transcribe_youtube(url: str) -> list:
    """
    Transcribe a YouTube video and return the transcript.

    Args:
        url (str): The URL of the YouTube video to transcribe.

    Returns:
        str: The transcript of the video.
    """
    try:
        # Check cache first
        cached_uuid, cached_transcript = get_cached_transcription(url)
        if cached_transcript is not None:
            logger.info(f"Cache hit for {url} (uuid={cached_uuid})")
            chunks = split_text_by_symbols(cached_transcript)
            return chunks

        # Download the video
        file_path = download_audio(url)
        logger.info(f"Downloaded video to {file_path}")

        # Extract uuid from file_path
        uuid_val = os.path.splitext(os.path.basename(file_path))[0]

        # Transcribe the audio
        transcript = transcribe_audio(file_path)

        # Save to cache
        save_transcription_cache(url, uuid_val, transcript)

        chunks = split_text_by_symbols(transcript)

        return chunks
    except Exception as e:
        logger.error(f"Error transcribing video: {e}")
        return traceback.format_exc()
    
app = FastAPI()

@app.get("/test")
async def test_endpoint():
    """
    Test endpoint to verify the server is running.
    
    Returns:
        dict: A simple response indicating the server status.
    """
    return {
        "status": "ok",
        "message": "YouTube MCP server is running",
        "endpoints": {
            "transcribe": "/transcribe_youtube (MCP tool)",
            "test": "/test"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Clean up data folder on server start"""
    data_folder = "data"
    
    # Create data folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        logger.info(f"Created data folder: {data_folder}")
    else:
        # Remove all files in the data folder
        files = glob.glob(os.path.join(data_folder, "*"))
        for file_path in files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.info(f"Removed directory: {file_path}")
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")
        
        logger.info(f"Cleaned up data folder: {data_folder}")

# @app.middleware("http")
# async def validate_api_key(request: Request, call_next):
#     auth_header = request.headers.get("Authorization")
#     API_KEY = os.environ.get("MCP_KEY")
#     if auth_header != API_KEY:
#         raise HTTPException(status_code=401, detail="Invalid API key")
#     return await call_next(request)

app.mount("/", mcp.sse_app())
