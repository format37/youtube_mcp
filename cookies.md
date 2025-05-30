# Extracting Cookies for yt-dlp

To download age-restricted YouTube videos using `yt-dlp`, you need to provide authentication cookies from a browser where you're logged into a YouTube account with access to such content. Follow these steps to extract cookies:

## Prerequisites
- A browser (e.g., Chrome, Firefox, Edge) where you're logged into a YouTube account.
- The `yt-dlp` library installed (`pip install yt-dlp`).
- (Optional) A browser extension like "cookies.txt" for easier cookie export.

## Instructions

1. **Log into YouTube**:
   - Open your browser and log into YouTube with an account that has access to age-restricted videos.

2. **Export Cookies Using a Browser Extension**:
   - Install a cookie export extension:
     - For Chrome: Use "cookies.txt" or "Get cookies.txt LOCALLY" from the Chrome Web Store.
     - For Firefox: Use "Export Cookies" or similar from the Firefox Add-ons site.
   - Navigate to `youtube.com` in your browser.
   - Use the extension to export cookies to a file named `cookies.txt`.
   - Save the `cookies.txt` file in a secure location (e.g., your project directory).

3. **Alternative: Use `yt-dlp` with `--cookies-from-browser`**:
   - If you don't want to manually export cookies, `yt-dlp` can extract them directly from your browser:
     ```bash
     yt-dlp --cookies-from-browser chrome https://youtu.be/<video_id> -o "output.%(ext)s"
     ```
     - Replace `chrome` with your browser (e.g., `firefox`, `edge`, `safari`).
     - Ensure the browser is open and logged into YouTube.

4. **Use Cookies in Your Script**:
   - In your `yt-dlp` script (e.g., `alter_dl.py`), add the `cookiefile` option to the `ydl_opts` dictionary:
     ```python
     ydl_opts = {
         'format': 'bestaudio/best',
         'postprocessors': [{
             'key': 'FFmpegExtractAudio',
             'preferredcodec': 'wav',
         }],
         'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
         'cookiefile': 'path/to/cookies.txt',  # Path to your cookies file
     }
     ```
   - Alternatively, use the `cookiesfrombrowser` option:
     ```python
     ydl_opts['cookiesfrombrowser'] = 'chrome'  # Or 'firefox', etc.
     ```

5. **Run Your Script**:
   - Execute your script with the YouTube URL:
     ```bash
     python alter_dl.py https://youtu.be/<video_id> output
     ```
   - The script should now bypass age restrictions using the provided cookies.

## Security Notes
- **Protect Your Cookies**: The `cookies.txt` file contains sensitive session data. Store it securely and avoid sharing it.
- **Clear Cookies After Use**: If you no longer need the cookies, delete the `cookies.txt` file or revoke the session from your YouTube account.

## Troubleshooting
- If the cookies don't work, ensure you're logged into YouTube in the browser and that the account has access to age-restricted content.
- Update `yt-dlp` to the latest version:
  ```bash
  pip install --upgrade yt-dlp
  ```
- Check the `yt-dlp` documentation for more details: [yt-dlp FAQ](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) and [Exporting YouTube Cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies).