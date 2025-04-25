#!/usr/bin/env python
"""
Script to download a single Sentinel-1 file from ASF.

Usage:
    python download_sentinel_file.py <url> [--output_dir DIR]

Example:
    python download_sentinel_file.py https://datapool.asf.alaska.edu/GRD_HD/SA/S1A_IW_GRDH_1SDV_20230427T154057_20230427T154122_048284_05CE71_87D9.zip --output_dir ../data/raw/sentinel
"""

import os
import sys
import argparse
import base64
import getpass
import re
import signal
import ssl
import tempfile
import time
from http.cookiejar import MozillaCookieJar
from urllib.error import HTTPError, URLError
from urllib.request import (HTTPCookieProcessor, HTTPHandler, HTTPSHandler,
                           Request, build_opener, install_opener, urlopen)

# Global variables
abort = False

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    global abort
    print("\n > Caught Signal. Exiting!\n")
    abort = True
    raise SystemExit

class SentinelDownloader:
    def __init__(self, url, output_dir='./data'):
        # URL to download
        self.url = url
        self.output_dir = output_dir
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Cookie handling
        self.cookie_jar_path = os.path.join(
            os.path.expanduser('~'),
            ".sentinel_cookiejar.txt"
        )
        self.cookie_jar = None
        
        # ASF auth settings
        self.asf_urs4 = {
            'url': 'https://urs.earthdata.nasa.gov/oauth/authorize',
            'client': 'BO_n7nTIlMljdvU6kRRB3g',
            'redir': 'https://auth.asf.alaska.edu/login'
        }
        
        # SSL context
        self.context = {}
        
        # Set up and validate cookie
        self.get_cookie()
        
    def get_cookie(self):
        """Get and validate cookie for NASA Earthdata access"""
        if os.path.isfile(self.cookie_jar_path):
            self.cookie_jar = MozillaCookieJar()
            self.cookie_jar.load(self.cookie_jar_path)
            
            if self.check_cookie():
                print(" > Reusing previous cookie jar.")
                return
            else:
                print(" > Could not validate old cookie jar")
        
        # We need a new cookie, prompt for credentials
        print("No existing URS cookie found, please enter Earthdata username & password:")
        print("(Credentials will not be stored, saved or logged anywhere)")
        
        # Keep trying until we get valid credentials
        while not self.check_cookie():
            self.get_new_cookie()
    
    def check_cookie(self):
        """Validate the cookie against NASA Earthdata"""
        if self.cookie_jar is None:
            return False
        
        # File we know is valid, used to validate cookie
        file_check = 'https://urs.earthdata.nasa.gov/profile'
        
        # Apply custom Redirect Handler
        opener = build_opener(
            HTTPCookieProcessor(self.cookie_jar),
            HTTPHandler(),
            HTTPSHandler(**self.context)
        )
        install_opener(opener)
        
        # Attempt a HEAD request
        request = Request(file_check)
        request.get_method = lambda: 'HEAD'
        try:
            response = urlopen(request, timeout=30)
            resp_code = response.getcode()
            
            # Make sure we're logged in
            if not self.check_cookie_is_logged_in(self.cookie_jar):
                return False
            
            # Save cookie jar
            self.cookie_jar.save(self.cookie_jar_path)
            
        except HTTPError:
            # User may not have permissions
            print("\nIMPORTANT: ")
            print("Your user appears to lack permissions to download data from the ASF Datapool.")
            print("\n\nNew users: you must first log into Vertex and accept the EULA. In addition, your Study Area must be set at Earthdata https://urs.earthdata.nasa.gov")
            return False
        
        # Check redirect codes
        if resp_code in (300, 301, 302, 303):
            print(f"Redirect ({resp_code}) occurred, invalid cookie value!")
            return False
        
        # Success codes
        if resp_code in (200, 307):
            return True
        else:
            return False
    
    def get_new_cookie(self):
        """Get a new authentication cookie"""
        # Prompt for credentials
        new_username = input("Username: ")
        new_password = getpass.getpass(prompt="Password (will not be displayed): ")
        
        # Build URS4 Cookie request
        auth_cookie_url = f"{self.asf_urs4['url']}?client_id={self.asf_urs4['client']}&redirect_uri={self.asf_urs4['redir']}&response_type=code&state="
        
        user_pass = base64.b64encode(bytes(new_username + ":" + new_password, "utf-8"))
        user_pass = user_pass.decode("utf-8")
        
        # Authenticate against URS, grab all cookies
        self.cookie_jar = MozillaCookieJar()
        opener = build_opener(HTTPCookieProcessor(self.cookie_jar), HTTPHandler(), HTTPSHandler(**self.context))
        request = Request(auth_cookie_url, headers={"Authorization": f"Basic {user_pass}"})
        
        # Handle authentication response
        try:
            response = opener.open(request)
        except HTTPError as e:
            if "WWW-Authenticate" in e.headers and "Please enter your Earthdata Login credentials" in e.headers["WWW-Authenticate"]:
                print(" > Username and Password combo was not successful. Please try again.")
                return False
            else:
                # If an error happens here, the user most likely has not confirmed EULA
                print("\nIMPORTANT: There was an error obtaining a download cookie!")
                print("Your user appears to lack permission to download data from the ASF Datapool.")
                print("\n\nNew users: you must first log into Vertex and accept the EULA. In addition, your Study Area must be set at Earthdata https://urs.earthdata.nasa.gov")
                return False
        except URLError as e:
            print("\nIMPORTANT: There was a problem communicating with URS, unable to obtain cookie.")
            print("Try cookie generation later.")
            return False
        
        # Did we get a cookie?
        if self.check_cookie_is_logged_in(self.cookie_jar):
            self.cookie_jar.save(self.cookie_jar_path)
            return True
        
        # If we aren't successful generating the cookie, nothing will work
        print("WARNING: Could not generate new cookie! Cannot proceed. Please try Username and Password again.")
        print(f"Response was {response.getcode()}.")
        print("\n\nNew users: you must first log into Vertex and accept the EULA. In addition, your Study Area must be set at Earthdata https://urs.earthdata.nasa.gov")
        return False
    
    def check_cookie_is_logged_in(self, cj):
        """Check if we're logged into URS"""
        for cookie in cj:
            if cookie.name == 'urs_user_already_logged':
                # Only get this cookie if we logged in successfully!
                return True
        return False
    
    def download_file_with_cookiejar(self, url=None, recursion=False):
        """Download file with progress reporting"""
        if url is None:
            url = self.url
        
        # Get filename from URL
        download_file = os.path.join(self.output_dir, os.path.basename(url).split('?')[0])
        
        # Check if file already exists and has correct size
        if os.path.isfile(download_file):
            try:
                request = Request(url)
                request.get_method = lambda: 'HEAD'
                response = urlopen(request, timeout=30)
                remote_size = self.get_total_size(response)
                
                if remote_size:
                    local_size = os.path.getsize(download_file)
                    if remote_size < (local_size + (local_size * .01)) and remote_size > (local_size - (local_size * .01)):
                        print(f" > Download file {download_file} exists!")
                        print(f" > Skipping download of {url}.")
                        return local_size, remote_size
                    
                    # Partial file size wasn't full file size, delete and restart
                    print(f" > Found {download_file} but it wasn't fully downloaded. Removing file and downloading again.")
                    os.remove(download_file)
                    
            except ssl.CertificateError as e:
                print(f" > ERROR: {e}")
                print(" > Could not validate SSL Cert. You may be able to overcome this using the --insecure flag")
                return False, None
            
            except HTTPError as e:
                if e.code == 401:
                    print(" > IMPORTANT: Your user may not have permission to download this type of data!")
                else:
                    print(f" > Unknown Error, Could not get file HEAD: {e}")
                
            except URLError as e:
                print(f"URL Error (from HEAD): {e.reason}, {url}")
                if "ssl.c" in f"{e.reason}":
                    print("IMPORTANT: Remote location may not be accepting your SSL configuration. This is a terminal error.")
                return False, None
        
        # Attempt HTTPS connection for download
        try:
            request = Request(url)
            response = urlopen(request, timeout=30)
            
            # Wait for burst extraction if needed
            while response.getcode() == 202:
                print(" > Waiting for burst extraction service...")
                time.sleep(5)
                response = urlopen(request, timeout=30)
            
            # Watch for redirect
            if response.geturl() != url:
                # Check if redirected to URS for re-auth
                if 'https://urs.earthdata.nasa.gov/oauth/authorize' in response.geturl():
                    if recursion:
                        print(" > Entering seemingly endless auth loop. Aborting.")
                        return False, None
                    
                    # Make URL easier to handle - add app_type if missing
                    new_auth_url = response.geturl()
                    if "app_type" not in new_auth_url:
                        new_auth_url += "&app_type=401"
                    
                    print(f" > While attempting to download {url}....")
                    print(f" > Need to obtain new cookie from {new_auth_url}")
                    
                    old_cookies = [cookie.name for cookie in self.cookie_jar]
                    opener = build_opener(HTTPCookieProcessor(self.cookie_jar), HTTPHandler(), HTTPSHandler(**self.context))
                    request = Request(new_auth_url)
                    
                    try:
                        response = opener.open(request)
                        for cookie in self.cookie_jar:
                            if cookie.name not in old_cookies:
                                print(f" > Saved new cookie: {cookie.name}")
                                
                                # Save session cookies that would normally be discarded
                                if cookie.discard:
                                    cookie.expires = int(time.time()) + 60 * 60 * 24 * 30
                                    print(" > Saving session Cookie that should have been discarded!")
                        
                        self.cookie_jar.save(self.cookie_jar_path, ignore_discard=True, ignore_expires=True)
                        
                    except HTTPError as e:
                        print(f"HTTP Error: {e.code}, {url}")
                        return False, None
                    
                    # Try again with new cookies
                    print(" > Attempting download again with new cookies!")
                    return self.download_file_with_cookiejar(url, recursion=True)
                
                print(f" > 'Temporary' Redirect download @ Remote archive:\n > {response.geturl()}")
            
            # Start download
            print(f"Downloading {url}")
            
            # Check for content disposition to get filename
            content_disposition = response.headers.get('Content-Disposition')
            if content_disposition and len(content_disposition):
                possible_filename = re.findall("filename=(\S+)", content_disposition)
                if possible_filename:
                    download_file = os.path.join(self.output_dir, possible_filename.pop())
            
            # Get file size
            file_size = self.get_total_size(response)
            
            # Open output file for writing
            with open(download_file, 'wb') as f:
                bytes_so_far = 0
                chunk_size = 8192
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    bytes_so_far += len(chunk)
                    
                    # Report progress
                    if file_size:
                        percent = float(bytes_so_far) / file_size
                        percent = round(percent * 100, 2)
                        sys.stdout.write(f" > Downloaded {bytes_so_far} of {file_size} bytes ({percent:0.2f}%)\r")
                    else:
                        sys.stdout.write(f" > Downloaded {bytes_so_far} of unknown Size\r")
                    
                    sys.stdout.flush()
            
            # Reset download status
            sys.stdout.write('\n')
            
        except HTTPError as e:
            print(f"HTTP Error: {e.code}, {url}")
            
            if e.code == 401:
                print(" > IMPORTANT: Your user does not have permission to download this type of data!")
            
            if e.code == 403:
                print(" > Got a 403 Error trying to download this file.")
                print(" > You MAY need to log in this app and agree to a EULA.")
            
            return False, None
        
        except URLError as e:
            print(f"URL Error (from GET): {e}, {e.reason}, {url}")
            
            if "ssl.c" in f"{e.reason}":
                print("IMPORTANT: Remote location may not be accepting your SSL configuration. This is a terminal error.")
            
            return False, None
        
        except ssl.CertificateError as e:
            print(f" > ERROR: {e}")
            print(" > Could not validate SSL Cert. You may be able to overcome this using the --insecure flag")
            return False, None
        
        except Exception as e:
            print(f"Error downloading file: {str(e)}")
            return False, None
        
        # Verify the download was successful
        if os.path.exists(download_file):
            actual_size = os.path.getsize(download_file)
            
            # If we couldn't get file_size from headers, use actual size
            if file_size is None:
                file_size = actual_size
                
            return actual_size, file_size
        else:
            print(f"Failed to save file to {download_file}")
            return False, None
    
    def get_total_size(self, response):
        """Get file size from response headers"""
        try:
            file_size = response.info().get('Content-Length')
            if file_size:
                return int(file_size.strip())
            return None
        except:
            return None
    
    def download(self):
        """Download the file and print status"""
        # Start timer
        start = time.time()
        
        # Run download
        size, total_size = self.download_file_with_cookiejar()
        
        # Calculate rate
        end = time.time()
        
        # Status report
        if size is None:
            print("File already exists, skipped download")
            # Return path to the existing file
            return os.path.join(self.output_dir, os.path.basename(self.url).split('?')[0])
        
        elif size and total_size and size > 0:  # More robust check
            elapsed = end - start
            elapsed = 1.0 if elapsed < 1 else elapsed
            rate = (size / 1024**2) / elapsed
            
            print(f"Successfully downloaded {size} bytes in {elapsed:.2f} seconds")
            print(f"Average download rate: {rate:.2f} MB/sec")
            filename = os.path.basename(self.url).split('?')[0]
            downloaded_path = os.path.join(self.output_dir, filename)
            print(f"Downloaded file: {downloaded_path}")
            return downloaded_path
        
        else:
            print(f"There was a problem downloading {self.url}")
            return None

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Download a Sentinel-1 file from ASF')
    parser.add_argument('url', help='URL of the Sentinel-1 file to download')
    parser.add_argument('--output_dir', default='./data', help='Directory to save the file')
    
    args = parser.parse_args()
    
    # Download the file
    downloader = SentinelDownloader(args.url, args.output_dir)
    output_path = downloader.download()
    
    if output_path and os.path.exists(output_path):
        return 0
    else:
        return 1

if __name__ == "__main__":
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    sys.exit(main())
