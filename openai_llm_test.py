import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Show current working directory
print(f"Current working directory: {os.getcwd()}")
print("-" * 50)

# Check if .env file exists
env_path = Path('.env')
if env_path.exists():
    print(f"[+] .env file found at: {env_path.absolute()}")
    # Show .env file contents (be careful with this in production!)
    with open('.env', 'r') as f:
        print("\n.env file contents:")
        print(f.read())
else:
    print("[-] .env file not found in current directory")
    print(f"Looking for: {env_path.absolute()}")

print("-" * 50)

# Load environment variables from .env file
load_dotenv(override=True)  # override=True ensures .env values take priority

# Get API key from environment
api_key = os.getenv('OPENAI_API_KEY')

# Print the API key
print(f"\nAPI Key loaded: {api_key}")
print(f"API Key length: {len(api_key) if api_key else 0}")
print("-" * 50)

# Check if API key exists
if not api_key:
    print("[-] Error: OPENAI_API_KEY not found in .env file")
    exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

try:
    # Test API by making a simple completion request
    print("Testing OpenAI API...")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Say 'API is working!' if you receive this message."}
        ],
        max_tokens=20
    )
    
    # Print success message
    print("[+] OpenAI API is working!")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model used: {response.model}")
    
except Exception as e:
    print(f"[-] Error: OpenAI API is not working")
    print(f"Error details: {str(e)}")