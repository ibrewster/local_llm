import requests

# Replace with your actual server URL
URL = "http://127.0.0.1:11434/generate"

# The payload as required by your endpoint
payload = {
#    "prompt": "Generate two or three sentences just to ensure that the code pathway is working. Output just the sentences. Make the last sentence be literally 'That's all folks', so I can be sure the entire response came through.",
    "prompt": "It is bedtime. The current weather is 22°C with a light breeze and clear skies.",
#    "system": "You are a calm and soothing bedtime storyteller."
}

def test_endpoint():
    try:
        # Sending the POST request with the JSON payload
        response = requests.post(URL, json=payload)

        # Output results
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")

        if response.status_code == 202:
            print("✅ Test Passed: Request accepted.")
        else:
            print("❌ Test Failed: Unexpected status code.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    test_endpoint()